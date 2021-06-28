# pytorch related package 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

print('pytorch version: ' + torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NVIDIA apex
from apex import amp

# only allow image width = hight
class deforming_medium(nn.Module):
    def __init__(self, opts):
        super(deforming_medium, self).__init__()
        self.opts = opts
        # args
        self.identity_mean, self.identity_offset = 0.5, 4.0
        self.deforming_offset = (2.0 + (self.identity_offset/(opts['imageWidth']-1))) / 2.0
        # accumulate filter
        self.accu_filter_x = torch.ones((1, 1, 1, opts['imageWidth']), requires_grad=False).to(device)
        self.accu_filter_y = torch.ones((1, 1, opts['imageHeight'], 1), requires_grad=False).to(device)
        # deviation filter
        self.dvia_filter_x = torch.cat((torch.full((1, 1, 1, 1), 1.), torch.full((1, 1, 1, 1), -1.)), 3).to(device)
        self.dvia_filter_y = torch.cat((torch.full((1, 1, 1, 1), 1.), torch.full((1, 1, 1, 1), -1.)), 2).to(device)
        # identity sampling grid
        self.samp_iden = self.init_samp_grid().to(device)
    
    def init_accu_grid(self):
        # output shape (N, 2. H, W)
        accu_grid = torch.full((self.opts['batchSize'], 2, self.opts['imageHeight'], self.opts['imageWidth']), self.identity_mean)
        return accu_grid

    def init_prim_grid(self):
        # output shape (N, 2. H, W)
        prim_grid = torch.zeros(self.opts['batchSize'], 2, self.opts['imageHeight'], self.opts['imageWidth'])
        return prim_grid

    def perturb_prim_grid(self, prim_grid):
        size = (2, self.opts['imageHeight'], self.opts['imageWidth'])
        pixel_width = 2.0/(self.opts['imageWidth'])
        prim_grid += (2*torch.rand(size)-1)*pixel_width*self.opts['epsilon']
        return prim_grid
    
    def init_samp_grid(self):
        # imageWidth must equal to imageHeight
        # output shape (N, 2. H, W)
        sequence = torch.arange(-(self.opts['imageWidth']-1), (self.opts['imageWidth']), 2)/(self.opts['imageWidth']-1.0)
        samp_grid_x = sequence.repeat(self.opts['imageHeight'],1)
        samp_grid_y = samp_grid_x.t()

        samp_grid = torch.cat((samp_grid_x.unsqueeze(0), samp_grid_y.unsqueeze(0)), 0)
        samp_grid = samp_grid.unsqueeze(0).repeat(self.opts['batchSize'], 1, 1, 1)
        return samp_grid
    
    def get_gaussian_kernel2d(self, kernel_size, sigma):
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()
        kernel2d = torch.mm(kernel1d[:, None], kernel1d[None, :])
        return kernel2d
    
    def gaussian_blur(self, prim_grid, kernel_size, sigma=None):
        # kernel_size should have odd and positive integers
        if sigma is None:
            sigma = kernel_size * 0.15 + 0.35
        kernel = self.get_gaussian_kernel2d(kernel_size, sigma).to(device)
        kernel = kernel.expand(prim_grid.shape[-3], 1, kernel.shape[0], kernel.shape[1])
        # padding = (left, right, top, bottom)
        padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
        prim_grid = F.pad(prim_grid, padding, mode="reflect")
        prim_grid = F.conv2d(prim_grid, kernel, groups=prim_grid.shape[-3])
        return prim_grid
    
    def accu_grid_2_samp_grid(self, accu_grid):
        accu_grid_x = accu_grid[:,0:1,:,:]*(self.identity_offset/(self.opts['imageWidth']-1))
        accu_grid_y = accu_grid[:,1:2,:,:]*(self.identity_offset/(self.opts['imageHeight']-1))
        # accumulation grid: N, 2, H, W
        accu_grid = torch.cat((accu_grid_x, accu_grid_y), 1)
        # Summation along channel X and Y
        samp_grid_x =  F.conv_transpose2d(accu_grid[:,0:1,:,:], self.accu_filter_x, stride=1, padding=0)
        samp_grid_y =  F.conv_transpose2d(accu_grid[:,1:2,:,:], self.accu_filter_y, stride=1, padding=0)
        samp_grid = torch.cat((samp_grid_x[:,:,0:self.opts['imageHeight'],0:self.opts['imageWidth']], \
                               samp_grid_y[:,:,0:self.opts['imageHeight'],0:self.opts['imageWidth']]), 1)
        # adding offset
        samp_grid = samp_grid-self.deforming_offset
        return samp_grid
    
    def samp_grid_2_accu_grid(self, samp_grid):
        # remove offset
        samp_grid = samp_grid+self.deforming_offset
        # deviate it back
        accu_grid_x =  F.conv_transpose2d(samp_grid[:,0:1,:,:], self.dvia_filter_x, stride=1, padding=0)
        accu_grid_y =  F.conv_transpose2d(samp_grid[:,1:2,:,:], self.dvia_filter_y, stride=1, padding=0)
        accu_grid = torch.cat((accu_grid_x[:,:,0:self.opts['imageHeight'],0:self.opts['imageWidth']], \
                               accu_grid_y[:,:,0:self.opts['imageHeight'],0:self.opts['imageWidth']]), 1)
        accu_grid_x = accu_grid[:,0:1,:,:]/(self.identity_offset/(self.opts['imageWidth']-1))
        accu_grid_y = accu_grid[:,1:2,:,:]/(self.identity_offset/(self.opts['imageHeight']-1))
        accu_grid = torch.cat((accu_grid_x, accu_grid_y), 1)
        return accu_grid
    
    def prim_grid_2_samp_grid(self, prim_grid):
        samp_grid = prim_grid + self.samp_iden
        return samp_grid
    
    def samp_grid_2_prim_grid(self, samp_grid):
        prim_grid = samp_grid - self.samp_iden
        return prim_grid

    def accu_clip(self, accu_grid):
        return F.relu(accu_grid, inplace=True)
    
    def prim_clip(self, prim_grid):
        pixel_width = 2.0/(self.opts['imageWidth'])
        prim_grid = torch.clamp(prim_grid, -pixel_width*self.opts['epsilon'], pixel_width*self.opts['epsilon'])
        return prim_grid
    
    def samp_clip(self, samp_grid):
        return F.hardtanh(samp_grid)
    
    def clip(self, prim_grid):
        # clip by attack budget epsilon
        prim_grid = self.prim_clip(prim_grid)
        # clip by image constrain
        accu_grid = self.samp_grid_2_accu_grid(self.prim_grid_2_samp_grid(prim_grid))
        accu_grid = self.accu_clip(accu_grid)
        samp_grid = self.accu_grid_2_samp_grid(accu_grid)
        samp_grid = self.samp_clip(samp_grid)
        prim_grid = self.samp_grid_2_prim_grid(samp_grid)
        return prim_grid

    def forward_grid(self, prim_grid):
        prim_grid = self.gaussian_blur(prim_grid, self.opts['kernelSize'])
        accu_grid = self.samp_grid_2_accu_grid(self.prim_grid_2_samp_grid(prim_grid))
        accu_grid = self.accu_clip(accu_grid)
        prim_grid = self.samp_grid_2_prim_grid(self.accu_grid_2_samp_grid(accu_grid))
        prim_grid = self.prim_clip(prim_grid)
        samp_grid = self.prim_grid_2_samp_grid(prim_grid)
        samp_grid = self.samp_clip(samp_grid)
        return samp_grid
    
    def get_grid_in_effect(self, prim_grid):
        samp_grid = self.forward_grid(prim_grid)
        prim_grid = self.samp_grid_2_prim_grid(samp_grid)
        return prim_grid
    
    def forward(self, image, prim_grid):
        samp_grid = self.forward_grid(prim_grid)

        # binding
        binding_grid = samp_grid.permute(0,2,3,1)
        distort_image = F.grid_sample(image, binding_grid, align_corners=True)
        return distort_image

