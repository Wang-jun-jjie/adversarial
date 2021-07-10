import apex.amp as amp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

def get_loaders(data_directory, batch_size, image_size, image_crop):
    print('==> Preparing ImageNet dataset..')
    train_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.RandomResizedCrop(image_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(imagenet_mean, imagenet_std), 
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.CenterCrop(image_crop),
        transforms.ToTensor(),
        # transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    train_dataset = datasets.ImageFolder(root=data_directory+'/train', \
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,\
        shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    test_dataset = datasets.ImageFolder(root=data_directory+'/val', \
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,\
        shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    return train_loader, test_loader

def normalize(image, batch_size):
    mean = torch.tensor(imagenet_mean).reshape(1, 3, 1, 1).repeat(batch_size, 1, 1, 1).to(device)
    std = torch.tensor(imagenet_std).reshape(1, 3, 1, 1).repeat(batch_size, 1, 1, 1).to(device)
    return (image-mean)/ std

class deforming_medium(nn.Module):
    def __init__(self, args):
        super(deforming_medium, self).__init__()
        self.args = args
        # args
        self.identity_mean, self.identity_offset = 0.5, 4.0
        self.deforming_offset = (2.0 + (self.identity_offset/(args.image_crop-1))) / 2.0
        # accumulate filter
        self.accu_filter_x = torch.ones((1, 1, 1, args.image_crop), requires_grad=False).to(device)
        self.accu_filter_y = torch.ones((1, 1, args.image_crop, 1), requires_grad=False).to(device)
        # deviation filter
        self.dvia_filter_x = torch.cat((torch.full((1, 1, 1, 1), 1.), torch.full((1, 1, 1, 1), -1.)), 3).to(device)
        self.dvia_filter_y = torch.cat((torch.full((1, 1, 1, 1), 1.), torch.full((1, 1, 1, 1), -1.)), 2).to(device)
        # identity sampling grid
        self.samp_iden = self.init_samp_grid().to(device)
    
    def init_accu_grid(self):
        # output shape (N, 2. H, W)
        accu_grid = torch.full((self.args.batch_size, 2, self.args.image_crop, self.args.image_crop), self.identity_mean)
        return accu_grid

    def init_prim_grid(self):
        # output shape (N, 2. H, W)
        prim_grid = torch.zeros(self.args.batch_size, 2, self.args.image_crop, self.args.image_crop)
        return prim_grid

    def perturb_prim_grid(self, prim_grid):
        size = (2, self.args.image_crop, self.args.image_crop)
        pixel_width = 2.0/(self.args.image_crop)
        prim_grid += (2*torch.rand(size)-1)*pixel_width*self.args.epsilon
        return prim_grid
    
    def init_samp_grid(self):
        # imageWidth must equal to imageHeight
        # output shape (N, 2. H, W)
        sequence = torch.arange(-(self.args.image_crop-1), (self.args.image_crop), 2)/(self.args.image_crop-1.0)
        samp_grid_x = sequence.repeat(self.args.image_crop,1)
        samp_grid_y = samp_grid_x.t()

        samp_grid = torch.cat((samp_grid_x.unsqueeze(0), samp_grid_y.unsqueeze(0)), 0)
        samp_grid = samp_grid.unsqueeze(0).repeat(self.args.batch_size, 1, 1, 1)
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
        accu_grid_x = accu_grid[:,0:1,:,:]*(self.identity_offset/(self.args.image_crop-1))
        accu_grid_y = accu_grid[:,1:2,:,:]*(self.identity_offset/(self.args.image_crop-1))
        # accumulation grid: N, 2, H, W
        accu_grid = torch.cat((accu_grid_x, accu_grid_y), 1)
        # Summation along channel X and Y
        samp_grid_x =  F.conv_transpose2d(accu_grid[:,0:1,:,:], self.accu_filter_x, stride=1, padding=0)
        samp_grid_y =  F.conv_transpose2d(accu_grid[:,1:2,:,:], self.accu_filter_y, stride=1, padding=0)
        samp_grid = torch.cat((samp_grid_x[:,:,0:self.args.image_crop,0:self.args.image_crop], \
                               samp_grid_y[:,:,0:self.args.image_crop,0:self.args.image_crop]), 1)
        # adding offset
        samp_grid = samp_grid-self.deforming_offset
        return samp_grid
    
    def samp_grid_2_accu_grid(self, samp_grid):
        # remove offset
        samp_grid = samp_grid+self.deforming_offset
        # deviate it back
        accu_grid_x =  F.conv_transpose2d(samp_grid[:,0:1,:,:], self.dvia_filter_x, stride=1, padding=0)
        accu_grid_y =  F.conv_transpose2d(samp_grid[:,1:2,:,:], self.dvia_filter_y, stride=1, padding=0)
        accu_grid = torch.cat((accu_grid_x[:,:,0:self.args.image_crop,0:self.args.image_crop], \
                               accu_grid_y[:,:,0:self.args.image_crop,0:self.args.image_crop]), 1)
        accu_grid_x = accu_grid[:,0:1,:,:]/(self.identity_offset/(self.args.image_crop-1))
        accu_grid_y = accu_grid[:,1:2,:,:]/(self.identity_offset/(self.args.image_crop-1))
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
        pixel_width = 2.0/(self.args.image_crop)
        prim_grid = torch.clamp(prim_grid, -pixel_width*self.args.epsilon, pixel_width*self.args.epsilon)
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
        prim_grid = self.gaussian_blur(prim_grid, self.args.kernel_size)
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