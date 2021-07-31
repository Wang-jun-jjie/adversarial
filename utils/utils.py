import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from utils.augment import RandAugment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

_normalize = transforms.Normalize(
    mean=imagenet_mean, std=imagenet_std)
_inv_normalize = transforms.Normalize(
    mean= [-m/s for m, s in zip(imagenet_mean, imagenet_std)],
    std= [1/s for s in imagenet_std])
def normalize(x):
    return _normalize(x)
def inv_normalize(x):
    return _inv_normalize(x)

def get_loaders(data_directory, batch_size, augment=True, N=2, M=9): # only support imagenet-size image
    print('==> Preparing dataset..')
    # move normalize into model, don't normalize here, 
    # is better for classic adversarial attacks
    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(imagenet_mean, imagenet_std), 
    ])
    test_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    
    if augment:
        # Add RandAugment with N, M(hyperparameter)
        train_transform.transforms.insert(0, RandAugment(N, M))

    train_dataset = datasets.ImageFolder(root=data_directory+'/train', \
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,\
        shuffle=True, drop_last=True, num_workers=12, pin_memory=True)
    test_dataset = datasets.ImageFolder(root=data_directory+'/val', \
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,\
        shuffle=True, drop_last=True, num_workers=12, pin_memory=True)
    return train_loader, test_loader

class Normalize_tops(nn.Module):
    def __init__(self, mean=imagenet_mean, std=imagenet_std):
        super(Normalize_tops, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
    
    def forward(self, x):
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (x-mean) / std

# TODO: change it to the buffer
class deforming_medium(nn.Module):
    def __init__(self, args):
        super(deforming_medium, self).__init__()
        self.args = args
        # args
        self.identity_mean, self.identity_offset = 0.5, 4.0
        self.deforming_offset = (2.0 + (self.identity_offset/(args.image_size-1))) / 2.0
        # accumulate filter
        self.accu_filter_x = torch.ones((1, 1, 1, args.image_size), requires_grad=False).to(device)
        self.accu_filter_y = torch.ones((1, 1, args.image_size, 1), requires_grad=False).to(device)
        # deviation filter
        self.dvia_filter_x = torch.cat((torch.full((1, 1, 1, 1), 1.), torch.full((1, 1, 1, 1), -1.)), 3).to(device)
        self.dvia_filter_y = torch.cat((torch.full((1, 1, 1, 1), 1.), torch.full((1, 1, 1, 1), -1.)), 2).to(device)
        # identity sampling grid
        self.samp_iden = self.init_samp_grid().to(device)
    
    def init_accu_grid(self):
        # output shape (N, 2. H, W)
        accu_grid = torch.full((self.args.batch_size, 2, self.args.image_size, self.args.image_size), self.identity_mean)
        return accu_grid

    def init_prim_grid(self):
        # output shape (N, 2. H, W)
        prim_grid = torch.zeros(self.args.batch_size, 2, self.args.image_size, self.args.image_size)
        return prim_grid

    def perturb_prim_grid(self, prim_grid):
        size = (2, self.args.image_size, self.args.image_size)
        pixel_width = 2.0/(self.args.image_size)
        prim_grid += (2*torch.rand(size)-1)*pixel_width*self.args.epsilon
        return prim_grid
    
    def init_samp_grid(self):
        # imageWidth must equal to imageHeight
        # output shape (N, 2. H, W)
        sequence = torch.arange(-(self.args.image_size-1), (self.args.image_size), 2)/(self.args.image_size-1.0)
        samp_grid_x = sequence.repeat(self.args.image_size,1)
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
        accu_grid_x = accu_grid[:,0:1,:,:]*(self.identity_offset/(self.args.image_size-1))
        accu_grid_y = accu_grid[:,1:2,:,:]*(self.identity_offset/(self.args.image_size-1))
        # accumulation grid: N, 2, H, W
        accu_grid = torch.cat((accu_grid_x, accu_grid_y), 1)
        # Summation along channel X and Y
        samp_grid_x =  F.conv_transpose2d(accu_grid[:,0:1,:,:], self.accu_filter_x, stride=1, padding=0)
        samp_grid_y =  F.conv_transpose2d(accu_grid[:,1:2,:,:], self.accu_filter_y, stride=1, padding=0)
        samp_grid = torch.cat((samp_grid_x[:,:,0:self.args.image_size,0:self.args.image_size], \
                               samp_grid_y[:,:,0:self.args.image_size,0:self.args.image_size]), 1)
        # adding offset
        samp_grid = samp_grid-self.deforming_offset
        return samp_grid
    
    def samp_grid_2_accu_grid(self, samp_grid):
        # remove offset
        samp_grid = samp_grid+self.deforming_offset
        # deviate it back
        accu_grid_x =  F.conv_transpose2d(samp_grid[:,0:1,:,:], self.dvia_filter_x, stride=1, padding=0)
        accu_grid_y =  F.conv_transpose2d(samp_grid[:,1:2,:,:], self.dvia_filter_y, stride=1, padding=0)
        accu_grid = torch.cat((accu_grid_x[:,:,0:self.args.image_size,0:self.args.image_size], \
                               accu_grid_y[:,:,0:self.args.image_size,0:self.args.image_size]), 1)
        accu_grid_x = accu_grid[:,0:1,:,:]/(self.identity_offset/(self.args.image_size-1))
        accu_grid_y = accu_grid[:,1:2,:,:]/(self.identity_offset/(self.args.image_size-1))
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
        pixel_width = 2.0/(self.args.image_size)
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

# adversarial attacks
# modify from https://github.com/metehancekic/deep-illusion

# https://github.com/Harry24k/adversarial-attacks-pytorch

def FGSM(model, data, target, eps):
    """
    Fast Gradient Sign Method
    model:      Neural Network to attack
    data:       Input batches
    target:     True labels
    eps:        Attack budget
    """
    data = data.clone().detach().to(device)
    target = target.clone().detach().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    data.requires_grad = True
    output = model(data)

    loss = criterion(output, target)
    # Update adversarial images
    grad = torch.autograd.grad(loss, data, \
        retain_graph=False, create_graph=False)[0]
    adv = data + eps*grad.sign()
    adv = torch.clamp(adv, min=0, max=1).detach()
    return adv

def IGSM(model, data, target, eps, alpha, iter=0):
    '''
    Iterative Gradient Sign Method
    model:      Neural Network to attack
    data:       Input batches
    target:     True labels
    eps:        Attack budget
    alpha:      step per iteration
    iter:       Number of iteration
    '''
    if iter == 0:
        iter = int(min(eps*255 + 4, 1.25*eps*255))
    data = data.clone().detach().to(device)
    target = target.clone().detach().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    data_ori = data.clone().detach()

    for i in range(iter):
        data.requires_grad = True
        output = model(data)
        loss = criterion(output, target)
        # Update adversarial images
        grad = torch.autograd.grad(loss, data, \
            retain_graph=False, create_graph=False)[0]
        adv = data + alpha*grad.sign()
        a = torch.clamp(data_ori - eps, min=0)
        b = (adv >= a).float()*adv \
            + (adv < a).float()*a
        c = (b > data_ori+eps).float()*(data_ori+eps) \
            + (b <= data_ori + eps).float()*b
        adv = torch.clamp(c, max=1).detach()

    return adv

def PGD(model, data, target, eps, alpha, iter=20):
    '''
    Project Gradient Descent
    model:      Neural Network to attack
    data:       Input batches
    target:     True labels
    eps:        Attack budget
    alpha:      step per iteration
    iter:       Number of iteration
    '''
    data = data.clone().detach().to(device)
    target = target.clone().detach().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    adv = data.clone().detach()
    # Starting at a uniformly random point
    adv = adv + torch.empty_like(adv).uniform_(-eps, eps)
    adv = torch.clamp(adv, min=0, max=1).detach()

    for i in range(iter):
        adv.requires_grad = True
        output = model(adv)
        loss = criterion(output, target)
        # Update adversarial images
        grad = torch.autograd.grad(loss, adv, \
            retain_graph=False, create_graph=False)[0]
        adv = adv.detach() + alpha*grad.sign()
        delta = torch.clamp(adv-data, min=-eps, max=eps)
        adv = torch.clamp(data+delta, min=0, max=1).detach()
    return adv, delta

def AET(model, warper, data, target, step, iter=20):
    '''
    Accumulated Elastic Transform
    model:      Neural Network to attack
    data:       Input batches
    target:     True labels
    eps:        Attack budget
    step:       step size per iteration
    iter:       Number of iteration
    '''
    data = data.clone().detach().to(device)
    grid = warper.init_prim_grid().detach().to(device)
    target = target.clone().detach().to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    for i in range(iter):
        grid.requires_grad = True
        adv = warper(data, grid)
        output = model(adv)
        loss = criterion(output, target)
        # Update adversarial images
        grad = torch.autograd.grad(loss, grid, \
            retain_graph=False, create_graph=False)[0]
        grid = grid + step*grad.sign()
        grid = grid.detach()
    # warp it back to image
    adv = warper(data, grid)
    return adv
