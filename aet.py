import argparse
import logging
import time
# select GPU on the server
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# pytorch related package 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

# print('pytorch version: ' + torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NVIDIA apex
from apex import amp
# math and showcase
import numpy as np

from utils import *

parser = argparse.ArgumentParser( description='Adversarial attacks')
parser.add_argument('--resume', '-r',       action='store_true',              help='resume from checkpoint')
parser.add_argument('--prefix',             default='Adversarial attacks',    type=str,   help='prefix used to define logs')
parser.add_argument('--seed',               default=59572406,     type=int,   help='random seed')
parser.add_argument('--batch-size', '-b',   default=120,          type=int,   help='mini-batch size (default: 120)')
parser.add_argument('--target', '-t',       default=None,         type=int,   help='adversarial attack target label')
parser.add_argument('--rnd-target', '--rt', action='store_true',              help='non-target attack using random label as target')
parser.add_argument('--iteration', '-i',    default=20,           type=int,   help='adversarial attack iterations (default: 20)')
parser.add_argument('--step-size', '--ss',  default=0.005,        type=float, help='step size for adversarial attacks')
parser.add_argument('--epsilon', '-e',      default=1,            type=float, help='epsilon for adversarial attacks')
parser.add_argument('--kernel-size', '-k',  default=13,           type=int,   help='kernel size for adversarial attacks, must be odd integer')
parser.add_argument('--image-size', '--is', default=256,          type=int,   help='resize input image (default: 256 for ImageNet)')
parser.add_argument('--image-crop', '--ic', default=224,          type=int,   help='centercrop input image after resize (default: 224 for ImageNet)')
parser.add_argument('--data-directory',     default='../ImageNet',type=str,   help='dataset inputs root directory')
parser.add_argument('--data-classname',     default='../ImageNet/LOC_synset_mapping.txt',type=str, help='dataset classname file')
parser.add_argument('--opt-level', '-o',    default='O1',         type=str,   help='Nvidia apex optimation level (default: O1)')
args = parser.parse_args()

# only allow image width = hight
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

def main():
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # load dataset (Imagenet)
    train_loader, test_loader = get_loaders(args.data_directory, args.batch_size, \
                                            args.image_size, args.image_crop)
    # load the class label (Imagenet)
    # classPath = args.data_classname
    # classes = list()
    # with open(classPath) as class_file:
    #     for line in class_file:
    #         class_name = line[10:].strip().split(',')[0]
    #         classes.append(class_name)
    # classes = tuple(classes)

    # Load model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        model = models.resnet50(pretrained=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

        checkpoint = torch.load('./checkpoint/' + args.sess + '_' + str(args.seed) + '.pth')
        prev_acc = checkpoint['acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        amp.load_state_dict(checkpoint['amp_state_dict'])
        epoch_start = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    else:
        print('==> Building model..')
        epoch_start = 0
        prev_acc = 0.0
        model = models.resnet50(pretrained=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    warper = deforming_medium(args)
    # Logger
    result_folder = './results/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    logger = logging.getLogger(__name__)
    logname = args.prefix + \
        '_' + args.opt_level + '_' + str(args.seed) + '.log'
    logfile = os.path.join(result_folder, logname)
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile
    )
    logger.info(args)

    #Attack
    def aet_attack(target_to=None, rnd_target=True):
        ignore, success, fail = 0, 0, 0
        correct, total = 0, 0
        model.eval()
        warper.eval()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            grids = warper.init_prim_grid().to(device)
            # test the pretrain model to ensure it predict correctly
            with torch.no_grad():
                output_logit = model(normalize(data, args.batch_size))
                preds = F.softmax(output_logit, dim=1)
                preds_top_p, preds_top_class = preds.topk(1, dim=1)
                total += target.size(0)
                correct += (preds_top_class.view(target.shape) == target).sum().item()

            # create random targets
            if rnd_target == True:
                while True:
                    non_target = torch.randint(0, 1000, target.shape).to(device)
                    collide = (target==non_target).sum().item()
                    if collide == 0:
                        break

            for i in range(args.iteration):
                grids.requires_grad = True
                distort_inputs = warper(normalize(data, args.batch_size), grids)
                output_logit = model(distort_inputs)

                model.zero_grad()
                warper.zero_grad()
                if target_to == None: #non-target attack
                    if rnd_target == False: # classic non-target attack
                        loss = F.cross_entropy(output_logit, target)
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        # loss.backward()
                        sign_data_grad = grids.grad.sign()
                        grids = grids + args.step_size * sign_data_grad # non-target attack
                    else: # random pick another target
                        loss = F.cross_entropy(output_logit, non_target)
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        # loss.backward()
                        sign_data_grad = grids.grad.sign()
                        grids = grids - args.step_size * sign_data_grad
                else: # targeted attack
                    target_lable = torch.full(target.shape, target_to).to(device)
                    loss = F.cross_entropy(output_logit, target_lable)
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    # loss.backward()
                    sign_data_grad = grids.grad.sign()
                    grids = grids - args.step_size * sign_data_grad
                grids = grids.detach_()
            # if this batch has overflow gradients, discard it
            unskipped_counter = amp._amp_state.loss_scalers[0]._unskipped
            if unskipped_counter%(args.iteration+0) != 0 or unskipped_counter == 0:
                amp._amp_state.loss_scalers[0]._unskipped = 0
                continue
            # test the adversarial example
            with torch.no_grad():
                # bind it back to image
                distort_image = warper(normalize(data, args.batch_size), grids)
                distort_logit = model(distort_image)
                preds_distort = F.softmax(distort_logit, dim=1)
                distort_top_p, distort_top_class = preds_distort.topk(1, dim=1)
                success += (distort_top_class.view(target.shape) == target).sum().item()

            if batch_idx > 20: 
                break
        return (100. * correct / total, 100. * success / total)
    
    # Run
    logger.info('Clean Model Accuracy \t Model Accuracy on Adversary')
    start_train_time = time.time()
    correct, success = aet_attack(args.target, args.rnd_target)
    logger.info('%20.4f \t %27.4f',correct, success)
    train_time = time.time()
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)


main()