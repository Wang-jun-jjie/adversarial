import argparse
import logging
import time
# select GPU on the server
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
# pytorch related package 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.utils import save_image

print('pytorch version: ' + torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NVIDIA apex
from apex import amp
# math and showcase
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import *

parser = argparse.ArgumentParser( description='eval')
# parser.add_argument('--resume', '-r',       action='store_true',              help='resume from checkpoint')
parser.add_argument('--prefix',             default='Small adv. training',    type=str,   help='prefix used to define logs')
parser.add_argument('--seed',               default=59572406,     type=int,   help='random seed')

parser.add_argument('--batch-size', '-b',   default=160,          type=int,   help='mini-batch size (default: 120)')
parser.add_argument('--epochs',             default=50,           type=int,   help='number of total epochs to run')

# parser.add_argument('--lr-min', default=0.005, type=float, help='minimum learning rate for optimizer')
parser.add_argument('--lr-max', default=0.001, type=float, help='maximum learning rate for optimizer')
# parser.add_argument('--momentum', '--mm', default=0.9, type=float, help='momentum for optimizer')
# parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float, help='weight decay for model training')

parser.add_argument('--target', '-t',       default=None,         type=int,   help='adversarial attack target label')
parser.add_argument('--rnd-target', '--rt', action='store_true',              help='non-target attack using random label as target')
parser.add_argument('--iteration', '-i',    default=20,           type=int,   help='adversarial attack iterations (default: 20)')
parser.add_argument('--step-size', '--ss',  default=0.005,        type=float, help='step size for adversarial attacks')
parser.add_argument('--epsilon',            default=1,            type=float, help='epsilon for adversarial attacks')
parser.add_argument('--kernel-size', '-k',  default=13,           type=int,   help='kernel size for adversarial attacks, must be odd integer')

parser.add_argument('--image-size', '--is', default=224,          type=int,   help='image size (default: 224 for ImageNet)')
parser.add_argument('--data-directory',     default='/tmp2/dataset/Restricted_ImageNet',type=str,   help='dataset inputs root directory')
# parser.add_argument('--data-classname',     default='../ImageNet/LOC_synset_mapping.txt',type=str, help='dataset classname file')
parser.add_argument('--opt-level', '-o',    default='O1',         type=str,   help='Nvidia apex optimation level (default: O1)')
args = parser.parse_args()

class Normalize_tops(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize_tops, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

def main():
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # load dataset (Imagenet)
    train_loader, test_loader = get_loaders(args.data_directory, args.batch_size, \
                                            image_size=args.image_size, augment=True)

    # Load model and optimizer
    norm_layer = Normalize_tops(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # model = nn.Sequential(norm_layer, 
    #     models.resnet50(pretrained=False, num_classes=10)
    # ).to(device)
    model = models.resnet50(pretrained=False, num_classes=10).to(device)
    # Add weight decay into the model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_max,
                                # momentum=args.momentum,
                                # weight_decay=args.weight_decay
                                )
    optimizer2 = torch.optim.Adam(model.parameters(), lr=args.lr_max,
                                # momentum=args.momentum,
                                # weight_decay=args.weight_decay
                                )
    # load the pre-train model
    print('==> Loading pre-trained model..')
    model, [optimizer, optimizer2] = amp.initialize(model, [optimizer, optimizer2], \
        opt_level=args.opt_level, verbosity=0)
    checkpoint = torch.load('./checkpoint/' + args.prefix + '.pth')
    prev_acc = checkpoint['acc']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    amp.load_state_dict(checkpoint['amp_state_dict'])
    # epoch_start = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])

    model = nn.Sequential(norm_layer, model).to(device)
    model.eval()
    warper = deforming_medium(args)

    # make something like evaluate_pgd(test_loader, model_test, args...)

    # reverse-normalization so the epsilon and alpha is correct
    correct_normal, correct_adv, total = 0, 0, 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        
        # adv = AET(model, warper, data, target, step=0.005, iter=20)
        adv, delta = PGD(model, data, target, eps=8/255, alpha=1/255, iter=20)
        
        y_normal = model(data)
        preds_normal = F.softmax(y_normal, dim=1)
        preds_top_p, preds_top_class = preds_normal.topk(1, dim=1)
        correct_normal += (preds_top_class.view(target.shape) == target).sum().item()
        # print(preds_top_class)
        y_adv = model(adv)
        preds_adv = F.softmax(y_adv, dim=1)
        preds_top_p, preds_top_class = preds_adv.topk(1, dim=1)
        correct_adv += (preds_top_class.view(target.shape) == target).sum().item()
        # print(preds_top_class)

        total += target.size(0)

        if batch_idx == 5:
            save_image(data[0], 'normal.png')
            save_image(adv[0], 'adv.png')
            save_image(delta[0]*8, 'delta.png')
        

        # data = inv_normalize(data)
        # data = data.to(device)
        # y = model(data)
        # preds = F.softmax(y, dim=1)
        # preds_top_p, preds_top_class = preds.topk(1, dim=1)
        # print(preds_top_class)
        # print(target)

        # batch size cannot be 1
        # pert = FGSM(model, data, target, 'inf', 8)
        # only run 1 batch
    print(correct_normal/total)
    print(correct_adv/total)

if __name__ == "__main__":
    main()