# utility package
import argparse
import logging
import time
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='Adversarial training')
parser.add_argument('--cuda',               default='0',            type=str,   help='select gpu on the server. (default: 0)')
parser.add_argument('--batch-size', '-b',   default=4,            type=int,    help='mini-batch size (default: 160)')
parser.add_argument('--image-size', '--is', default=224,            type=int,   help='image size (default: 224 for ImageNet)')
parser.add_argument('--dataset-root', '--ds', default='/tmp2/dataset/Restricted_ImageNet_Hendrycks', \
    type=str, help='input dataset, default: Restricted Imagenet Hendrycks A')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda
# pytorch related package 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
# NVIDIA apex
from apex import amp
# math and showcase
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import get_loaders, normalize

def main():
    print('pytorch version: ' + torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load dataset
    train_loader, test_loader = get_loaders(args.dataset_root, args.batch_size, \
                                            image_size=args.image_size, augment=True)
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data_roll = torch.roll(data, shifts=-1, dims=0)
        # get a minibatch save the original batch and the rolled one
        save_image(data, './picture/data.png')
        save_image(data_roll, './picture/data_roll.png')
        break

if __name__ == "__main__":
    main()