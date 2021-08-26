# utility package
import argparse
import logging
import time
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='Evaluation on model')
parser.add_argument('--cuda',               default='0',            type=str,   help='select gpu on the server. (default: 0)')
parser.add_argument('--description', '--de',default='default',      type=str,   help='description used to define different model')
parser.add_argument('--prefix',             default='Epoch_0001',   type=str,   help='prefix used to define logs')
parser.add_argument('--seed',               default=6869,           type=int,   help='random seed')

parser.add_argument('--batch-size', '-b',   default=80,            type=int,   help='mini-batch size (default: 160)')
parser.add_argument('--epochs',             default=50,             type=int,   help='number of total epochs to run')
# parser.add_argument('--lr-min', default=0.005, type=float, help='minimum learning rate for optimizer')
parser.add_argument('--lr-max',             default=0.001,          type=float, help='maximum learning rate for optimizer')
# parser.add_argument('--momentum', '--mm', default=0.9, type=float, help='momentum for optimizer')
# parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float, help='weight decay for model training')

parser.add_argument('--image-size', '--is', default=224,            type=int,   help='image size (default: 224 for ImageNet)')
parser.add_argument('--dataset-root', '--ds', default='/tmp2/dataset/Restricted_ImageNet_A', \
    type=str, help='input dataset, default: Restricted Imagenet A')
parser.add_argument('--ckpt-root', '--ckpt', default='/tmp2/aislab/adv_ckpt', \
    type=str, help='root directory of checkpoints')
parser.add_argument('--opt-level', '-o',    default='O0',           type=str,   help='Nvidia apex optimation level (default: O1)')
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
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # load dataset (Imagenet)
    train_loader, test_loader = get_loaders(args.dataset_root, args.batch_size, \
                                            image_size=args.image_size, augment=False)

    # Load model and optimizer
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
        opt_level=args.opt_level, verbosity=1)
    ckpt_path = Path(args.ckpt_root) / args.description / (args.prefix + '.pth')
    checkpoint = torch.load(ckpt_path)
    prev_acc = checkpoint['acc']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    amp.load_state_dict(checkpoint['amp_state_dict'])
    epoch_start = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])

    criterion = nn.CrossEntropyLoss().to(device)

    model.eval()
    
    start_time = time.time()
    correct_normal, correct_adv, correct_adv2, total = 0, 0, 0, 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        
        adv = AET(model, warper, data, target, step=0.005, iter=20)
        # adv = AET_apex(model, warper, data, target, optimizer=optimizer2, step=0.005, iter=20)
        adv2, delta = PGD(model, data, target, eps=8/255, alpha=0.5/255, iter=20)
        # adv, delta = PGD_apex(model, data, target, optimizer=optimizer2, eps=8/255, alpha=0.5/255, iter=20)
        
        with torch.no_grad():
            y_normal = model(normalize(data))
            preds_normal = F.softmax(y_normal, dim=1)
            preds_top_p, preds_top_class = preds_normal.topk(1, dim=1)
            correct_normal += (preds_top_class.view(target.shape) == target).sum().item()

            y_adv = model(normalize(adv))
            preds_adv = F.softmax(y_adv, dim=1)
            preds_top_p, preds_top_class = preds_adv.topk(1, dim=1)
            correct_adv += (preds_top_class.view(target.shape) == target).sum().item()

            y_adv2 = model(normalize(adv2))
            preds_adv2 = F.softmax(y_adv2, dim=1)
            preds_top_p2, preds_top_class2 = preds_adv2.topk(1, dim=1)
            correct_adv2 += (preds_top_class2.view(target.shape) == target).sum().item()
            
            total += target.size(0)

        # if batch_idx == 5:
            # save_image(data[0], './picture/normal.png')
            # save_image(adv[0], './picture/adv.png')
            # save_image(delta[0]*8, './picture/delta.png')
        if batch_idx == 10:
            break
        
    print(correct_normal/total)
    print(correct_adv/total)
    print(correct_adv2/total)

    eval_time = time.time()
    print('Total eval time: {:.4f} minutes'.format((eval_time-start_time)/60))

if __name__ == "__main__":
    main()