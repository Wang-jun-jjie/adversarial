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
parser.add_argument('--dataset-root', '--ds', default='/tmp2/dataset/Restricted_ImageNet_C', \
    type=str, help='input dataset, default: Restricted Imagenet C')
parser.add_argument('--ckpt-root', '--ckpt', default='/tmp2/aislab/adv_ckpt', \
    type=str, help='root directory of checkpoints')
parser.add_argument('--opt-level', '-o',    default='O1',           type=str,   help='Nvidia apex optimation level (default: O1)')
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

    # Load resnet50 model structure and optimizer
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

    # There're 15 testing dataset, each with 5 serverities
    distortion = {
        'noise' : ['gaussian_noise', 'shot_noise', 'impulse_noise'],
        'blur'  : ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
        'weather' : ['snow', 'frost', 'fog', 'brightness'],
        'digital' : ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']        
    }
    path_data = Path(args.dataset_root)
    average_error = []
    for ss in distortion: # 4 types of distortion
        path_dis_cls = path_data / str(ss)
        for s in distortion[ss]: # distortions
            path_dis = path_dis_cls / s
            error_rate_per_types = []
            for severity in range(1, 6): # severities
                path_in = path_dis / str(severity)
                # load the testing dataset
                dataloader = get_loaders(path_in, args.batch_size, \
                                            image_size=args.image_size, testonly=True)
                correct, total = 0, 0
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(dataloader):
                        data, target = data.to(device), target.to(device)

                        output_logit = model(normalize(data))
                        loss = criterion(output_logit, target)
                        preds = F.softmax(output_logit, dim=1)
                        preds_top_p, preds_top_class = preds.topk(1, dim=1)
    
                        total += target.size(0)
                        correct += (preds_top_class.view(target.shape) == target).sum().item()
                # Error rate
                error = (total-correct) / total
                error_rate_per_types.append(error)
                print('Type: {}, Name: {}, Severity: {} has error rate {:.5f}'.format(ss, s, severity, error))
            error = sum(error_rate_per_types)/len(error_rate_per_types)
            average_error.append(error)
            print('Type: {}, Name: {} has average error rate {:.5f}'.format(ss, s, error))
    error = sum(average_error)/len(average_error)
    print('Average Error Rate: {:.5f}'.format(error))

if __name__ == "__main__":
    main()