# utility package
import argparse
import logging
import time
import os
from pathlib import Path

# adversarial attack and showcase
parser = argparse.ArgumentParser( description='Adversarial attacks')
parser.add_argument('--cuda',               default='0',            type=str,   help='select gpu on the server. (default: 0)')
parser.add_argument('--description', '--de',default='default',      type=str,   help='description used to define different model')
parser.add_argument('--prefix',             default='',             type=str,   help='prefix to specify checkpoints')
parser.add_argument('--seed',               default=6869,           type=int,   help='random seed')

parser.add_argument('--batch-size', '-b',   default=160,            type=int,    help='mini-batch size (default: 160)')
parser.add_argument('--epochs',             default=80,             type=int,    help='number of total epochs to run')
# parser.add_argument('--lr-min', default=0.005, type=float, help='minimum learning rate for optimizer')
parser.add_argument('--lr-max',             default=0.001,          type=float,  help='learning rate for optimizer')
# parser.add_argument('--momentum', '--mm', default=0.9, type=float, help='momentum for optimizer')
# parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float, help='weight decay for model training')

parser.add_argument('--iteration', '-i',    default=20,             type=int,    help='adversarial attack iterations (default: 20)')
parser.add_argument('--step-size', '--ss',  default=0.005,          type=float,  help='step size for adversarial attacks')
parser.add_argument('--epsilon',            default=1,              type=float,  help='epsilon for adversarial attacks')
parser.add_argument('--kernel-size', '-k',  default=13,             type=int,    help='kernel size for adversarial attacks, must be odd integer')

parser.add_argument('--image-size', '--is', default=224,            type=int,    help='image size (default: 224 for ImageNet)')
parser.add_argument('--dataset-root', '--ds', default='/tmp2/dataset/Restricted_ImageNet_Hendrycks', \
    type=str, help='input dataset, default: Restricted Imagenet Hendrycks A')
parser.add_argument('--ckpt-root', '--ckpt', default='/tmp2/aislab/adv_ckpt', \
    type=str, help='root directory of checkpoints')
parser.add_argument('--opt-level', '-o',    default='O1',           type=str,    help='Nvidia apex optimation level (default: O1)')
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

from utils.utils import get_loaders, deforming_medium, normalize

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
    if args.prefix:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
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
    else:
        print('==> Building model..')
        epoch_start = 0
        prev_acc = 0.0
        model, [optimizer, optimizer2] = amp.initialize(model, [optimizer, optimizer2], \
            opt_level=args.opt_level, verbosity=1)
    
    warper = deforming_medium(args)
    criterion = nn.CrossEntropyLoss().to(device)

    # Logger
    log_dir = Path('./logs/')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (args.description + '.log')
    if log_path.exists():
        log_path.unlink()
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=log_path
    )
    logger.info(args)

    def show_trans(grid_hist, grad_hist):
        if len(grid_hist) < 1:
            print('Iteration smaller than 1, cannot show the grid transistion')
            return
        # print('==> Show the grid transistion')
        # value_range = (2.0/args.image_size)*args.epsilon
        # fig, axs = plt.subplots(4, len(grid_hist), figsize=(6*len(grid_hist)+1, 24))
        # for i, grid in enumerate(grid_hist):
        #     axs[0, i].imshow(grid[0,0,:,:], cmap='rainbow', vmin=-value_range, vmax=value_range)
        #     axs[1, i].imshow(grid[0,1,:,:], cmap='rainbow', vmin=-value_range, vmax=value_range)
        # plt.savefig('./picture/imagetest.png')
        # plt.close()

        print('==> Show the gradient magnitude')
        plt.plot(range(len(grad_hist)), grad_hist, label='gradient magnitude')
        plt.savefig('./picture/gdtest.png')
        plt.close()

    def show_adv():
        pass

    def attack(record=False):
        correct_normal, correct_adv, total = 0, 0, 0
        grid_hist = []
        grad_hist = []
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            '''
            Random Start Accumulated Elastic Transform (apex)
            '''
            grid = warper.init_prim_grid().detach().to(device)
            data_rolled = torch.roll(data, shifts=-1, dims=0)
            for i in range(args.iteration):
                grid.requires_grad = True
                adv = warper(data, grid)
                # adv = warper(data_rolled, grid) if i==0 else warper(data, grid)
                output = model(normalize(adv))
                loss = criterion(output, target)
                with amp.scale_loss(loss, optimizer2) as scaled_loss:
                    scaled_loss.backward()
                sign_data_grad = grid.grad.sign()
                grad = torch.sum(torch.abs(grid.grad)[0].type(torch.float64))
                # Update adversarial images
                if i==0:
                    # grid =  grid + torch.empty_like(grid).uniform_(0, args.epsilon/20)*sign_data_grad
                    grid = grid + args.step_size*15*sign_data_grad
                    # grid = grid + args.step_size*sign_data_grad
                else:
                    grid = grid + args.step_size*sign_data_grad
                # grid = grid + args.step_size*sign_data_grad
                grid = grid.detach_()
                if record:
                    grid_hist.append(grid.clone().detach().cpu().numpy())
                    grad_hist.append(grad.clone().detach().cpu().numpy())
                    
            # warp it back to image
            adv = warper(data, grid).detach()
            # save the grid histogram
            record = False
            show_trans(grid_hist, grad_hist)
            '''
            Evaluate attack performance
            '''
            with torch.no_grad():
                y_normal = model(normalize(data))
                preds_normal = F.softmax(y_normal, dim=1)
                preds_top_p, preds_top_class = preds_normal.topk(1, dim=1)
                correct_normal += (preds_top_class.view(target.shape) == target).sum().item()

                y_adv = model(normalize(adv))
                preds_adv = F.softmax(y_adv, dim=1)
                preds_top_p, preds_top_class = preds_adv.topk(1, dim=1)
                correct_adv += (preds_top_class.view(target.shape) == target).sum().item()

                total += target.size(0)
            
            if batch_idx == 10:
                print('==> early break in testing')
                break

        return  (correct_normal / total), (correct_adv / total)
    
    # Run
    accuracy_normal, accuracy_adv = attack(record=True)
    print('Model original accuracy: {:.4f}'.format(accuracy_normal))
    print('Model adversarial accuracy: {:.4f}'.format(accuracy_adv))

if __name__ == "__main__":
    main()