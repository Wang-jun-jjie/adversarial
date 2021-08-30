# utility package
import argparse
import logging
import time
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='Adversarial training')
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

parser.add_argument('--target', '-t',       default=None,           type=int,    help='adversarial attack target label')
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
                                            image_size=args.image_size, augment=True)

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
    # cyclic learning rate
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min, max_lr=args.lr_max)

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

    # Training
    def train(epoch):
        print('\nEpoch: {:04}'.format(epoch))
        train_loss, correct, total = 0, 0, 0
        model.train()
        warper.eval()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            '''
            Accumulated Elastic Transform (apex)
            '''
            grid = warper.init_prim_grid().detach().to(device)
            data_rolled = torch.roll(data, shifts=-1, dims=0)
            for i in range(args.iteration):
                grid.requires_grad = True
                adv = warper(data_rolled, grid) if i==0 else warper(data, grid)
                output = model(normalize(adv))
                loss = criterion(output, target)
                with amp.scale_loss(loss, optimizer2) as scaled_loss:
                    scaled_loss.backward()
                sign_data_grad = grid.grad.sign()
                # Update adversarial images
                grid = grid + args.epsilon*sign_data_grad if i==0 else grid + args.step_size*sign_data_grad
                grid = grid.detach_()
            # warp it back to image
            adv = warper(data, grid).detach()
            '''
            Adversarial training
            '''
            distort_logit = model(normalize(adv))
            loss = criterion(distort_logit, target)
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()
            optimizer.step()
            preds = F.softmax(distort_logit, dim=1)
            preds_top_p, preds_top_class = preds.topk(1, dim=1)
            train_loss += loss.item() * target.size(0)
            total += target.size(0)
            correct += (preds_top_class.view(target.shape) == target).sum().item()
            # scheduler
            # unskipped_counter = amp._amp_state.loss_scalers[0]._unskipped
            # if unskipped_counter%(args.iteration+1) != 0 or unskipped_counter == 0:
                # amp._amp_state.loss_scalers[0]._unskipped = 0
            # else:
            #     scheduler.step()
            
            # if batch_idx > 10:
            #     print('==> early break in training')
            #     break
            
        return (train_loss / batch_idx, 100. * correct / total)

    # Test
    def test(epoch):
        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
    
                output_logit = model(normalize(data))
                loss = criterion(output_logit, target)
                preds = F.softmax(output_logit, dim=1)
                preds_top_p, preds_top_class = preds.topk(1, dim=1)
    
                test_loss += loss.item() * target.size(0)
                total += target.size(0)
                correct += (preds_top_class.view(target.shape) == target).sum().item()

                # if batch_idx > 10:
                #     print('==> early break in testing')
                #     break
        
        return (test_loss / batch_idx, 100. * correct / total)
            
    # Save checkpoint
    def checkpoint(acc, epoch):
        print('==> Saving checkpoint..')
        ckpt_dir = Path(args.ckpt_root) / args.description
        ckpt_path = Path(args.ckpt_root) / args.description / ('Epoch_' + '{:04}'.format(epoch) + '.pth')
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            'epoch': epoch,
            'acc': acc,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'amp_state_dict': amp.state_dict(),
            'rng_state': torch.get_rng_state(),
            }, ckpt_path)
    
    # Run
    logger.info('Epoch  Seconds    Train Loss  Train Acc    Test Loss  Test Acc')
    start_train_time = time.time()
    for epoch in range(epoch_start, args.epochs):
        start_epoch_time = time.time()
        
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        epoch_time = time.time()
        logger.info('%5d  %7.1f    %10.4f  %9.4f    %9.4f  %8.4f',
            epoch, epoch_time - start_epoch_time, train_loss, train_acc, test_loss, test_acc)
        # Save checkpoint.
        if train_acc - prev_acc  > 0.1:
            prev_acc = train_acc
            checkpoint(train_acc, epoch)
    train_time = time.time()
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)


if __name__ == "__main__":
    main()