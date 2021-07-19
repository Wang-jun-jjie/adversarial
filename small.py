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

print('pytorch version: ' + torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NVIDIA apex
from apex import amp
# math and showcase
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import get_loaders, deforming_medium

parser = argparse.ArgumentParser( description='Adversarial training')
parser.add_argument('--resume', '-r',       action='store_true',              help='resume from checkpoint')
parser.add_argument('--prefix',             default='Small adv. training',    type=str,   help='prefix used to define logs')
parser.add_argument('--seed',               default=59572406,     type=int,   help='random seed')

parser.add_argument('--batch-size', '-b',   default=232,          type=int,   help='mini-batch size (default: 120)')
parser.add_argument('--epochs',             default=20,           type=int,   help='number of total epochs to run')

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
parser.add_argument('--data-directory',     default='../Restricted_ImageNet',type=str,   help='dataset inputs root directory')
# parser.add_argument('--data-classname',     default='../ImageNet/LOC_synset_mapping.txt',type=str, help='dataset classname file')
parser.add_argument('--opt-level', '-o',    default='O1',         type=str,   help='Nvidia apex optimation level (default: O1)')
args = parser.parse_args()

def main():
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # load dataset (Imagenet)
    train_loader, test_loader = get_loaders(args.data_directory, args.batch_size, )

    # Load model and optimizer
    model = models.resnet18(pretrained=False, num_classes=10).to(device)
    # Add weight decay into the model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_max,
                                # momentum=args.momentum,
                                # weight_decay=args.weight_decay
                                )
    optimizer2 = torch.optim.Adam(model.parameters(), lr=args.lr_max,
                                # momentum=args.momentum,
                                # weight_decay=args.weight_decay
                                )
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        model, [optimizer, optimizer2] = amp.initialize(model, [optimizer, optimizer2], opt_level=args.opt_level)

        checkpoint = torch.load('./checkpoint/' + args.prefix + '_' + str(args.seed) + '.pth')
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
        model, [optimizer, optimizer2] = amp.initialize(model, [optimizer, optimizer2], opt_level=args.opt_level)
    warper = deforming_medium(args)
    criterion = nn.CrossEntropyLoss().to(device)
    # cyclic learning rate
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min, max_lr=args.lr_max)

    # Logger
    result_folder = './logs/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    logger = logging.getLogger(__name__)
    logname = args.prefix + '_' + args.opt_level + '_' + str(args.seed) + '.log'
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

    # Training
    def train(epoch):
        print('\nEpoch: {:04}'.format(epoch))
        train_loss, correct, total = 0, 0, 0
        model.train()
        warper.eval()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            grids = warper.init_prim_grid().to(device)
            
            # create adversarial examples
            if args.rnd_target == True:
                while True:
                    non_target = torch.randint(0, 10, target.shape).to(device)
                    collide = (target==non_target).sum().item()
                    if collide == 0:
                        break
            for i in range(args.iteration):
                grids.requires_grad = True
                distort_inputs = warper(data, grids)
                output_logit = model(distort_inputs)

                model.zero_grad()
                warper.zero_grad()
                if args.target == None: #non-target attack
                    if args.rnd_target == False: # classic non-target attack
                        loss = criterion(output_logit, target)
                        with amp.scale_loss(loss, optimizer2) as scaled_loss:
                            scaled_loss.backward()
                        # loss.backward()
                        sign_data_grad = grids.grad.sign()
                        grids = grids + args.step_size * sign_data_grad # non-target attack
                    else: # random pick another target
                        loss = criterion(output_logit, target)
                        with amp.scale_loss(loss, optimizer2) as scaled_loss:
                            scaled_loss.backward()
                        # loss.backward()
                        sign_data_grad = grids.grad.sign()
                        grids = grids - args.step_size * sign_data_grad
                else: # targeted attack
                    target_lable = torch.full(target.shape, args.target).to(device)
                    loss = criterion(output_logit, target)
                    with amp.scale_loss(loss, optimizer2) as scaled_loss:
                        scaled_loss.backward()
                    # loss.backward()
                    sign_data_grad = grids.grad.sign()
                    grids = grids - args.step_size * sign_data_grad
                grids = grids.detach_()

            # use adversarial examples to train the model
            optimizer.zero_grad()
            distort_image = warper(data, grids)
            distort_logit = model(distort_image)
            loss = criterion(distort_logit, target)
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
            unskipped_counter = amp._amp_state.loss_scalers[0]._unskipped
            # if unskipped_counter%(args.iteration+1) != 0 or unskipped_counter == 0:
                # amp._amp_state.loss_scalers[0]._unskipped = 0
            # else:
            #     scheduler.step()
            
            # if batch_idx > 200:
            #     break
        return (train_loss / batch_idx, 100. * correct / total)

    # Test
    def test(epoch):
        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
    
                optimizer.zero_grad()
                output_logit = model(data)
                loss = F.cross_entropy(output_logit, target)
                preds = F.softmax(output_logit, dim=1)
                preds_top_p, preds_top_class = preds.topk(1, dim=1)
    
                test_loss += loss.item() * target.size(0)
                total += target.size(0)
                correct += (preds_top_class.view(target.shape) == target).sum().item()
                if batch_idx > 200:
                    break
        
        return (test_loss / batch_idx, 100. * correct / total)
            
    # Save checkpoint
    def checkpoint(acc, epoch):
        print('==> Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_path = './checkpoint/' + args.prefix + '_' + str(args.seed) + '.pth'
        torch.save({
            'epoch': epoch,
            'acc': acc,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'amp_state_dict': amp.state_dict(),
            'rng_state': torch.get_rng_state(),
            }, save_path)
    
    # Run
    logger.info('Epoch \t Seconds \t \t Train Loss \t Train Acc')
    start_train_time = time.time()
    for epoch in range(epoch_start, args.epochs):
        start_epoch_time = time.time()
        
        train_loss, train_acc = train(epoch)
        epoch_time = time.time()
        logger.info('%5d \t %7.1f \t \t %10.4f \t %9.4f',
            epoch, epoch_time - start_epoch_time, train_loss, train_acc)
        logger.info('Test Loss \t Test Acc')
        test_loss, test_acc = test(epoch)
        logger.info('%9.4f \t %8.4f', test_loss, test_acc)
        # Save checkpoint.
        if train_acc - prev_acc  > 0.1:
            prev_acc = train_acc
            checkpoint(train_acc, epoch)
    train_time = time.time()
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    # logger.info('Test Loss \t Test Acc')
    # test_loss, test_acc = test(epoch)
    # logger.info('%9.4f \t %8.4f', test_loss, test_acc)



if __name__ == "__main__":
    main()