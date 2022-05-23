import argparse
import time
import csv
import datetime
from path import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn as nn

import models

import custom_transforms
from utils import tensor2array, save_checkpoint
from datasets.sequence_folders import SequenceFolder
from datasets.pair_folders import PairFolder
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors
from logger import TermLogger, AverageMeter
# from tensorboardX import SummaryWriter
import wandb
import os
import warnings


parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--folder-type', type=str, choices=['sequence', 'pair'], default='sequence', help='the dataset dype to train')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs at validation step')
parser.add_argument('--resnet-layers',  type=int, default=18, choices=[18, 50], help='number of ResNet layers for depth estimation.')
parser.add_argument('--num-scales', '--number-of-scales', type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss', metavar='W', default=0.5)
parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
parser.add_argument('--with-mask', type=int, default=1, help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int,  default=0, help='with the the mask for stationary points')
parser.add_argument('--with-pretrain', type=int,  default=1, help='with or without imagenet pretrain for resnet')
parser.add_argument('--dataset', type=str, choices=['kitti', 'nyu'], default='kitti', help='the dataset to train')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')

# !DL project changes!
parser.add_argument('--with-coord-conv', type=int, default=0, help='change first Conv2d layer to CoordConv')
parser.add_argument('--conv1-weight-mode', type=str, choices=['zeros', 'random', 'all_random'], default=None, help='how to initailize weights for conv1')
parser.add_argument('--fine-tune-mode', type=str, choices=['whole', 'first_then_all'], default=None, help='how to perform fine-tunning with CoordConv')
parser.add_argument('--unfreeze-epoch', type=int, default=None, help='if the fine_tune_mode=first_then_all, start with unfreeze_epoch all the layers are unfrozen')

parser.add_argument('--use-scheduler', type=int, default=0, help='intrudce shceduler for fine-tuning')
parser.add_argument('--warmup-lr', type=float, default=None, help='start value of lr for warm up')
parser.add_argument('--warmup-epoch', type=int, default=None, help='at epoch warm_up_epoch lr equals specified lr of optimizer')
parser.add_argument('--step-size', type=int, default=None, help='each step_size lr decays by gamma')
parser.add_argument('--gamma-lr', type=float, default=None, help='multiplicative factor of learning rate decay')
parser.add_argument('--min-lr', type=float, default=None, help='the smallest value of lr with scheduler')

parser.add_argument('--run-id', type=str, default=None, help='wanb run description')


best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)


def main():
    global best_error, n_iter, device
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = Path(f'../checkpoints_/{args.name}/{timestamp}')
    args.save_path = save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()

#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     cudnn.deterministic = True
#     cudnn.benchmark = True

    wandb.init(project='vo_exploring', name=args.run_id)

    # define a metric we are interested in the minimum of
    wandb.define_metric('train/photometric_error', summary='min')
    wandb.define_metric('train/disparity_smoothness_loss', summary='min')
    wandb.define_metric('train/geometry_consistency_loss', summary='min')
    wandb.define_metric('train/total_loss', summary='min')

    wandb.define_metric('val/photometric_error', summary='min')
    wandb.define_metric('val/disparity_smoothness_loss', summary='min')
    wandb.define_metric('val/geometry_consistency_loss', summary='min')
    wandb.define_metric('val/total_loss', summary='min')
    
    wandb.define_metric('learning_rate', summary='last')

    training_writer = {}

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])

    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    if args.folder_type == 'sequence':
        train_set = SequenceFolder(
            args.data,
            transform=train_transform,
            seed=args.seed,
            train=True,
            sequence_length=args.sequence_length,
            dataset=args.dataset
        )
    else:
        train_set = PairFolder(
            args.data,
            seed=args.seed,
            train=True,
            transform=train_transform
        )


    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    if args.with_gt:
        from datasets.validation_folders import ValidationSet
        val_set = ValidationSet(
            args.data,
            transform=valid_transform,
            dataset=args.dataset
        )
    else:
        val_set = SequenceFolder(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            sequence_length=args.sequence_length,
            dataset=args.dataset
        )
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    # !DL project changes!
    if not args.with_coord_conv:
        disp_net = models.DispResNet(args.resnet_layers, args.with_pretrain).to(device)
        pose_net = models.PoseResNet(18, args.with_pretrain).to(device)

        # load parameters
        if args.pretrained_disp:
            print("=> using pre-trained weights for DispResNet")
            weights = torch.load(args.pretrained_disp)
            disp_net.load_state_dict(weights['state_dict'], strict=False)

        if args.pretrained_pose:
            print("=> using pre-trained weights for PoseResNet")
            weights = torch.load(args.pretrained_pose)
            pose_net.load_state_dict(weights['state_dict'], strict=False)

        disp_net = torch.nn.DataParallel(disp_net)
        pose_net = torch.nn.DataParallel(pose_net)

        print('=> setting adam solver')
        optim_params = [
            {'params': disp_net.parameters(), 'lr': args.lr},
            {'params': pose_net.parameters(), 'lr': args.lr}
        ]
        optimizer = torch.optim.Adam(optim_params,
                                    betas=(args.momentum, args.beta),
                                    weight_decay=args.weight_decay)

    ####
    elif args.with_coord_conv:
        print("=> using model with CoordConv")
        disp_net = models.DispResNet(args.resnet_layers, False).to(device)
        pose_net = models.PoseResNet_with_CC(18).to(device)

        # load parameters
        if args.pretrained_disp:
            print("=> using pre-trained weights for DispResNet")
            weights = torch.load(args.pretrained_disp)
            disp_net.load_state_dict(weights['state_dict'], strict=False)
            
            # freeze DispResNet
            for param in disp_net.parameters():
                param.requires_grad = False

        if args.pretrained_pose:
            print("=> using pre-trained weights for PoseResNet")
            weights = torch.load(args.pretrained_pose)

            # 1. Since we added 2 channels, we need iniailize additional 
            # weights which correspond to them. Final weight of
            # encoder.conv1 has to be longer by two in dim=1.
            w_conv1 = weights['state_dict']['encoder.encoder.conv1.weight']
            w_b, w_c, w_h, w_w = w_conv1.shape

            if args.conv1_weight_mode == 'random':
                extra_channels = nn.init.kaiming_normal_(torch.empty(w_b, 2, w_h, w_w), mode='fan_out', nonlinearity='relu').to(device)
                w_conv1 = torch.cat((w_conv1, extra_channels), 1)

            elif args.conv1_weight_mode == 'zeros':
                extra_channels = nn.init.zeros_(torch.empty(w_b, 2, w_h, w_w)).to(device)
                w_conv1 = torch.cat((w_conv1, extra_channels), 1)

            elif args.conv1_weight_mode == 'all_random':
                w_conv1 = nn.init.kaiming_normal_(torch.empty(w_b, w_c+2, w_h, w_w), mode='fan_out', nonlinearity='relu').to(device)

            # 2. Replace first weight
            weights['state_dict']['encoder.encoder.conv1.weight'] = w_conv1

            # 3. Load the new state dict
            pose_net.load_state_dict(weights, strict=False)

            # 4. Freeze parts of PoseResNet (also done during training)
            if args.fine_tune_mode == 'whole':
                pass
            elif args.fine_tune_mode == 'first_then_all':
                for param in pose_net.parameters():
                    param.requires_grad = False
                pose_net.encoder.encoder.conv1.weight.requires_grad = True

        disp_net = torch.nn.DataParallel(disp_net)
        pose_net = torch.nn.DataParallel(pose_net)

        print('=> setting adam solver for PoseResNet only')
        optim_params = [
            {'params': pose_net.parameters(), 'lr': args.lr}
        ]
        optimizer = torch.optim.Adam(optim_params,
                                    betas=(args.momentum, args.beta),
                                    weight_decay=args.weight_decay)
        if args.use_scheduler:
            print('=> using lr scheduler StepLRWithWarmup')
            scheduler = StepLRWithWarmup(
                optimizer, step_size=args.step_size, gamma=args.gamma_lr, 
                warmup_epochs=args.warmup_epoch, warmup_lr_init=args.warmup_lr,
                min_lr=args.min_lr
            )
        else:
            scheduler = None
#     with open(args.save_path/args.log_summary, 'w') as csvfile:
#         writer = csv.writer(csvfile, delimiter='\t')
#         writer.writerow(['train_loss', 'validation_loss'])

#     with open(args.save_path/args.log_full, 'w') as csvfile:
#         writer = csv.writer(csvfile, delimiter='\t')
#         writer.writerow(['train_loss', 'photo_loss', 'smooth_loss', 'geometry_consistency_loss'])

    logger = TermLogger(args.epochs, min(len(train_loader), args.epoch_size), len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # Unfreeze layers if it is the mode
        if (args.fine_tune_mode == 'first_then_all') & (epoch == args.unfreeze_epoch):
            for param in pose_net.parameters():
                param.requires_grad = True

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, disp_net, pose_net, optimizer, scheduler, args.epoch_size, logger, training_writer)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        if args.with_gt:
            errors, error_names = validate_with_gt(args, val_loader, disp_net, epoch, logger)#, output_writers)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger)#, output_writers)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer[name] = error
            
        wandb.log(train_writer)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': disp_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': pose_net.module.state_dict()
            },
            is_best)

#         with open(args.save_path/args.log_summary, 'a') as csvfile:
#             writer = csv.writer(csvfile, delimiter='\t')
#             writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


def train(args, train_loader, disp_net, pose_net, optimizer, scheduler, epoch_size, logger, train_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    # switch to train mode
    disp_net.train()
    pose_net.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)

        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3

        if log_losses:
            
            train_writer['train/photometric_error'] = loss_1.item()
            train_writer['train/disparity_smoothness_loss'] = loss_2.item()
            train_writer['train/geometry_consistency_loss'] = loss_3.item()
            train_writer['train/total_loss'] = loss.item()
            
            if scheduler:
                train_writer['learning_rate'] = scheduler.optimizer.param_groups[0]['lr']
            
            wandb.log(train_writer)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        
        print(scheduler.last_epoch)
        
        # !DL project changes!
        if scheduler:
            scheduler.step()            
        else:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

#         with open(args.save_path/args.log_full, 'a') as csvfile:
#             writer = csv.writer(csvfile, delimiter='\t')
#             writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
        logger.train_bar.update(i+1)
#         if i % args.print_freq == 0:
#             logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


@torch.no_grad()
def validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger):# output_writers=[]):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
#     log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        # compute output
        tgt_depth = [1 / disp_net(tgt_img)]
        ref_depths = []
        for ref_img in ref_imgs:
            ref_depth = [1 / disp_net(ref_img)]
            ref_depths.append(ref_depth)

#         if log_outputs and i < len(output_writers):
#             if epoch == 0:
#                 output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)

#             output_writers[i].add_image('val Dispnet Output Normalized',
#                                         tensor2array(1/tgt_depth[0][0], max_value=None, colormap='magma'),
#                                         epoch)
#             output_writers[i].add_image('val Depth Output',
#                                         tensor2array(tgt_depth[0][0], max_value=10),
#                                         epoch)

        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()

        loss = loss_1
        losses.update([loss, loss_1, loss_2, loss_3])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    logger.valid_bar.update(len(val_loader))
    
    return losses.avg, ['val/total_loss', 'val/photometric_error', 'val/disparity_smoothness_loss', 'val/geometry_consistency_loss']


@torch.no_grad()
def validate_with_gt(args, val_loader, disp_net, epoch, logger):#, output_writers=[]):
    global device
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
#     log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, depth) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        depth = depth.to(device)

        # check gt
        if depth.nelement() == 0:
            continue

        # compute output
        output_disp = disp_net(tgt_img)
        output_depth = 1/output_disp[:, 0]

#         if log_outputs and i < len(output_writers):
#             if epoch == 0:
#                 output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)
#                 depth_to_show = depth[0]
#                 output_writers[i].add_image('val target Depth',
#                                             tensor2array(depth_to_show, max_value=10),
#                                             epoch)
#                 depth_to_show[depth_to_show == 0] = 1000
#                 disp_to_show = (1/depth_to_show).clamp(0, 10)
#                 output_writers[i].add_image('val target Disparity Normalized',
#                                             tensor2array(disp_to_show, max_value=None, colormap='magma'),
#                                             epoch)

#             output_writers[i].add_image('val Dispnet Output Normalized',
#                                         tensor2array(output_disp[0], max_value=None, colormap='magma'),
#                                         epoch)
#             output_writers[i].add_image('val Depth Output',
#                                         tensor2array(output_depth[0], max_value=10),
#                                         epoch)

        if depth.nelement() != output_depth.nelement():
            b, h, w = depth.size()
            output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [h, w]).squeeze(1)

        errors.update(compute_errors(depth, output_depth, args.dataset))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
    logger.valid_bar.update(len(val_loader))
    return errors.avg, error_names

class StepLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, warmup_epochs=50, warmup_lr_init=1e-5,
                 min_lr=1e-5,
                 last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init
        self.min_lr = min_lr
        self.last_epoch = 0

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        warmup_incr = (self.base_lrs[0] - self.warmup_lr_init) / self.warmup_epochs
        if (self.last_epoch == 0):
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        
        elif (self.last_epoch > 0) & (self.last_epoch < self.warmup_epochs):
            return [self.warmup_lr_init + self.last_epoch * warmup_incr for _ in self.optimizer.param_groups]
        
        elif self.last_epoch == self.warmup_epochs:
            return [self.base_lrs for _ in self.optimizer.param_groups][0]

        elif (self.last_epoch > self.warmup_epochs) & ((self.last_epoch - self.warmup_epochs) % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]

        elif [group['lr'] * self.gamma for group in self.optimizer.param_groups][0] < self.min_lr:
            return [self.min_lr for group in self.optimizer.param_groups]

        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = [1/disp for disp in disp_net(tgt_img)]

    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = [1/disp for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return poses, poses_inv


if __name__ == '__main__':
    main()