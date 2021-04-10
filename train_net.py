import os
import sys
import time
import numpy as np
import torch
import torch.multiprocessing as mp

from utils.log_util import create_logger
import aff_cf_model
from Dataloader import make_data_loader
from common import TrainingStats, stats
import utils.common_utils as utils


def val_net(model, val_loader, device):
    score = {}
    model.train()  # need to show loss

    with torch.no_grad():
        for (images, targets) in val_loader:
            images = images.to(device)
            for num in targets:
                for keys in num:
                    if torch.is_tensor(num[keys]):
                        num[keys] = num[keys].to(device)
            detections, features, losses = model(images, targets)
            for item in losses:
                if item in score:
                    score[item] += losses[item]
                else:
                    score[item] = losses[item]

    for item in score:
        score[item] = score[item] / len(val_loader)
    return score


accumulation_steps = 8


def train_net(args):
    if args.workers >= 1:
        mp.set_start_method('spawn')
    logger = create_logger("mask_train", filepath=args.log_dir)
    logger.info(' '.join(sys.argv))
    logger.info('\n'.join("{} : {}".format(k, v) for k, v in args.__dict__.items()))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    train_loader, classes = make_data_loader('train', args.dataset, args.data_dir,
                                             batch_size=args.batch_size, workers=args.workers, shuffle=True, device=device)
    val_loader, _ = make_data_loader('val', args.dataset, args.data_dir,
                                     batch_size=args.batch_size, workers=args.workers, shuffle=False, device=device)


    model = aff_cf_model.ACFNetwork('resnet50', True, len(classes), args.input_mode, args.acf_head)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    print('layers require gradients: ')
    print(names)
    if args.adam:
        optimizer = torch.optim.Adam(params, lr=args.adam_lr)
    elif args.sgd:
        optimizer = torch.optim.SGD(params, lr=args.sgd_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)
    else:
        print('error: no optimizer specified')
        exit(0)

    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)
    if args.resume is not None and os.path.isfile(args.checkpoint_path + '/' + args.resume + '.pth'):
        state_dict = torch.load(args.checkpoint_path + '/' + args.resume + '.pth', map_location = device)
        model.load_state_dict(state_dict['model_state_dict'], strict=False)
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        print('read model_best and resume')
    else:
        print('cannot find pth file in ', args.checkpoint_path + '/' + args.resume + '.pth')

    vis = TrainingStats(len(classes))
    vis.show_args(args)

    batch_time = stats.AverageMeter()
    data_time = stats.AverageMeter()
    loss_meter = {}


    current_best_loss = 0

    end = time.time()
    for epoch in range(args.resume_epoch, args.epochs):
        if args.sgd:
            print(lr_scheduler.get_lr())
            logger.info('Epoch: [{}]\tlr {}'.format(epoch, lr_scheduler.get_lr()[-1]))

        for i, (images, targets) in enumerate(train_loader):
            # Measure time to load data.
            data_time.update(time.time() - end)
            N = images.size(0)
            images = images.to(device)
            for num in targets:
                for keys in num:
                    if torch.is_tensor(num[keys]):
                        num[keys] = num[keys].to(device)
            model.train()
            detections, features, losses = model(images, targets)
            torch.cuda.empty_cache()
            if loss_meter == {}:
                for loss_term in losses:
                    loss_meter[loss_term] = stats.AverageMeter()
            for loss_term in losses:
                loss_meter[loss_term].update(losses[loss_term].item(), N)

            losses = sum(loss for loss in losses.values())

            loss_value = losses.item()

            if not np.isfinite(loss_value):
                logger.error("Loss is {}, stopping training".format(loss_value))
                logger.error(losses)
                sys.exit(1)

            losses = losses / accumulation_steps
            losses.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)

            # Calculate time remaining.
            time_per_epoch = batch_time.avg * len(train_loader)
            epochs_left = args.epochs - epoch - 1
            batches_left = len(train_loader) - i - 1

            time_left = stats.sec_to_str(batches_left * batch_time.avg + epochs_left * time_per_epoch)
            time_elapsed = stats.sec_to_str(batch_time.sum)
            time_estimate = stats.sec_to_str(args.epochs * time_per_epoch)

            end = time.time()

            if i % args.log_freq == 0:
                tmp_str = 'Epoch: [{}/{}] Batch: [{}/{}]  ' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  ' \
                          'Elapsed: {}  ' \
                          'ETA: {} / {}  ' \
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})  '.format(
                    epoch + 1, args.epochs, i, len(train_loader), time_elapsed, time_left, time_estimate,
                    batch_time=batch_time, data_time=data_time)
                for loss_term in loss_meter:
                    tmp_str += '{}: {loss.val:.4f} ({loss.avg:.4f})  '.format(loss_term, loss=loss_meter[loss_term])
                logger.info(tmp_str)

                end = time.time()

        # Validate once per epoch.
        val_score = val_net(model, val_loader, device)
        checkpoint = {'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint, args.checkpoint_path + '/model_{}.pth'.format(epoch))
        if current_best_loss == 0 or sum(val_score.values()) < sum(current_best_loss.values()):
            torch.save(checkpoint, args.checkpoint_path + '/endpoints.pth')
            current_best_loss = val_score

        logger.info('Epoch: [{}]   Val score {}'.format(epoch, val_score))

        if args.sgd:
            lr_scheduler.step()


if __name__ == '__main__':
    args = utils.parse_args('train')
    train_net(args)
