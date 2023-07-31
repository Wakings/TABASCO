#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import torch
import time
from utils.util import *
from models.metric import *
def validate(args,val_loader, model, criterion, device,epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # init confusion_matrix
    conf_matrix = torch.zeros(args.num_classes, args.num_classes)

    for i, (input, target) in enumerate(val_loader):
        input = set_tensor(input, False, device)
        target = set_tensor(target,False,device)


        # compute output
        with torch.no_grad():
            output = model(input)
        loss = criterion(output, target)


        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard

    return conf_matrix,losses.avg,top1.avg





