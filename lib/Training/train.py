import time
import torch
import torch.nn as nn
import copy
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import accuracy

def train(Dataset, model, criterion, epoch, optimizer, writer, device, args):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        Dataset (torch.utils.data.Dataset): The dataset
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        epoch (int): Continuous epoch counter
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        writer (tensorboard.SummaryWriter): TensorBoard writer instance
        device (str): device name where data is transferred to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int) and log_weights (bool).
    """

    # Create instances to accumulate losses etc.
    losses = AverageMeter()
    class_losses = AverageMeter()
    inos_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    # train
    for i, (inp, target) in enumerate(Dataset.train_loader):
        inp = inp.to(device)
        target = target.to(device)

        class_target = target[0]

        # measure data loading time
        data_time.update(time.time() - end)

        # compute model forward
        output, score = model(inp)

        # calculate loss
        cl, rl = criterion(output, target, score, device, args)
        loss = cl + rl

        # record precision/accuracy and losses
        prec1 = accuracy(output, class_target)[0]
        top1.update(prec1.item(), inp.size(0))
        class_losses.update(cl.item(), inp.size(0))
        inos_losses.update(rl.item(), inp.size(0))
        losses.update(loss.item(), inp.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        if i % args.print_freq == 0:
            print('Training: [{0}][{1}/{2}]\t' 
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  .format(
                   epoch+1, i, len(Dataset.train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

    # TensorBoard summary logging
    writer.add_scalar('train/precision@1', top1.avg, epoch)
    writer.add_scalar('train/average_loss', losses.avg, epoch)
    writer.add_scalar('train/class_loss',class_losses.avg, epoch)
    writer.add_scalar('train/inos_loss', inos_losses.avg, epoch)

    # If the log weights argument is specified also add parameter and gradient histograms to TensorBoard.
    if args.log_weights:
        # Histograms and distributions of network parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(tag, value.data.cpu().numpy(), epoch, bins="auto")
            # second check required for buffers that appear in the parameters dict but don't receive gradients
            if value.requires_grad and value.grad is not None:
                writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch, bins="auto")

    print(' * Train: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))
