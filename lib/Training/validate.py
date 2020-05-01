import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import ConfusionMeter
from lib.Utility.metrics import accuracy
from lib.Utility.visualization import visualize_confusion
from lib.Utility.visualization import visualize_image_grid
from lib.Utility.visualization import visualize_class_image_grid 


def validate(Dataset, model, criterion, epoch, writer, device, save_path, args):
    """
    Evaluates/validates the model

    Parameters:
        Dataset (torch.utils.data.Dataset): The dataset
        model (torch.nn.module): Model to be evaluated/validated
        criterion (torch.nn.criterion): Loss function
        epoch (int): Epoch counter
        writer (tensorboard.SummaryWriter): TensorBoard writer instance
        device (str): device name where data is transferred to
        save_path (str): path to save data to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int), epochs (int), incremental_data (bool), autoregression (bool),
            visualization_epoch (int), cross_dataset (bool), num_base_tasks (int), num_increment_tasks (int) and
            patch_size (int).

    Returns:
        float: top1 precision/accuracy
        float: average loss
    """

    # initialize average meters to accumulate values
    losses = AverageMeter()
    class_losses = AverageMeter()
    inos_losses = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # confusion matrix
    confusion = ConfusionMeter(model.module.num_classes, normalized=True)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # evaluate the entire validation dataset
    with torch.no_grad():
        for i, (inp, target) in enumerate(Dataset.val_loader):
            inp = inp.to(device)
            target = target.to(device)

            recon_target = inp
            class_target = target[0]

            # compute output
            output, score = model(inp)

            # compute loss
            cl,rl = criterion(output, target, score, device, args)
            loss = cl + rl

            # measure accuracy, record loss, fill confusion matrix
            prec1 = accuracy(output, class_target)[0]
            top1.update(prec1.item(), inp.size(0))
            class_losses.update(cl.item(), inp.size(0))
            inos_losses.update(rl.item(), inp.size(0))
            confusion.add(output.data, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), inp.size(0))

            # Print progress
            if i % args.print_freq == 0:
                print('Validate: [{0}][{1}/{2}]\t' 
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                       .format(
                       epoch+1, i, len(Dataset.val_loader), batch_time=batch_time, loss=losses, 
                       top1=top1))

    # TensorBoard summary logging
    writer.add_scalar('validation/precision@1', top1.avg, epoch)
    writer.add_scalar('validation/average_loss', losses.avg, epoch)
    writer.add_scalar('validation/class_loss',class_losses.avg, epoch)
    writer.add_scalar('validation/inos_loss', inos_losses.avg, epoch)

    print(' * Validation: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))

    # At the end of training isolated, or at the end of every task visualize the confusion matrix
    if (epoch + 1) % args.epochs == 0 and epoch > 0:
        # visualize the confusion matrix
        visualize_confusion(writer, epoch + 1, confusion.value(), Dataset.class_to_idx, save_path)

    return top1.avg, losses.avg