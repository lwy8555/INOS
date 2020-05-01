########################
# Importing libraries
########################
# System libraries
import os
import random
from time import gmtime, strftime
import numpy as np
import pickle
import copy
import pdb

# Tensorboard for PyTorch logging and visualization
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

# Torch libraries
import torch
import torch.backends.cudnn as cudnn

# Custom library
import lib.Models.architectures as architectures
import lib.Datasets.datasets as datasets
from lib.Models.initialization import WeightInit
from lib.cmdparser import parser
from lib.Training.train import train
from lib.Training.validate import validate
from lib.Training.loss_functions import loss_function as criterion
from lib.Utility.utils import save_checkpoint, save_task_checkpoint
from lib.Utility.visualization import args_to_tensorboard


#python main.py --pretrained --inos -d
# Comment this if CUDNN benchmarking is not desired
cudnn.benchmark = True


def main():
    # Command line options
    args = parser.parse_args()
    print("Command line options:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    if args.debug:
        pdb.set_trace()

    # Check whether GPU is available and can be used
    # if CUDA is found then device is set accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Launch a writer for the tensorboard summary writer instance
    save_path = 'runs/' + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + '_' + args.dataset + '_' + args.architecture

    # if we are resuming a previous training, note it in the name
    if args.resume:
        save_path = save_path + '_resumed'
    writer = SummaryWriter(save_path)

    # saving the parsed args to file
    log_file = os.path.join(save_path, "stdout")
    log = open(log_file, "a")
    for arg in vars(args):
        log.write(arg + ':' + str(getattr(args, arg)) + '\n')

    # Dataset loading
    data_init_method = getattr(datasets, args.dataset)
    dataset = data_init_method(torch.cuda.is_available(), args)
    # get the number of classes from the class dictionary
    num_classes = dataset.num_classes

    # we set an epoch multiplier to 1 for isolated training and increase it proportional to amount of tasks in CL
    epoch_multiplier = 1

    # add command line options to TensorBoard
    args_to_tensorboard(writer, args)
    log.close()
    
    # build the model
    model = architectures.Inos_model(1000, args)

    # Parallel container for multi GPU use and cast to available device
    model = torch.nn.DataParallel(model).to(device)
    print(model)

    # Initialize the weights of the model, by default according to He et al.
    print("Initializing network with: " + args.weight_init)
    WeightInitializer = WeightInit(args.weight_init)
    WeightInitializer.init_model(model)

    # Define optimizer and loss function (criterion)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,80,100], gamma=0.5)

    epoch = 0
    best_prec = 0
    best_loss = random.getrandbits(128)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # optimize until final amount of epochs is reached. Final amount of epochs is determined through the
    while epoch < (args.epochs * epoch_multiplier):
        if epoch+2 == epoch%args.epochs:
            print("debug perpose")

        # train
        train(dataset, model, criterion, epoch, optimizer, writer, device, args)

        # evaluate on validation set
        prec, loss = validate(dataset, model, criterion, epoch, writer, device, save_path, args)

        # remember best prec@1 and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        best_prec = max(prec, best_prec)
        save_checkpoint({'epoch': epoch,
                         'arch': args.architecture,
                         'state_dict': model.state_dict(),
                         'best_prec': best_prec,
                         'best_loss': best_loss,
                         'optimizer': optimizer.state_dict()},
                        is_best, save_path)

        # increment epoch counters
        epoch += 1
        scheduler.step()

    writer.close()


if __name__ == '__main__':     
    main()
