import torch
import torchvision
import os
import math
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap

# matplotlib backend, required for plotting of images to tensorboard
matplotlib.use('Agg')

# setting font sizes
title_font_size = 60
axes_font_size = 45
legend_font_size = 36
ticks_font_size = 48

# setting seaborn specifics
sns.set(font_scale=2.5)
sns.set_style("whitegrid")
colors = sns.color_palette("Set2")
pal = sns.cubehelix_palette(10, light=0.0)
linestyles = [(0, (1, 3)),  # 'dotted'
              (0, (1, 1)),  # 'densely dotted'
              (0, (2, 2)),  # 'dashed'
              (0, (3, 1)),  # 'densely dashed'
              (0, (3, 3, 1, 3)),  # 'dashdotted'
              (0, (3, 1, 1, 1)),  # 'densely dashdotted'
              (0, (3, 3, 1, 3, 1, 3)),  # 'dashdotdotted'
              (0, (3, 1, 1, 1, 1, 1))]  # 'densely dashdotdotted'


def args_to_tensorboard(writer, args):
    """
    Takes command line parser arguments and formats them to
    display them in TensorBoard text.

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        args (dict): dictionary of command line arguments
    """

    txt = ""
    for arg in vars(args):
        txt += arg + ": " + str(getattr(args, arg)) + "<br/>"

    writer.add_text('command_line_parameters', txt, 0)


def visualize_image_grid(images, writer, count, name, save_path):
    """
    Visualizes a grid of images and saves it to both hard-drive as well as TensorBoard

    Parameters:
        images (torch.Tensor): Tensor of images.
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        count (int): counter usually specifying steps/epochs/time.
        name (str): name of the figure in tensorboard.
        save_path (str): path where image grid is going to be saved.
    """
    size = images.size(0)
    # imgs = torchvision.utils.make_grid(images, nrow=int(math.sqrt(size)), padding=5)
    imgs = torchvision.utils.make_grid(images, nrow=int(math.sqrt(size)), padding=5, normalize=True, range=(-1,1))
    torchvision.utils.save_image(images, os.path.join(save_path, name + '_epoch_' + str(count + 1) + '.png'),
                                 nrow=int(math.sqrt(size)), padding=5)#, normalize=True, range=(-1,1))
    writer.add_image(name, imgs, count)


def visualize_class_image_grid(images, writer, count, name, save_path, iter):
    """
    Visualizes a grid of images and saves it to both hard-drive as well as TensorBoard

    Parameters:
        images (torch.Tensor): Tensor of images.
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        count (int): counter usually specifying steps/epochs/time.
        name (str): name of the figure in tensorboard.
        save_path (str): path where image grid is going to be saved.
    """
    size = images.size(0)
    # imgs = torchvision.utils.make_grid(images, nrow=int(math.sqrt(size)), padding=5)
    imgs = torchvision.utils.make_grid(images, nrow=int(math.sqrt(size)), padding=5, normalize=True, range=(-1,1))
    torchvision.utils.save_image(images, os.path.join(save_path, name + '_epoch_' + str(count + 1)+"_"+str(iter) + '.png'),
                                 nrow=int(math.sqrt(size)), padding=5)#, normalize=True, range=(-1,1))
    writer.add_image(name, imgs, count)


def visualize_confusion(writer, step, matrix, class_dict, save_path):
    """
    Visualization of confusion matrix. Is saved to hard-drive and TensorBoard.

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        step (int): Counter usually specifying steps/epochs/time.
        matrix (numpy.array): Square-shaped array of size class x class.
            Should specify cross-class accuracies/confusion in percent
            values (range 0-1).
        class_dict (dict): Dictionary specifying class names as keys and
            corresponding integer labels/targets as values.
        save_path (str): Path used for saving
    """

    all_categories = sorted(class_dict, key=class_dict.get)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax, boundaries=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Turn off the grid for this plot
    ax.grid(False)
    plt.tight_layout()

    writer.add_figure("Training data", fig, global_step=str(step))
    plt.savefig(os.path.join(save_path, 'confusion_epoch_' + str(step) + '.png'), bbox_inches='tight')

def visualize_classification_scores(data, other_data_dicts, dict_key, data_name, save_path):
    """
    Visualization of classification scores per dataset.

    Parameters:
        data (list): Classification scores.
        other_data_dicts (dictionary of dictionaries): Dictionary of key-value pairs per dataset
        dict_key (string): Dictionary key to plot
        data_name (str): Original trained dataset's name.
        save_path (str): Saving path.
    """

    data = [y for x in data for y in x]

    plt.figure(figsize=(20, 20))
    plt.hist(data, label=data_name, alpha=1.0, bins=20, color=colors[0])

    c = 0
    for other_data_name, other_data_dict in other_data_dicts.items():
        other_data = [y for x in other_data_dict[dict_key] for y in x]
        plt.hist(other_data, label=other_data_name, alpha=0.5, bins=20, color=colors[c])
        c += 1

    plt.title("Dataset classification", fontsize=title_font_size)
    plt.xlabel("Classification confidence", fontsize=axes_font_size)
    plt.ylabel("Number of images", fontsize=axes_font_size)
    plt.legend(loc=0)
    plt.xlim(left=-0.0, right=1.05)

    plt.savefig(os.path.join(save_path, data_name + '_' + ",".join(list(other_data_dicts.keys()))
                             + '_classification_scores.png'),
                bbox_inches='tight')
