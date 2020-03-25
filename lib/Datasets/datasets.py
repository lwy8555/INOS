import os
import errno
import wget
import zipfile
import glob
import librosa
import scipy
import math
from tqdm import tqdm
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import scipy.io.wavfile as wavf


class Crop_ImageNet:
    """
    Oxford Flower dataset featuring gray-scale 28x28 images of
    Zalando clothing items belonging to ten different classes.
    Dataset implemented with torchvision.datasets.FashionMNIST.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
        class_to_idx (dict): Defines mapping from class names to integers.
    """

    def __init__(self, is_gpu, args):
        self.num_classes = 1000

        self.train_transforms, self.val_transforms = self.__get_transforms(256)

        # self.wnids_to_name = self.get_context()
        self.class_to_idx = {}

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)
    
    def __get_transforms(self, patch_size):
        # optionally scale the images and repeat to three channels
        # important note: these transforms will only be called once during the
        # creation of the dataset and no longer in the incremental datasets that inherit.
        # Adding data augmentation here is thus the wrong place!
        train_transforms = transforms.Compose([
            transforms.RandomCrop(patch_size, patch_size),
            transforms.RandomHorizontalFlip(),
            # transforms.Resize(size=(patch_size, patch_size)),
            transforms.ToTensor(),
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(size=(patch_size, patch_size)),
            transforms.ToTensor(),
        ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.ImageFoder to load dataset.
        Downloads dataset if doesn't exist already.
        url:https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        root = './datasets/ImageNet_cropped_sam' 
        
        trainset = datasets.ImageFolder(root=root+'/train/', transform=self.train_transforms,
                                         target_transform=None)
        valset = datasets.ImageFolder(root=root+'/valid/', transform=self.val_transforms,
                                       target_transform=None)
        self.class_to_idx = trainset.class_to_idx
        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.DataLoader: train_loader, val_loader
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        basename = os.path.basename(path)[:-4]
        score = int(basename.split('_')[-1])
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, score
