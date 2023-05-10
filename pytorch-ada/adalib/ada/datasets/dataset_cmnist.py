"""Dataset setting and data loader for CMNIST
Modified from the USPS dataset file
"""
import gzip
import os
import pickle
import urllib
import logging

import numpy as np
import torch
import torch.utils.data as data


class CMNIST(data.Dataset):
    """CMNIST Dataset.

    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    path = '../../datasets/cmnist/fgbg_cmnist_cpr0.5-0.5'

    def __init__(self, root, train=True, transform=None, download=False):
        """Init USPS dataset."""
        self.transform = transform
        self.dataset_size = None
        self.train = train
        self.fout = 'C:/Users/al_mo/Code/mit/6.S052/dann-mnist/cmnist.torch'

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            print(f'Path: {os.path.abspath(os.path.join(self.path, "train_x.npy"))}')
            raise RuntimeError(
                "CMNIST dataset not found. Generate it using gen_color_mnist.py"
            )

        self.data, self.targets = self.load_samples()
        self.targets = torch.LongTensor(self.targets)
        if self.train:
            total_num_samples = self.data.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.data = self.data[indices[0 : self.dataset_size], ::]
            self.targets = self.targets[indices[0 : self.dataset_size]]
        # self.train_data *= 255.0  # TODO check bug
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.data[index, ::], self.targets[index]
        if self.transform is not None:
            img: torch.Tensor = self.transform(img)
            torch.save(img, self.fout)
        label = torch.LongTensor([np.int64(label).item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.path, "train_x.npy"))

    def download(self):
        # TODO download it right here instead of using external script
        pass

    def load_samples(self):
        """Load sample images from dataset."""
        train_x_filename = os.path.join(self.path, "train_x.npy")
        train_y_filename = os.path.join(self.path, "train_y.npy")
        test_x_filename = os.path.join(self.path, "test_x.npy")
        test_y_filename = os.path.join(self.path, "test_y.npy")
        train_x = np.float32(np.load(train_x_filename))
        train_y = (np.load(train_y_filename))
        test_x = np.float32(np.load(test_x_filename))
        test_y = (np.load(test_y_filename))
        if self.train:
            images = train_x
            labels = train_y
            self.dataset_size = labels.shape[0]
        else:
            images = test_x
            labels = test_y
            self.dataset_size = labels.shape[0]
        return images, labels
