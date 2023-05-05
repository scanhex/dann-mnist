from enum import Enum

from ada.datasets.dataset_cmnist import CMNIST
from torchvision.datasets import MNIST, SVHN
import ada.datasets.preprocessing as proc
from ada.datasets.dataset_usps import USPS
from ada.datasets.dataset_mnistm import MNISTM
from ada.datasets.dataset_access import DatasetAccess


class DigitDataset(Enum):
    MNIST = "MNIST"
    MNISTM = "MNISTM"
    USPS = "USPS"
    SVHN = "SVHN"
    CMNIST = "CMNIST"

    @staticmethod
    def get_accesses(source: "DigitDataset", target: "DigitDataset", data_path):
        channel_numbers = {
            DigitDataset.MNIST: 1,
            DigitDataset.MNISTM: 3,
            DigitDataset.USPS: 1,
            DigitDataset.SVHN: 3,
            DigitDataset.CMNIST: 3,
        }

        transform_names = {
            (DigitDataset.MNIST, 1): "mnist32",
            (DigitDataset.MNIST, 3): "mnist32rgb",
            (DigitDataset.MNISTM, 3): "mnistm",
            (DigitDataset.USPS, 1): "usps32",
            (DigitDataset.USPS, 3): "usps32rgb",
            (DigitDataset.SVHN, 3): "svhn",
            (DigitDataset.CMNIST, 3): "cmnist",
        }

        factories = {
            DigitDataset.MNIST: MNISTDatasetAccess,
            DigitDataset.MNISTM: MNISTMDatasetAccess,
            DigitDataset.USPS: USPSDatasetAccess,
            DigitDataset.SVHN: SVHNDatasetAccess,
            DigitDataset.CMNIST: CMNISTDatasetAccess,
        }

        # handle color/nb channels
        num_channels = max(channel_numbers[source], channel_numbers[target])
        source_tf = transform_names[(source, num_channels)]
        target_tf = transform_names[(target, num_channels)]

        return (
            factories[source](data_path, source_tf),
            factories[target](data_path, target_tf),
            num_channels,
        )


class DigitDatasetAccess(DatasetAccess):
    def __init__(self, data_path, transform_kind):
        super().__init__(n_classes=10)
        self._data_path = data_path
        self._transform = proc.get_transform(transform_kind)


class MNISTDatasetAccess(DigitDatasetAccess):
    def get_train(self):
        return MNIST(
            self._data_path, train=True, transform=self._transform, download=True
        )

    def get_test(self):
        return MNIST(
            self._data_path, train=False, transform=self._transform, download=True
        )


class MNISTMDatasetAccess(DigitDatasetAccess):
    def get_train(self):
        return MNISTM(
            self._data_path, train=True, transform=self._transform, download=True
        )

    def get_test(self):
        return MNISTM(
            self._data_path, train=False, transform=self._transform, download=True
        )


class USPSDatasetAccess(DigitDatasetAccess):
    def get_train(self):
        return USPS(
            self._data_path, train=True, transform=self._transform, download=True
        )

    def get_test(self):
        return USPS(
            self._data_path, train=False, transform=self._transform, download=True
        )


class SVHNDatasetAccess(DigitDatasetAccess):
    def get_train(self):
        return SVHN(
            self._data_path, split="train", transform=self._transform, download=True
        )

    def get_test(self):
        return SVHN(
            self._data_path, split="test", transform=self._transform, download=True
        )


class CMNISTDatasetAccess(DigitDatasetAccess):
    def get_train(self):
        return CMNIST(
            self._data_path, train=True, transform=self._transform, download=True
        )

    def get_test(self):
        return CMNIST(
            self._data_path, train=False, transform=self._transform, download=True
        )
