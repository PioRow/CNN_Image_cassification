import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import  ImageFolder

class CINIC10DataLoader:

    def __init__(self):
        self.root =  os.path.join(os.getcwd(), 'data')
        self.batch_size = 64
        self.val_split = 0.2
        self.num_workers = 2
        self.seed = 42
        self.default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def set_root(self, root):
        self.root = root
        return self
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self
    def set_val_split(self, val_split):
        self.val_split = val_split
        return self
    def set_num_workers(self, num_workers):
        self.num_workers = num_workers
        return self
    def set_transform(self, transform):
        self.transform = transform
        return self
    def set_default_transform(self, transform):
        self.default_transform = transform
        return self
    def set_seed(self, seed):
        self.seed = seed
        return self
    def build(self):
        self._prepare_data()
        return self


    def _prepare_data(self):
        self.train_dataset = ImageFolder(root=os.path.join(self.root, 'train'),
                                        transform=self.transform)
        self.val_dataset = ImageFolder(root=os.path.join(self.root, 'valid'),
                                      transform=self.default_transform)
        self.test_dataset = ImageFolder(root=os.path.join(self.root, 'test'),
                                       transform=self.default_transform)

    def get_train_loader(self):
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers
        )
        return train_loader
    def get_val_loader(self):
        val_loader = DataLoader(
            self.val_dataset , batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers
        )
        return val_loader
    def get_test_loader(self):
        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers
        )
        return test_loader

    @property
    def classes(self):
        return ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')