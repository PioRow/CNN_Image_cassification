import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


class CIFAR10DataLoader:
    def __init__(self):
        self.root = './data'
        self.batch_size = 64
        self.val_split = 0.2
        self.num_workers = 2
        self.seed = 42
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
    def set_seed(self, seed):
        self.seed = seed
        return self

    def build(self):
        self._prepare_data()
        return self


    def _prepare_data(self):
        full_train_dataset = torchvision.datasets.CIFAR10(
            root=self.root, train=True, download=True, transform=self.transform
        )
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.root, train=False, download=True, transform=self.transform
        )
        generator = torch.Generator().manual_seed(self.seed)
        num_train = len(full_train_dataset)
        num_val = int(num_train * self.val_split)
        num_train_final = num_train - num_val
        self.train_subset, self.val_subset = random_split(
            full_train_dataset, [num_train_final, num_val],
            generator
        )

    def get_train_loader(self):
        train_loader = DataLoader(
            self.train_subset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers
        )
        return train_loader
    def get_val_loader(self):
        val_loader = DataLoader(
            self.val_subset, batch_size=self.batch_size,
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