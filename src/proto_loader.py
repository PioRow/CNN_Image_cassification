import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import  ImageFolder
import os
import torch
import numpy as np
from torch.utils.data import Sampler, DataLoader
class ProtoCINIC10DataLoader:

    class ProtoSampler(Sampler):
        def __init__(self,labels,k_shot,n_way,query,iters):
            super().__init__()
            self.k_shot = k_shot
            self.n_way = n_way
            self.query = query
            self.iters = iters
            self.labels = np.array(labels)
            self.classes=np.unique(self.labels)
            self.grouped_classes=[np.argwhere(self.labels==c).flatten() for c in self.classes]
        def __iter__(self):

            for _ in range(self.iters):
                selected=torch.randperm(len(self.classes))[:self.n_way]
                res=[]

                for c_s in selected:

                    chosen=np.random.choice(self.grouped_classes[c_s],self.k_shot+self.query,replace=False)

                    res.append(torch.tensor(chosen,dtype=torch.long))

                yield torch.stack(res).view(-1)


        def __len__(self):
            return self.iters

    def __init__(self):
        self.root = os.path.join(os.getcwd(), 'data')
        self.num_workers = 2
        self.seed = 42
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.k_shot = 10
        self.n_way = 10
        self.query = 10
        self.iters = 1000
    def set_root(self, root):
        self.root = root
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
    def set_k_shot(self, k_shot):
        self.k_shot = k_shot
        return self
    def set_n_way(self, n_way):
        self.n_way = n_way
        return self
    def set_query(self, query):
        self.query = query
        return self
    def set_iters(self, iters):
        self.iters = iters
        return self
    def build(self):
        self._prepare_data()
        return self

    def _prepare_data(self):
        self.train_dataset = ImageFolder(root=os.path.join(self.root, 'train'),
                                        transform=self.transform)
        self.val_dataset = ImageFolder(root=os.path.join(self.root, 'valid'),
                                      transform=self.transform)
        self.test_dataset = ImageFolder(root=os.path.join(self.root, 'test'),
                                       transform=self.transform)

    def get_train_loader(self):
        sampler=self.ProtoSampler(self.train_dataset.targets,self.k_shot,self.n_way,self.query,self.iters)
        return DataLoader(dataset=self.train_dataset,batch_sampler=sampler,num_workers=self.num_workers)

    def get_val_loader(self):
        sampler = self.ProtoSampler(self.val_dataset.targets, self.k_shot, self.n_way, self.query, self.iters)
        return DataLoader(dataset=self.val_dataset, batch_sampler=sampler, num_workers=self.num_workers)

    def get_test_loader(self):
        sampler=self.ProtoSampler(self.test_dataset.targets, self.k_shot, self.n_way, self.query, self.iters)
        return DataLoader(dataset=self.test_dataset, batch_sampler=sampler, num_workers=self.num_workers)

