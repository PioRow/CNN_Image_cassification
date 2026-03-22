import torch
import torch.optim as optim
import os
import torch.utils.tensorboard as tb
from torchvision.transforms import  v2
from  tqdm import tqdm
class Trainer(object):
    def __init__(self):
        self.model = None
        self.num_classes= 10
        self.train_loader = None
        self.val_loader = None
        self.epoch=0
        self.learning_rate = 1e-3
        self.weight_decay = 0
        self.dro = 0
        self.batch_size=0
        self.augmentation=None
        self.cutmix_flag=False


    def set_model(self, model):
        self.model = model
        return self
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self
    def set_dropout(self, dro):
        self.dro = dro
        return self
    def set_train_loader(self, train_loader):
        self.train_loader = train_loader
        return self
    def set_val_loader(self, val_loader):
        self.val_loader = val_loader
        return self
    def set_epoch(self, epoch):
        self.epoch = epoch
        return self
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        return self
    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay
        return self
    def set_augmentation(self, augmentation):
        self.augmentation = augmentation
        self.__set_cutmix(augmentation)
        return self
    def set_num_classes(self, num_classes):
        self.num_classes = num_classes
        return self
    def __set_cutmix(self,code):
        self.cutmix_flag= (((code&8)>>3)&1)==1

    def build(self):
        return self
    def name(self):
        name= f"{self.model.__class__.__name__}bs_{self.batch_size}_ep{self.epoch}_lr{self.learning_rate}dp{self.dro}wd_{self.weight_decay}"
        if self.augmentation is not None:
            name+=f"aug_{self.augmentation}"
        return name
    def train(self,run_id,
              optimizer_class=torch.optim.Adam,
              criterion_class=torch.nn.CrossEntropyLoss,):

        cwd=os.getcwd()
        total_batches=len(self.train_loader)
        name=self.name()

        os.makedirs(os.path.join(cwd,'checkpoints',f'{name}',f'run_{run_id}'), exist_ok=True)
        cp_path=os.path.join(cwd,'checkpoints',f'{name}',f'run_{run_id}','model.pth')
        log_file=os.path.join(cwd,'runs', name, f"run_{run_id}")
        writer=tb.SummaryWriter(log_dir=log_file)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer=optimizer_class(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion=criterion_class()
        best_val_loss = float('inf')
        cutMix=None
        if self.cutmix_flag:
            cutMix=v2.CutMix(num_classes=self.num_classes)
        pbar_epoch = tqdm(range(self.epoch), desc="Training Progress")
        for epoch in pbar_epoch:
            train_loss = 0.0
            self.model.train()
            pbar_batch = tqdm(enumerate(self.train_loader),
                              total=total_batches,
                              leave=False,
                              desc=f"Epoch {epoch + 1}")
            for batch_idx, (inputs, targets) in pbar_batch:

                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                if self.cutmix_flag:
                    inputs_mixed,targets_mixed=cutMix(inputs,targets)
                    outputs= self.model(inputs_mixed)
                    loss= criterion(outputs, targets_mixed)
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(self.train_loader)


            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.model(inputs)
                    val_loss += criterion(outputs, targets).item()
            avg_val_loss = val_loss / len(self.val_loader)
            writer.add_scalars("loss",{
                "train":avg_train_loss,
                "val":avg_val_loss
            },epoch)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), cp_path)
            pbar_epoch.set_postfix(val_loss=f"{avg_val_loss:.4f}", train_loss=f"{avg_train_loss:.4f}")

        writer.close()