import torch
import torch.optim as optim
import os
import torch.utils.tensorboard as tb
from  tqdm import tqdm
class Trainer(object):
    def __init__(self):
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.epoch=0
        self.learning_rate = 1e-3
        self.weight_decay = 0
        self.dro = 0
        self.batch_size=0


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
    def build(self):
        return self
    def name(self):
        return f"{self.model.__class__.__name__}bs_{self.batch_size}_ep{self.epoch}_lr{self.learning_rate}dp{self.dro}wd_{self.weight_decay}"

    def train(self,run_id,interval,
              optimizer_class=torch.optim.Adam,
              criterion_class=torch.nn.CrossEntropyLoss,):

        total_batches=len(self.train_loader)
        name=self.name()
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs(f'checkpoints/{name}/run_{run_id}', exist_ok=True)
        cp_path=f"checkpoints/{name}/run_{run_id}/model.pth"
        log_file=os.path.join("runs", name, f"run_{run_id}")
        writer=tb.SummaryWriter(log_dir=log_file)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer=optimizer_class(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion=criterion_class()
        best_val_loss = float('inf')

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
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
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
            writer.add_scalar("loss",{
                "train":avg_train_loss,
                "val":avg_val_loss
            },epoch)

            pbar_epoch.set_postfix(val_loss=f"{avg_val_loss:.4f}", train_loss=f"{avg_train_loss:.4%}")

        writer.close()