import torch
import os
import torch
import torch.optim as optim
import os
import torch.utils.tensorboard as tb
from torchvision.transforms import  v2
from  tqdm import tqdm
class ProtoTrainer:
    def __init__(self):
        self.model=None
        self.learning_rate=1e-3
        self.train_loader=None
        self.val_loader=None
        self.em_dim=None
        self.k_shot=10
        self.n_way=10
        self.query=10
        self.epoch=30
    def set_train_loader(self, train_loader):
        self.train_loader = train_loader
        return self
    def set_val_loader(self, val_loader):
        self.val_loader = val_loader
        return self
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        return self
    def set_epoch(self, epoch):
        self.epoch = epoch
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
    def set_em_dim(self, em_dim):
        self.em_dim = em_dim
        return self
    def build(self):
        return self

    def set_model(self, model):
        self.model = model
        return self
    def name(self):
        name = f"{self.model.__class__.__name__}with_{self.model.backbone.__class__.__name__}_nway{self.n_way}_kshot{self.k_shot}_lr{self.learning_rate}"
        return name


    def train(self,run_id,
              optimizer_class=torch.optim.Adam,
              criterion_class=torch.nn.CrossEntropyLoss):
        cwd = os.getcwd()
        total_batches = len(self.train_loader)
        name = self.name()
        best_val_loss = float('inf')
        os.makedirs(os.path.join(cwd, 'checkpoints', f'{name}', f'run_{run_id}'), exist_ok=True)
        cp_path = os.path.join(cwd, 'checkpoints', f'{name}', f'run_{run_id}', 'model.pth')
        log_file = os.path.join(cwd, 'runs', name, f"run_{run_id}")
        writer = tb.SummaryWriter(log_dir=log_file)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer = optimizer_class(self.model.parameters(), lr=self.learning_rate)
        pbar_epoch = tqdm(range(self.epoch), desc="Training Progress")
        criterion=criterion_class()
        for epoch in pbar_epoch:
            train_loss = 0.0
            self.model.train()
            pbar_batch = tqdm(enumerate(self.train_loader),
                              total=total_batches,
                              leave=False,
                              desc=f"Epoch {epoch + 1}")
            for batch_idx, (inputs, _) in pbar_batch:
                inputs.to(device)
                inputs = inputs.view(self.n_way,self.k_shot+self.query,3,32,32)
                support=inputs[:,:self.k_shot].reshape(-1,3,32,32)
                query=inputs[:,self.k_shot:].reshape(-1,3,32,32)


                support=support.to(device)
                query=query.to(device)
                optimizer.zero_grad()
                output=self.model(support,query,self.n_way,self.k_shot)
                targets=torch.arange(self.n_way).repeat_interleave(self.query).to(device)

                loss=criterion(output,targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(self.train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs.to(device)
                    inputs = inputs.view(self.n_way, self.k_shot + self.query, 3, 32, 32)
                    support = inputs[:, :self.k_shot].reshape(-1, 3, 32, 32)
                    query = inputs[:, self.k_shot:].reshape(-1, 3, 32, 32)

                    support = support.to(device)
                    query = query.to(device)

                    output = self.model(support, query, self.n_way, self.k_shot)
                    targets = torch.arange(self.n_way).repeat_interleave(self.query).to(device)
                    loss = criterion(output, targets)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(self.val_loader)
            writer.add_scalars("loss", {
                "train": avg_train_loss,
                "val": avg_val_loss
            }, epoch)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), cp_path)
            pbar_epoch.set_postfix(val_loss=f"{avg_val_loss:.4f}", train_loss=f"{avg_train_loss:.4f}")

        writer.close()
