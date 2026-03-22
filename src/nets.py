import torch
import torch.nn as nn
import torch.nn.functional as f

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10,drop_pr=0.0):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        self.dropout = nn.Dropout(p=drop_pr)
        self.fc=nn.Linear(256*4*4,num_classes)

    def forward(self, x):
        x = self.pool(f.relu(self.bn1(self.conv1(x))))
        x = self.pool(f.relu(self.bn2(self.conv2(x))))
        x = self.pool(f.relu(self.bn3(self.conv3(x))))
        x = (f.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x



class MixedCNN(nn.Module):
    def __init__(self, num_classes=10,drop_pr=0.0):
        super(MixedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        self.fc1=nn.Linear(256*4*4,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,num_classes)
        self.dropout = nn.Dropout(p=drop_pr)

    def forward(self, x):
        x = self.pool(f.relu(self.bn1(self.conv1(x))))
        x = self.pool(f.relu(self.bn2(self.conv2(x))))
        x = self.pool(f.relu(self.bn3(self.conv3(x))))
        x = (f.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        x = f.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x



class PrototypicalNet(nn.Module):
    def __init__(self,backbone,embedding_dim):
        super(PrototypicalNet, self).__init__()
        self.backbone = backbone(num_classes=embedding_dim)

    def forward(self,support,query,n_way,k_shot):

        support_embedding = self.backbone(support).view(n_way,k_shot,-1)
        query_embedding = self.backbone(query)

        prototype_reps=support_embedding.mean(dim=1)

        dist=torch.cdist(query_embedding,prototype_reps)
        return -dist


