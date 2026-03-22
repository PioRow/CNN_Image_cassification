
from experiments.batch_size_config import *
from src.nets import *
from experiments.learning_rate_config import *
from experiments.weight_decay_config import *
from experiments.dropout_config import *
from experiments.augmentation_config import *
from experiments.prototype_config import *
def schedule_batch_experiment():
    for i in range(3):
        batch_size_experiment(i,64,SimpleCNN())
    for i in range(3):
        batch_size_experiment(i, 64, MixedCNN())
    for i in range(3):
        batch_size_experiment(i, 16, SimpleCNN())
    for i in range(3):
        batch_size_experiment(i, 16, MixedCNN())
    for i in range(3):
        batch_size_experiment(i, 128, SimpleCNN())
    for i in range(3):
        batch_size_experiment(i, 128, MixedCNN())
def schedule_learning_rate_experiment():
    params={
        "SimpleCNN":{"batch_size":64},
        "MixedCNN":{"batch_size":64},
    }
    for i in range(3):
        learning_rate_experiment(i,1e-2,SimpleCNN(),params["SimpleCNN"])
    for i in range(3):
        learning_rate_experiment(i,1e-5,SimpleCNN(),params["SimpleCNN"])
    for i in range(3):
        learning_rate_experiment(i,1e-2,MixedCNN(),params["MixedCNN"])
    for i in range(3):
        learning_rate_experiment(i,1e-5,MixedCNN(),params["MixedCNN"])

def schedule_weight_decay_experiment():
    params={
        "SimpleCNN":{"batch_size":64,"learning_rate":1e-3},
        "MixedCNN":{"batch_size":64,"learning_rate":1e-3},
    }
    for i in range(3):
        weight_decay_experiment(i,1e-3,SimpleCNN(),params["SimpleCNN"])
    for i in range(3):
        weight_decay_experiment(i,1e-4,SimpleCNN(),params["SimpleCNN"])
    for i in range(3):
        weight_decay_experiment(i,1e-3,MixedCNN(),params["MixedCNN"])
    for i in range(3):
        weight_decay_experiment(i,1e-4,MixedCNN(),params["MixedCNN"])


def schedule_dropout_experiment():
    params = {
        "SimpleCNN": {"batch_size": 64, "learning_rate": 1e-3,'weight_decay': 1e-3},
        "MixedCNN": {"batch_size": 64, "learning_rate": 1e-3,'weight_decay': 1e-3},
    }
    for i in range(3):
        dropout_experiment(i,0.3,SimpleCNN(drop_pr=0.3),params["SimpleCNN"])
    for i in range(3):
        dropout_experiment(i,0.5,SimpleCNN(drop_pr=0.5),params["SimpleCNN"])
    for i in range(3):
        dropout_experiment(i,0.3,MixedCNN(drop_pr=0.3),params["MixedCNN"])
    for i in range(3):
        dropout_experiment(i,0.5,MixedCNN(drop_pr=0.5),params["MixedCNN"])

def schedule_basic_augmentation_experiment():
    params = {
        "SimpleCNN": {"batch_size": 64, "learning_rate": 1e-3, 'weight_decay': 1e-3,'dropout': 0.5},
        "MixedCNN": {"batch_size": 64, "learning_rate": 1e-3, 'weight_decay': 1e-3,'dropout': 0.3},
    }
    for i in range(3):
        augmentation_experiment(i, 1, SimpleCNN(drop_pr=0.5), params["SimpleCNN"])
    for i in range(3):
        augmentation_experiment(i, 2, SimpleCNN(drop_pr=0.5), params["SimpleCNN"])
    for i in range(3):
        augmentation_experiment(i, 4, SimpleCNN(drop_pr=0.5), params["SimpleCNN"])
    for i in range(3):
        augmentation_experiment(i, 7, SimpleCNN(drop_pr=0.5), params["SimpleCNN"])

    for i in range(3):
        augmentation_experiment(i, 1, MixedCNN(drop_pr=0.3), params["MixedCNN"])
    for i in range(3):
        augmentation_experiment(i, 2, MixedCNN(drop_pr=0.3), params["MixedCNN"])
    for i in range(3):
        augmentation_experiment(i, 4, MixedCNN(drop_pr=0.3), params["MixedCNN"])
    for i in range(3):
        augmentation_experiment(i, 7, MixedCNN(drop_pr=0.3), params["MixedCNN"])

def schedule_advanced_augmentation_experiment():
    params = {
        "SimpleCNN": {"batch_size": 64, "learning_rate": 1e-3, 'weight_decay': 1e-3, 'dropout': 0.5},
        "MixedCNN": {"batch_size": 64, "learning_rate": 1e-3, 'weight_decay': 1e-3, 'dropout': 0.3},
    }
    for i in range(3):
        augmentation_experiment(i, 8, SimpleCNN(drop_pr=0.5), params["SimpleCNN"])
    for i in range(3):
        augmentation_experiment(i, 5, SimpleCNN(drop_pr=0.5), params["SimpleCNN"])
    for i in range(3):
        augmentation_experiment(i, 13, SimpleCNN(drop_pr=0.5), params["SimpleCNN"])

    for i in range(3):
        augmentation_experiment(i, 8, MixedCNN(drop_pr=0.3), params["MixedCNN"])
    for i in range(3):
        augmentation_experiment(i, 5, MixedCNN(drop_pr=0.3), params["MixedCNN"])
    for i in range(3):
        augmentation_experiment(i, 13, MixedCNN(drop_pr=0.3), params["MixedCNN"])

def schedule_proto_experiment():
    for i in range(3):
        prototype_experiment(i,PrototypicalNet(SimpleCNN,512),5,5,10)
    for i in range(3):
        prototype_experiment(i,PrototypicalNet(SimpleCNN,512),5,1,10)
    for i in range(3):
        prototype_experiment(i,PrototypicalNet(MixedCNN,128),5,5,10)
    for i in range(3):
        prototype_experiment(i,PrototypicalNet(MixedCNN,128),5,1,10)

