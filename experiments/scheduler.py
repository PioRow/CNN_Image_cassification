
from experiments.batch_size_config import *
from nets import *

def schedule_batch_experiment():
    for i in range(3):
        batch_size_experiment(i,64,SimpleCNN())
    for i in range(3):
        batch_size_experiment(i, 64, MixedCNN())
    for i in range(3):
        batch_size_experiment(i, 16, SimpleCNN())
    for i in range(3):
        batch_size_experiment(i, 16, MixedCNN())
