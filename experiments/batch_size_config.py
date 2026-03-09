
from loader import *
from utils import *
from trainer import *
def batch_size_experiment(i,batch_size,model):

    simple_cnn = model
    set_all_seeds(42 + i)
    dl_manager = CIFAR10DataLoader() \
        .set_batch_size(batch_size) \
        .set_seed(42 + i) \
        .set_transform(transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])) \
        .build()

    trainer = Trainer()
    (
        trainer
        .set_model(simple_cnn)
        .set_batch_size(batch_size)
        .set_epoch(30)
        .set_train_loader(dl_manager.get_train_loader())
        .set_val_loader(dl_manager.get_val_loader())

    )
    trainer.train(i, 20)
