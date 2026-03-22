
from src.loader import *
from seed import *
from src.trainer import *
def weight_decay_experiment(i,weight_decay,model,legacy_params):


    set_all_seeds(42 + i)
    dl_manager = CINIC10DataLoader() \
        .set_batch_size(legacy_params["batch_size"]) \
        .set_seed(42 + i) \
        .set_transform(transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])) \
        .build()

    trainer = Trainer()
    (
        trainer
        .set_model(model)
        .set_learning_rate(legacy_params["learning_rate"])
        .set_batch_size(legacy_params["batch_size"])
        .set_weight_decay(weight_decay)
        .set_epoch(30)
        .set_train_loader(dl_manager.get_train_loader())
        .set_val_loader(dl_manager.get_val_loader())

    )
    trainer.train(i)
