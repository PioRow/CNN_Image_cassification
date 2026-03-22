from src.loader import *
from seed import *
from src.trainer import *

def build_augmentation_pipeline(aug_code):
    res=[]
    # 1 bit Flip
    # 2 bit Random Rotation
    # 3 bit Color Jutter
    if (aug_code&1)==1:
        res.append(transforms.RandomHorizontalFlip())
    if ((aug_code&2)>>1)==1:
        res.append(transforms.RandomRotation(30))
    if ((aug_code&4)>>2)==1:
        res.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

    res.append(transforms.ToTensor())
    res.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return res
def augmentation_experiment(i, aug_code, model, legacy_params,epochs=30):
    trasforms_list=build_augmentation_pipeline(aug_code)

    set_all_seeds(42 + i)
    dl_manager = CINIC10DataLoader() \
        .set_batch_size(legacy_params["batch_size"]) \
        .set_seed(42 + i) \
        .set_transform(transforms.Compose(trasforms_list)) \
        .build()

    trainer = Trainer()
    (
        trainer
        .set_model(model)
        .set_learning_rate(legacy_params["learning_rate"])
        .set_batch_size(legacy_params["batch_size"])
        .set_weight_decay(legacy_params["weight_decay"])
        .set_dropout(legacy_params["dropout"])
        .set_augmentation(aug_code)
        .set_epoch(epochs)
        .set_train_loader(dl_manager.get_train_loader())
        .set_val_loader(dl_manager.get_val_loader())

    )
    trainer.train(i)