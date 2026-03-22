from src.loader import *
from seed import *
from src.proto_trainer import ProtoTrainer
from src.trainer import *
from src.nets import *
from src.proto_loader import ProtoCINIC10DataLoader
def prototype_experiment(i,model,n_way,k_shot,query):
    set_all_seeds(42+i)
    dl_manager = ProtoCINIC10DataLoader()
    (
        dl_manager
        .set_seed(42+i)
        .set_n_way(n_way)
        .set_k_shot(k_shot)
        .set_query(query)
        .build()
    )
    proto_trainer = ProtoTrainer()
    (
        proto_trainer
        .set_train_loader(dl_manager.get_train_loader())
        .set_val_loader(dl_manager.get_val_loader())
        .set_model(model)
        .set_em_dim(512)
        .set_n_way(n_way)
        .set_k_shot(k_shot)
        .set_query(query)
        .set_epoch(30)
        .build()
    )
    proto_trainer.train(i)