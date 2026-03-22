from src.loader import CINIC10DataLoader
from src.proto_loader import  ProtoCINIC10DataLoader
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from src.nets import *
import re
name_to_model = {
    "SimpleCNN":SimpleCNN,
    "MixedCNN":MixedCNN,
    "ProtoSimpleCNN":{"model":PrototypicalNet,"args":{"backbone":SimpleCNN,"embedding_dim":512}},
    "ProtoMixedCNN":{"model":PrototypicalNet,"args":{"backbone":MixedCNN,"embedding_dim":128}}
}
def evaluate_model(model_path,test_loader,name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = name.split("b")[0]
    row={"model_name":name}
    for run in os.listdir(model_path):
        experiment_path = os.path.join(model_path,run,"model.pth")
        model=name_to_model[model_name]()
        state_dict = torch.load(experiment_path,map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        tes_tq = tqdm(test_loader,
                          total=len(test_loader),
                          leave=False)
        with torch.no_grad():
            for inputs, labels in tes_tq:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100.0 * correct / total
        row[f"{run}_acc"]=accuracy
    res=[row[f"{run}_acc"] for run in os.listdir(model_path)]

    row["mean_acc"]=np.round(np.mean(res),3)
    row["std_acc"]=np.round(np.std(res),3)
    print(row)
    return row

def evaluate_proto(model_path,model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    match = re.search(r"PrototypicalNetwith_([^_]+)_nway(\d+)_kshot(\d+)", model_name)

    if match:
        backbone_type = match.group(1)
        nway = int(match.group(2))
        kshot = int(match.group(3))
    row={"model_name":model_name}
    dl_manager = ProtoCINIC10DataLoader()
    (
        dl_manager
        .set_n_way(nway)
        .set_k_shot(kshot)
        .set_query(10)
        .build()
    )
    test_loader = dl_manager.get_test_loader()
    for run in os.listdir(model_path):
        experiment_path = os.path.join(model_path,run,"model.pth")
        state_dict = torch.load(experiment_path, map_location=device)
        key="Proto"+backbone_type
        entry=name_to_model[key]
        model=entry["model"](**entry["args"])
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        tes_tq = tqdm(test_loader,
                      total=len(test_loader),
                      leave=False)
        with torch.no_grad():
            for inputs,_ in tes_tq:
                inputs = inputs.view(nway, kshot + 10, 3, 32, 32)

                # Split into Support and Query
                support = inputs[:, :kshot].reshape(-1, 3, 32, 32).to(device)
                query = inputs[:, kshot:].reshape(-1, 3, 32, 32).to(device)
                outputs = model(support,query,nway,kshot)
                targets=torch.arange(nway).repeat_interleave(10).to(device)
                predicted = outputs.argmax(dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = 100.0 * correct / total
        row[f"{run}_acc"] = accuracy

    res = [row[f"{run}_acc"] for run in os.listdir(model_path)]
    row["mean_acc"] = np.round(np.mean(res), 3)
    row["std_acc"] = np.round(np.std(res), 3)
    print(row)
    return row
def evaluate_models(cp_dir,pre_eval):
    dl_manager = CINIC10DataLoader() \
        .build()
    test_loader = dl_manager.get_test_loader()
    data_rows=[]

    for model in os.listdir(cp_dir):

        if model in pre_eval:
            continue
        model_path = os.path.join(cp_dir,model)
        if 'Proto' in model:
            row=evaluate_proto(model_path,model)
        else:
            row=evaluate_model(model_path,test_loader,model)
        data_rows.append(row)
    df = pd.DataFrame(data_rows)
    res_path = os.path.join("evaluation_results","results.csv")
    df.to_csv(res_path,index=False,mode="a",header=False)

if __name__=="__main__":
    df=pd.read_csv(os.path.join("evaluation_results","results.csv"))
    pre_eval=set(df['model_name'])

    evaluate_models("checkpoints",pre_eval)