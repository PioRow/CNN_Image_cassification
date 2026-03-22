import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import models
from tensorboard.backend.event_processing import event_accumulator
from nets import *
from torch.utils.tensorboard import SummaryWriter
def save_plots_for_models(log_dir):
    for model_name in os.listdir(log_dir):
        model_path = os.path.join(log_dir, model_name)
        save_path = os.path.join("figs", model_name + ".png")
        if os.path.isfile(save_path):
            continue
        plt.figure(figsize=(10, 6))
        for timestamp_folder in os.listdir(model_path):

            run_path = os.path.join(model_path, timestamp_folder)

            for loss,name in zip(["loss_train","loss_val"],["training","validation"]):
                loss_path = os.path.join(run_path, loss)

                ea=event_accumulator.EventAccumulator(loss_path)
                ea.Reload()
                tag="loss"       
                data = ea.Scalars(tag)
                df = pd.DataFrame(data)

                label = f"{timestamp_folder} {name} loss"
                plt.plot(df["step"], df["value"], label=label)



        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_path = os.path.join("figs",model_name+".png")
        plt.savefig(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

def plot_nets():
    dummy_input = torch.randn(1, 3, 32, 32)
    for(m,n) in [(SimpleCNN(),"SimpleCNN"),(MixedCNN(),"MixedCNN"),(models.resnet18(),"ResNet18")]:
        writer=SummaryWriter(f"nets/{n}")
        writer.add_graph(m,dummy_input)
        writer.close()

if __name__ == "__main__":
    save_plots_for_models("runs")
    #plot_nets()
