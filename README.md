# Convolutional Neural Network

this project investigates influence of different CNN architectures,
hyper-parameters and augmentation techniques on models effectiveness.

This research is done using CINIC-10 dataset.

# requirements:
- python 3.13
- Nvidia GPU and matching CUDA drivers advised
### Usage:
- download CINIC-10 dataset end extract it into ```data``` directory
- install all dependencies listed in ```requirements.txt``` file
- to conduct training execute ```main.py``` script
- to create plots run ```plotter.py``` script. It will output graphs showcasing training and validation loss accross all epochs
for every considered model
- to evaluate model on testing dataset, execute script ```evaluator.py```
it will create csv file listing accuracies of all models across 3 independent runs,
as well as aggregated mean and standard deviation