from torch.utils.data import Dataset
import numpy as np

class CifarDataset(Dataset):
    def __init__(self, data, labels):
        """
        Initializes the dataset with data and labels.

        Args:
            data (np.ndarray): Array of shape (N, C, H, W) containing the images.
            labels (np.ndarray): Array of shape (N,) containing the labels.
        """
        self.data = data
        self.labels = labels


    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns a sample and its label at the specified index."""
        return self.data[idx], self.labels[idx]