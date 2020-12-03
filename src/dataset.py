import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import Dataset

_FILE_PATH = "data/processed"
_FILE_NAME = "70man"
_DAY = 96

class ASPDataset(Dataset):
    """
    This is our custom dataset. In this project, it will load data to train from csv files.
    If databases which have all of informations for training exist, it will load data to train from the database

    """
    def __init__(self, file_path = _FILE_PATH, file_name = _FILE_NAME, mode = "train"):
        super().__init__()
        self.file_path = file_path
        self.file_name = file_name
        self.mode = mode
        self.dataframe = pd.read_csv(os.path.join(self.file_path, "{}_{}.csv".format(self.file_name, self.mode)), header = 0, names = ["Time", "Glucose"])
        self.predict_length = 3
        self.sequence_length =  3
        self.dataset_length = len(self.dataframe) - (self.predict_length*_DAY + self.sequence_length*_DAY) + 1
        
    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        # Get data for training
        data = np.array(self.dataframe["Glucose"][idx : idx + (self.predict_length + self.sequence_length)*_DAY])

        # Split data to input and label
        input_data = torch.tensor(data[:-(self.predict_length*_DAY)], dtype = torch.long).clone()
        label = torch.tensor(data[-(self.predict_length*_DAY):], dtype = torch.float32).clone()
        return input_data, label


