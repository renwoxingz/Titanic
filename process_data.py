import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class TitanicDataset(Dataset):
    def __init__(self, params, data_path):
        super(TitanicDataset, self).__init__()
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
        self.features = self.data[params.columns]
        self.features['Sex'] = pd.factorize(self.features['Sex'])[0]
        self.features = torch.tensor(self.features.to_numpy(), dtype=torch.float32)
        if 'Survived' in self.data.columns:
            self.label = self.data['Survived']
            self.label = torch.tensor(self.label.to_numpy(), dtype=torch.float32)
        else:
            self.label = torch.full((self.data.size(0), 1), -1)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.features[index], self.label[index]