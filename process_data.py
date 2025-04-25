import os
import numpy as np
import pandas as pd
import torch
import argparse
from utils import Params
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def spilt(params):
    df = pd.read_csv(os.path.join(params.data_dir, 'origin', 'train.csv'))
    data_train, data_test = train_test_split(df, test_size=0.2, random_state=42)
    data_train.to_csv(os.path.join(params.data_dir, 'processed', 'train.csv'), index=False)
    data_test.to_csv(os.path.join(params.data_dir, 'processed', 'test.csv'), index=False)

def collate_fn(batch):
    features = [p[0] for p in batch]
    lables = [p[1] for p in batch]
    features = torch.tensor(features, dtype=torch.float32)
    lables = torch.tensor(lables, dtype=torch.float32).view(-1, 1)
    return features, lables
    

class TitanicDataset(Dataset):
    def __init__(self, params, data_path):
        super(TitanicDataset, self).__init__()
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
        self.id = self.data['PassengerId']
        self.id = torch.tensor(self.id.to_numpy().astype(np.float32), dtype=torch.int32).view(-1, 1)
        self.features = self.data[params.columns]
        self.features.loc[:,'Sex'] = pd.factorize(self.features['Sex'])[0]
        self.features = torch.tensor(self.features.to_numpy().astype(np.float32), dtype=torch.float32)
        if 'Survived' in self.data.columns:
            self.label = self.data['Survived']
            self.label = torch.tensor(self.label.to_numpy().astype(np.float32), dtype=torch.float32).view(-1, 1)
        else:
            self.label = torch.full((self.features.size(0), 1), -1)

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, index):
        return self.id[index], self.features[index], self.label[index]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_path', default='model/params.json', help="params json path")
    args = parser.parse_args()

    params = Params(args.params_path)
    spilt(params)