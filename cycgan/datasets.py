from torch.utils import data
import pandas as pd
import torch
import random


class DemographicDataset(data.Dataset):
    def __init__(self, data_A_path, data_B_path):
        data_A = pd.read_csv(f'{data_A_path}', index_col=None).values
        data_B = pd.read_csv(f'{data_B_path}', index_col=None).values
        self.data_A = torch.from_numpy(data_A).to(torch.float)
        self.data_B = torch.from_numpy(data_B).to(torch.float)

    def __getitem__(self, index):
        item_A = self.data_A[index % self.data_A.shape[0]]
        item_B = self.data_B[random.randint(0, self.data_B.shape[0]-1)]
        return item_A, item_B

    def __len__(self):
        return max(self.data_A.shape[0], self.data_B.shape[0])


def train_loder(data_A_path, data_B_path, batch_size, num_workers):
    dataset = DemographicDataset(data_A_path, data_B_path)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

