# src/data/ddi_dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset

class DDIDataset(Dataset):
    def __init__(self, df, drug_map, max_seq_len=2):
        self.df = df
        self.drug_map = drug_map
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        drugs = row['drug_sequence'].split(";")[:self.max_seq_len]
        doses = [float(d) for d in row['doses'].split(";")[:self.max_seq_len]]
        label = float(row['label'])

        drug_indices = [self.drug_map.get(d, 0) for d in drugs]  # 0 for unknown
        padded_drugs = drug_indices + [0] * (self.max_seq_len - len(drug_indices))
        padded_doses = doses + [0.0] * (self.max_seq_len - len(doses))

        return {
            'drug_indices': torch.tensor(padded_drugs),
            'doses': torch.tensor(padded_doses).unsqueeze(-1),
            'label': torch.tensor(label, dtype=torch.float32)
        }