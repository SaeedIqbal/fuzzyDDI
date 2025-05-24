# src/data/ddi_dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset

class DDIDataset(Dataset):
    def __init__(self, df, drug_map, max_seq_len=2, use_neg_sampling=True):
        self.df = df
        self.drug_map = drug_map
        self.max_seq_len = max_seq_len
        self.use_neg_sampling = use_neg_sampling
        self.entity_types = self._load_entity_types()

    def _load_entity_types(self):
        """Mocked function — load from DRKG or external source"""
        return {
            'DB00402': 'opiates',
            'DB01175': 'anticoagulant',
            'DB00855': 'antihistamine',
            # ... more types
        }
    def generate_negative_samples(self, drug1, drug2, k=5):
        if not self.use_neg_sampling:
            return []

        drug1_type = self.entity_types.get(drug1, None)
        drug2_type = self.entity_types.get(drug2, None)

        if drug1_type != drug2_type:
            return []

        negatives = []
        while len(negatives) < k:
            neg_drug = random.choice(list(self.drug_map.keys()))
            if neg_drug == drug1 or neg_drug == drug2:
                continue
            if self.entity_types.get(neg_drug, None) == drug1_type:
                negatives.append(neg_drug)

        return negatives

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