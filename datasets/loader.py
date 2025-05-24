import torch
import torch.nn.functional as F
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets.ddi_dataset import DDIDataset
from data_loader import load_drkg_embeddings

def build_dataloader(data_path, batch_size=32, shuffle=True):
    df = pd.read_csv(data_path)
    drug_list = set()
    for drugs in df['drug_sequence']:
        drug_list.update(set(drugs.split(";")))
    drug_map = {d: i+1 for i, d in enumerate(drug_list)}

    dataset = DDIDataset(df, drug_map)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), drug_map

def load_drkg_embeddings(drkg_path):
    """
    Loads pre-trained DRKG embeddings.
    Returns:
        - A tensor of shape (num_entities, emb_dim)
        - A dictionary mapping DrugBank ID -> index in embedding matrix
    """
    df = pd.read_csv(drkg_path)
    embedding_matrix = []
    entity_to_idx = {}

    for idx, row in df.iterrows():
        entity_id = row['entity']
        if entity_id.startswith("DrugBank::"):
            drug_id = entity_id.split("::")[1]
            embedding_matrix.append(row.values[1:].astype(np.float32))
            entity_to_idx[drug_id] = len(entity_to_idx)

    embedding_tensor = torch.tensor(np.array(embedding_matrix), dtype=torch.float32)
    print(f"Loaded DRKG embeddings for {len(entity_to_idx)} DrugBank entities")
    return embedding_tensor, entity_to_idx