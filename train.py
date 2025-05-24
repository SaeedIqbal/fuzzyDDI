# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.models.fuzzy_ddi import FuzzyDDI
from src.models.mc_dropout import MCDropout
from src.data.ddi_dataset import DDIDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from src.data.ddi_dataset import DDIDataset, load_drkg_embeddings
from src.utils.zero_shot_utils import build_similarity_matrix
# -------------------------------
# Configurations
# -------------------------------

CONFIG = {
    "data_path": "/home/phd/dataset/fuzzyddi/drugcombdb.csv",
    "batch_size": 32,
    "embedding_dim": 512,
    "hidden_dim": 1024,
    "num_epochs": 30,
    "learning_rate": 1e-3,
    "use_transformer": True,
    "use_mc_dropout": True,
    "num_samples": 10,  # For MC Dropout uncertainty
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
'''
# Configuration
CONFIG = {
    "data_path": "/home/phd/dataset/fuzzyddi/drugcombdb.csv",
    "drkg_embedding_path": "/home/phd/dataset/fuzzyddi/drkg_embeddings.csv",
    "batch_size": 32,
    "embedding_dim": 512,
    "hidden_dim": 1024,
    "use_transformer": True,
    "use_mc_dropout": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
'''


# -------------------------------
# Load Dataset
# -------------------------------

def load_dataset(data_path):
    df = pd.read_csv(data_path)
    drug_list = set()
    for drugs in df['drug_sequence']:
        drug_list.update(drug_list.union(set(drugs.split(";"))))
    drug_map = {d: i+1 for i, d in enumerate(drug_list)}  # 1-based indexing
    dataset = DDIDataset(df, drug_map)
    return dataset, drug_map


# Load DRKG embeddings
drkg_embeddings, entity_to_idx = load_drkg_embeddings(CONFIG["drkg_embedding_path"])

# Modify dataset to use DRKG mapping
dataset = DDIDataset(df, entity_to_idx)  # Previously used drug_map — now use DRKG's entity_to_idx
loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
# -------------------------------
# Attention Visualization
# -------------------------------

def visualize_attention(attn_weights, title="Transformer Attention"):
    """Visualize attention weights from Transformer layer"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_weights[0].cpu().numpy(), cmap='viridis', annot=True)
    plt.title(title)
    plt.xlabel("Head")
    plt.ylabel("Token")
    plt.show()

-----------------
# MC Dropout Prediction
# -------------------------------

def predict_with_uncertainty(model, loader, device, num_samples=10):
    model.train()  # Keep in train mode to enable dropout
    probs = []
    
    for batch in loader:
        drug_indices = batch['drug_indices'].to(device)
        doses = batch['doses'].to(device)
        with torch.no_grad():
            prob, _, _ = model(drug_indices, doses=doses)
            probs.append(prob.cpu().numpy())

    probs = np.concatenate(probs, axis=0)
    mean_prob = np.mean(probs, axis=0)
    std_prob = np.std(probs, axis=0)
    return mean_prob, std_prob

# -------------------------------
# Train Function
# -------------------------------

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []
    # train.py

    mechanism_criterion = nn.CrossEntropyLoss()

    # In training loop
    
    for batch in loader:
        drug_indices = batch['drug_indices'].to(device)
        doses = batch['doses'].to(device)
        labels = batch['label'].to(device)
        
        mechanism_logits = model.mechanism_head(out)  # Assume you added this head
        mechanism_labels = batch['mechanism_label'].to(device)
        loss += 0.5 * mechanism_criterion(mechanism_logits, mechanism_labels)
        
        optimizer.zero_grad()
        prob, logits, _ = model(drug_indices, doses=doses)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(prob.detach().cpu().numpy())

    auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)
    return total_loss / len(loader), auc, aupr

# -------------------------------
# Evaluate Function
# -------------------------------

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            drug_indices = batch['drug_indices'].to(device)
            doses = batch['doses'].to(device)
            labels = batch['label'].to(device)

            prob, _, _ = model(drug_indices, doses=doses)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(prob.cpu().numpy())

    auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)
    return auc, aupr

# -------------------------------
# Main Function
# -------------------------------

if __name__ == "__main__":
    # Set up config
    data_path = CONFIG["data_path"]
    batch_size = CONFIG["batch_size"]
    embedding_dim = CONFIG["embedding_dim"]
    hidden_dim = CONFIG["hidden_dim"]
    use_transformer = CONFIG["use_transformer"]
    use_mc_dropout = CONFIG["use_mc_dropout"]
    num_epochs = CONFIG["num_epochs"]
    lr = CONFIG["learning_rate"]
    device = torch.device(CONFIG["device"])

    print(f"Using device: {device}")

    # Load dataset
    dataset, drug_map = load_dataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = FuzzyDDI(
        num_entities=len(drug_map) + 1,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        use_transformer=use_transformer,
        use_mc_dropout=use_mc_dropout
    ).to(device)
    '''
    model = FuzzyDDI.from_pretrained_embeddings(
    drkg_embeddings,
    embedding_dim=CONFIG["embedding_dim"],
    hidden_dim=CONFIG["hidden_dim"],
    use_transformer=CONFIG["use_transformer"],
    use_mc_dropout=CONFIG["use_mc_dropout"]
).to(CONFIG["device"])
    '''
    # Freeze DRKG embeddings
    #model.entity_embeddings.weight.requires_grad = False
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_auc, train_aupr = train(model, loader, optimizer, criterion, device)
        val_auc, val_aupr = evaluate(model, loader, device)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_model.pth")

    # Predict with uncertainty
    mean_probs, std_probs = predict_with_uncertainty(model, loader, device, num_samples=20)
    print("\nUncertainty Estimation:")
    print("Mean Probabilities:", mean_probs[:5])
    print("Uncertainty (std):", std_probs[:5])

    # Visualize attention (if using Transformer)
    sample_batch = next(iter(loader))
    model.eval()
    '''
    sample_batch = next(iter(loader))
    drug_indices = sample_batch['drug_indices'].to(device)
    doses = sample_batch['doses'].to(device)

    prob, logits, _ = model(drug_indices, doses=doses)
    print("Predicted Probability:", prob.item())
    '''
    with torch.no_grad():
        _, _, attn_weights = model(sample_batch['drug_indices'].to(device), sample_batch['doses'].to(device))
        if attn_weights is not None:
            visualize_attention(attn_weights, title="Self-Attention Weights")