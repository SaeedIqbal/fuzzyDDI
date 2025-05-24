# utils/train_utils.py

import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score

def train(model, loader, device, optimizer=None, criterion=None):
    model.train()
    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=1e-3)
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    total_loss = 0
    y_true, y_pred = [], []

    for batch in loader:
        drug_indices = batch['drug_indices'].to(device)
        doses = batch['doses'].to(device)
        labels = batch['label'].to(device)

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