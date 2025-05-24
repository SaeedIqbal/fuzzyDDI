import torch
import torch.nn.functional as F
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score
)

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
    y_pred_bin = (np.array(y_pred) > 0.5).astype(int)
    kappa = cohen_kappa_score(y_true, y_pred_bin)
    return auc, aupr, kappa