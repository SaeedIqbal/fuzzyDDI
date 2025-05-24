# src/models/fuzzy_ddi.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .temporal_encoder import TemporalEncoder
from .mc_dropout import MCDropout

def fuzzy_conjunction(set1, set2):
    return torch.min(set1, set2)

def fuzzy_disjunction(set1, set2):
    return torch.max(set1, set2)

class FuzzyDDI(nn.Module):
    """
    Fuzzy-DDI Model:
    - Uses GCN and MLP to learn relational projections
    - Applies fuzzy logic operations (conjunctive/disjunctive)
    - Supports temporal modeling and MC Dropout for uncertainty
    """

    def __init__(self, num_entities, embedding_dim=512, hidden_dim=1024,
                 use_transformer=True, use_mc_dropout=True):
        super(FuzzyDDI, self).__init__()

        # Entity Embedding Layer
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)

        # GCN-based Projection Module
        self.gcn1 = nn.Linear(embedding_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, embedding_dim)

        # MLP Module
        self.mlp = nn.Sequential(
            MCDropout(p=0.3) if use_mc_dropout else nn.Dropout(0.3),
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            MCDropout(p=0.3) if use_mc_dropout else nn.Dropout(0.3),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # Final Classifier
        self.classifier = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # Temporal Encoder (optional)
        self.temporal_encoder = TemporalEncoder(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            use_transformer=use_transformer
        )

    def forward(self, drug_indices, doses=None):
        """
        Forward pass for predicting DDIs using fuzzy logic and relational projections.

        Args:
            drug_indices: Tensor of shape (batch_size, seq_len)
            doses:        Tensor of shape (batch_size, seq_len, 1), optional

        Returns:
            prob:         Predicted probability of interaction
            logits:       Raw logits before sigmoid
            embeddings:   Final fused embeddings used for prediction
        """
        # Step 1: Get base embeddings
        d_embs = self.entity_embeddings(drug_indices)

        # Step 2: Apply GCN-style projection
        d_embs = F.relu(self.gcn1(d_embs))
        d_embs = self.gcn2(d_embs)

        # Step 3: Temporal Encoding
        if d_embs.shape[1] > 1:  # If there's a sequence
            d_embs, _ = self.temporal_encoder(d_embs, doses=doses)
        else:
            d_embs = d_embs.squeeze(1)

        # Step 4: Fuzzy Logic Operations
        # Assume we're predicting interactions between two drugs
        d1_emb = d_embs[:, :1] if d_embs.dim() == 3 else d_embs[:1]
        d2_emb = d_embs[:, 1:] if d_embs.dim() == 3 else d_embs[1:]

        conjunctive_emb = fuzzy_conjunction(d1_emb, d2_emb)
        disjunctive_emb = fuzzy_disjunction(d1_emb, d2_emb)

        combined = torch.cat([conjunctive_emb, disjunctive_emb], dim=-1)

        # Step 5: MLP + Classification
        out = self.mlp(combined)
        logits = self.classifier(out).squeeze(-1)
        prob = self.sigmoid(logits)

        return prob, logits, out