# src/models/temporal_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalEncoder(nn.Module):
    """
    Temporal Encoder with optional GRU or Transformer layers.
    Encodes sequences of drug administrations over time.
    """

    def __init__(self, input_dim, hidden_dim, use_transformer=True, num_heads=4):
        super(TemporalEncoder, self).__init__()
        self.use_transformer = use_transformer
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)

        self.dose_proj = nn.Linear(1, input_dim)  # Project dose into embedding space
        self.final_proj = nn.Linear(hidden_dim if not use_transformer else input_dim, input_dim)

    def forward(self, embeddings, doses=None):
        """
        Args:
            embeddings: Drug embeddings (batch_size, seq_len, emb_dim)
            doses:      Optional drug doses (batch_size, seq_len, 1)

        Returns:
            encoded_embeddings: Final representation after temporal encoding
            attn_weights:       Attention weights (only available if using Transformer)
        """
        if doses is not None:
            dose_emb = self.dose_proj(doses)
            embeddings = embeddings + dose_emb

        if self.use_transformer:
            encoded = self.encoder(embeddings)
            attn_weights = None  # For now, no explicit attention output
        else:
            encoded, _ = self.encoder(embeddings)
            attn_weights = None

        # Use mean pooling over sequence dimension
        encoded_embeddings = encoded.mean(dim=1)
        return encoded_embeddings, attn_weights
    
'''


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_transformer=True, num_heads=4):
        super(TemporalEncoder, self).__init__()
        self.use_transformer = use_transformer
        if use_transformer:
            self.attn_weights = None
            self.attn_layer = nn.MultiheadAttention(input_dim, num_heads=num_heads)
        else:
            self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, embeddings, doses=None):
        if doses is not None:
            embeddings += self.dose_proj(doses)

        if self.use_transformer:
            embeddings = embeddings.transpose(0, 1)  # (T, B, E)
            out, self.attn_weights = self.attn_layer(
                embeddings, embeddings, embeddings
            )
            out = out.transpose(0, 1)
            encoded = out.mean(dim=1)
        else:
            encoded, _ = self.encoder(embeddings)
            encoded = encoded.mean(dim=1)
            self.attn_weights = None

        return encoded, self.attn_weights
'''