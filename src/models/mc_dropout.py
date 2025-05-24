# src/models/mc_dropout.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MCDropout(nn.Module):
    """
    Monte Carlo Dropout module for uncertainty estimation.
    This dropout layer remains active during inference to allow multiple stochastic forward passes.
    """
    def __init__(self, p=0.5):
        super(MCDropout, self).__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)

    def extra_repr(self):
        return f'dropout_rate={self.p}'