"""
layers.py
=========

Reusable neural network layers used by ISTVT.

These are generic transformer components used throughout the transformer.
"""

import torch
import torch.nn as nn


class Residual(nn.Module):
    """
    Residual Skip Connection.

    Computes:
        y = x + f(x)
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class PreNorm(nn.Module):
    """
    Applies LayerNorm before the given module.
    """

    def __init__(self, dim, fn):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    Standard Transformer Feed Forward Network.

    Linear
      ↓
    GELU
      ↓
    Dropout
      ↓
    Linear
      ↓
    Dropout
    """

    def __init__(
        self,
        dim,
        hidden_dim,
        dropout=0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)