"""
transformer.py
==============

Transformer encoder used in the Interpretable Spatial-Temporal Video
Transformer (ISTVT).

Each TransformerBlock consists of:

    Spatial Self-Attention
        ↓
    Temporal Residual Self-Attention
        ↓
    Feed Forward Network

Each sub-layer is wrapped with:

    Residual
        +
    PreNorm

The complete encoder stacks multiple TransformerBlocks.
"""

import torch.nn as nn

from configs.config import (
    EMBED_DIM,
    DEPTH,
    MLP_DIM,
    DROPOUT,
)

from models.layers import (
    Residual,
    PreNorm,
    FeedForward,
)

from models.attention import (
    SpatialOnlyAttention,
    TemporalResidualAttention,
)


class TransformerBlock(nn.Module):
    """
    One ISTVT Transformer Block.
    """

    def __init__(
        self,
        dim=EMBED_DIM,
        mlp_dim=MLP_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.spatial = Residual(
            PreNorm(
                dim,
                SpatialOnlyAttention(
                    dim=dim,
                    dropout=dropout,
                ),
            )
        )

        self.temporal = Residual(
            PreNorm(
                dim,
                TemporalResidualAttention(
                    dim=dim,
                    dropout=dropout,
                ),
            )
        )

        self.feedforward = Residual(
            PreNorm(
                dim,
                FeedForward(
                    dim=dim,
                    hidden_dim=mlp_dim,
                    dropout=dropout,
                ),
            )
        )

    def forward(self, x):

        x = self.spatial(x)

        x = self.temporal(x)

        x = self.feedforward(x)

        return x


class TransformerEncoder(nn.Module):
    """
    Stack of TransformerBlocks.
    """

    def __init__(
        self,
        depth=DEPTH,
        dim=EMBED_DIM,
        mlp_dim=MLP_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x