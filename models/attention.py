"""
attention.py
============

Attention modules used in ISTVT.

This file contains:

    1. AttentionBase (generic multi-head attention utilities)
    2. SpatialOnlyAttention
    3. TemporalResidualAttention

The attention mechanisms are adapted from the ISTVT paper while keeping the
implementation modular and easy to understand.
"""

import torch
import torch.nn as nn
from einops import rearrange
from configs.config import NUM_HEADS, DIM_HEAD, DROPOUT
from configs.config import FEATURE_GRID, SEQ_LEN


class AttentionBase(nn.Module):
    """
    Base class for ISTVT attention modules.

    Stores the common attention hyperparameters
    (number of heads, head dimension, scaling factor)
    and the output projection layer.

    Each subclass is responsible for implementing its own
    query/key/value projections and attention mechanism.
    """

    def __init__(
        self,
        dim: int,
        heads: int = NUM_HEADS,
        dim_head: int = DIM_HEAD,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head

        # Project input -> Query, Key, Value
        self.to_qkv = nn.Linear(
            dim,
            inner_dim * 3,
            bias=False,
        )

        # Project attention output back to embedding dimension
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def output_projection(self, x):
        """
        Project concatenated heads back to embedding dimension.
        """

        return self.to_out(x)
    

class SpatialOnlyAttention(AttentionBase):
    """
    Spatial Self-Attention.

    Attention is performed independently inside each frame.

    Input:
        (B, T*N, C)

    Output:
        (B, T*N, C)
    """

    def forward(self, x):

        tokens_per_frame = FEATURE_GRID * FEATURE_GRID + 1

        assert x.shape[1] == SEQ_LEN * tokens_per_frame, (
            f"Expected {SEQ_LEN * tokens_per_frame} tokens, got {x.shape[1]}"
        )
        # --------------------------------------------------
        # Linear projection
        # --------------------------------------------------

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (t n) (h d) -> b h t n d",
                t=SEQ_LEN,
                n=tokens_per_frame,
                h=self.heads,
            ),
            qkv,
        )

        # --------------------------------------------------
        # Spatial attention
        # --------------------------------------------------

        scores = torch.matmul(
            q,
            k.transpose(-1, -2),
        ) * self.scale

        attention = scores.softmax(dim=-1)

        out = torch.matmul(attention, v)

        # --------------------------------------------------
        # Merge heads
        # --------------------------------------------------

        out = rearrange(
            out,
            "b h t n d -> b (t n) (h d)",
        )

        return self.output_projection(out)


class TemporalResidualAttention(AttentionBase):
    """
    Temporal Residual Self-Attention.

    Implements the Self-Subtract mechanism proposed in the ISTVT paper.

    Input:
        (B, T*N, C)

    Output:
        (B, T*N, C)
    """

    def __init__(
        self,
        dim: int,
        heads: int = NUM_HEADS,
        dim_head: int = DIM_HEAD,
        dropout: float = DROPOUT,
    ):
        super().__init__(dim, heads, dim_head, dropout)

        inner_dim = heads * dim_head

        # Temporal attention uses separate QK and V projections
        self.to_qk = nn.Linear(
            dim,
            inner_dim * 2,
            bias=False,
        )

        self.to_v = nn.Linear(
            dim,
            inner_dim,
            bias=False,
        )
    


    def forward(self, x):

        tokens_per_frame = FEATURE_GRID * FEATURE_GRID + 1

        assert x.shape[1] == SEQ_LEN * tokens_per_frame, (
            f"Expected {SEQ_LEN * tokens_per_frame} tokens, got {x.shape[1]}"
        )
        # --------------------------------------------------
        # Compute temporal residual
        # --------------------------------------------------

        x = rearrange(
            x,
            "b (t n) c -> b t n c",
            t=SEQ_LEN,
            n=tokens_per_frame,
        )

        residual = torch.cat(
            (
                x[:, :2],
                x[:, 2:] - x[:, 1:-1],
            ),
            dim=1,
        )

        residual = rearrange(
            residual,
            "b t n c -> b (t n) c",
        )

        x = rearrange(
            x,
            "b t n c -> b (t n) c",
        )

        # --------------------------------------------------
        # Q,K from residual
        # --------------------------------------------------

        qk = self.to_qk(residual).chunk(2, dim=-1)

        q, k = map(
            lambda t: rearrange(
                t,
                "b (t n) (h d) -> b h n t d",
                t=SEQ_LEN,
                n=tokens_per_frame,
                h=self.heads,
            ),
            qk,
        )

        # --------------------------------------------------
        # V from original features
        # --------------------------------------------------

        v = self.to_v(x)

        v = rearrange(
            v,
            "b (t n) (h d) -> b h n t d",
            t=SEQ_LEN,
            n=tokens_per_frame,
            h=self.heads,
        )

        # --------------------------------------------------
        # Temporal attention
        # --------------------------------------------------

        scores = torch.matmul(
            q,
            k.transpose(-1, -2),
        ) * self.scale

        attention = scores.softmax(dim=-1)

        out = torch.matmul(attention, v)

        # --------------------------------------------------
        # Merge heads
        # --------------------------------------------------

        out = rearrange(
            out,
            "b h n t d -> b (t n) (h d)",
        )

        return self.output_projection(out)