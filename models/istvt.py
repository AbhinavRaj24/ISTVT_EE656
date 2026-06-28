"""
istvt.py
========

Complete implementation of the

Interpretable Spatial-Temporal Video Transformer (ISTVT)
for Deepfake Detection.

Pipeline:

Input Frames
        ↓
Xception Entry Flow
        ↓
Patch Tokenization
        ↓
CLS Token + Positional Embedding
        ↓
Transformer Encoder
        ↓
Classification Head
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat

from configs.config import (
    IMAGE_SIZE,
    FEATURE_GRID,
    EMBED_DIM,
    SEQ_LEN,
    PATCH_SIZE,
)

from models.backbone import XceptionBackbone
from models.transformer import TransformerEncoder


class ISTVT(nn.Module):
    """
    Interpretable Spatial-Temporal Video Transformer.
    """

    def __init__(self, pretrained_backbone=True):

        super().__init__()

        # --------------------------------------------------
        # Backbone
        # --------------------------------------------------

        self.backbone = XceptionBackbone(
            pretrained=pretrained_backbone
        )
        # --------------------------------------------------
        # Tokens
        # --------------------------------------------------

        self.tokens_per_frame = FEATURE_GRID * FEATURE_GRID

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, EMBED_DIM)
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                SEQ_LEN * (self.tokens_per_frame + 1),
                EMBED_DIM,
            )
        )

        # --------------------------------------------------
        # Transformer
        # --------------------------------------------------

        self.transformer = TransformerEncoder()

        # --------------------------------------------------
        # Classification Head
        # --------------------------------------------------

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, 2),
        )

    def forward(self, x):
        """
        Input:
            x : (B,T,3,300,300)

        Output:
            logits : (B,2)
        """

        B, T, C, H, W = x.shape

        assert T == SEQ_LEN

        # --------------------------------------------------
        # Backbone
        # --------------------------------------------------

        x = rearrange(
            x,
            "b t c h w -> (b t) c h w",
        )

        x = self.backbone(x)

        # (B*T,728,19,19)

        # --------------------------------------------------
        # Patch tokens
        # --------------------------------------------------

        x = rearrange(
            x,
            "(b t) c h w -> b t (h w) c",
            b=B,
            t=T,
        )

        # (B,T,361,728)

        # --------------------------------------------------
        # CLS token
        # --------------------------------------------------

        cls = repeat(
            self.cls_token,
            "1 1 d -> b t 1 d",
            b=B,
            t=T,
        )

        x = torch.cat(
            (cls, x),
            dim=2,
        )

        # (B,T,362,728)

        # --------------------------------------------------
        # Flatten temporal dimension
        # --------------------------------------------------

        x = rearrange(
            x,
            "b t n d -> b (t n) d",
        )

        # (B,2172,728)

        # --------------------------------------------------
        # Positional embedding
        # --------------------------------------------------

        x = x + self.pos_embedding

        # --------------------------------------------------
        # Transformer
        # --------------------------------------------------

        x = self.transformer(x)

        # --------------------------------------------------
        # Frame CLS tokens
        # --------------------------------------------------

        x = rearrange(
            x,
            "b (t n) d -> b t n d",
            t=SEQ_LEN,
            n=self.tokens_per_frame + 1,
        )

        cls_tokens = x[:, :, 0]

        # (B,T,728)

        cls_tokens = cls_tokens.mean(dim=1)

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        logits = self.mlp_head(cls_tokens)

        return logits