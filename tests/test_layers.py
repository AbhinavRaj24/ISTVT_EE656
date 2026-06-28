import torch
import torch.nn as nn

from models.layers import FeedForward, Residual

# FeedForward test
ff = FeedForward(
    dim=728,
    hidden_dim=728 * 4,
)

x = torch.randn(2, 100, 728)

y = ff(x)

assert y.shape == x.shape

print("FeedForward ✓")


# Residual test
identity = Residual(nn.Identity())

z = identity(x)

assert z.shape == x.shape

print("Residual ✓")