import torch

from models.attention import TemporalResidualAttention

model = TemporalResidualAttention(dim=728)

x = torch.randn(
    2,
    6 * (19 * 19 + 1),
    728,
)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)

assert x.shape == y.shape

print("Temporal Residual Attention test passed!")

x = torch.randn(
    2,
    6 * (19 * 19 + 1),
    728,
    requires_grad=True,
)

y = model(x)

loss = y.mean()

loss.backward()

assert x.grad is not None

print("Backward pass ✓")