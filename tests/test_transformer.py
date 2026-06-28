import torch

from models.transformer import TransformerEncoder

model = TransformerEncoder(depth=2)

x = torch.randn(
    2,
    6 * (19 * 19 + 1),
    728,
)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)

assert x.shape == y.shape

print("Transformer test passed!")

# Backward pass
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