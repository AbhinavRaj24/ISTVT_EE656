import torch

from models.istvt import ISTVT

model = ISTVT(pretrained_backbone=False)

x = torch.randn(
    2,
    6,
    3,
    300,
    300,
)

with torch.no_grad():
    y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)

assert y.shape == (2, 2)

print("ISTVT model test passed!")