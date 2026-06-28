import torch

from models.backbone import XceptionBackbone

model = XceptionBackbone(pretrained=False)

x = torch.randn(2, 3, 300, 300)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)

assert y.shape == (2, 728, 19, 19)

print("Backbone test passed!")