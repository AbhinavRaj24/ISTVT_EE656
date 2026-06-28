import torch

from models.attention import AttentionBase

model = AttentionBase(dim=728)

x = torch.randn(2, 10, 728)

q, k, v = model.project_qkv(x)

print("Q:", q.shape)
print("K:", k.shape)
print("V:", v.shape)

out = model.scaled_dot_product(q, k, v)

print("Attention output:", out.shape)

out = model.output_projection(out)

print("Final output:", out.shape)