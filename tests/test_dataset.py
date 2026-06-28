import tempfile
from pathlib import Path

import torch
from PIL import Image

from datasets.dataset import ISTVTDataset


# Create a temporary fake dataset
with tempfile.TemporaryDirectory() as tmp:

    root = Path(tmp)

    video = root / "real" / "video1"
    video.mkdir(parents=True)

    # Create 8 dummy frames
    for i in range(8):
        img = Image.new("RGB", (300, 300), color=(i * 20, 0, 0))
        img.save(video / f"{i:06d}.jpg")

    dataset = ISTVTDataset(root=root)

    print("Number of sequences:", len(dataset))

    x, y = dataset[0]

    print("Frames:", x.shape)
    print("Label :", y)

    assert len(dataset) == 3          # 8 frames -> 3 sliding windows
    assert x.shape == (6, 3, 300, 300)
    assert y.item() == 0

print("Dataset test passed!")