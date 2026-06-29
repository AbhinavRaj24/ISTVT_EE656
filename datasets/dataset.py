"""
dataset.py
==========

PyTorch Dataset for ISTVT.

Loads sequences of six consecutive face images.
"""

import pandas as pd
from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from configs.config import (
    PROCESSED_DATA_DIR,
    IMAGE_SIZE,
    SEQ_LEN,
)
class ISTVTDataset(Dataset):

    def __init__(
        self,
        manifest,
        root=PROCESSED_DATA_DIR,
        transform=None,
    ):
        self.root = Path(root)
        self.manifest = Path(manifest)
        self.samples = []

        self._scan_videos()
        self.transform = transform

        if self.transform is None:

            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])


    def _scan_videos(self):
        """
        Read the manifest and build all valid sequences.
        """

        if not self.manifest.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest}"
            )

        df = pd.read_csv(self.manifest)

        for _, row in df.iterrows():

            video_dir = self.root / row["video"]

            label = int(row["label"])

            if not video_dir.exists():
                continue

            self._build_sequences(
                video_dir,
                label,
            )

    def _build_sequences(self, video_dir, label):
        """
        Build sliding-window sequences of length SEQ_LEN.
        """

        images = sorted(video_dir.glob("*.jpg"))

        if len(images) < SEQ_LEN:
            return

        for start in range(len(images) - SEQ_LEN + 1):

            sequence = images[start:start + SEQ_LEN]

            self.samples.append(
                (
                    sequence,
                    label,
                )
            )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        image_paths, label = self.samples[index]

        frames = []

        for image_path in image_paths:

            image = Image.open(image_path).convert("RGB")

            image = self.transform(image)

            frames.append(image)

        frames = torch.stack(frames)

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        return frames, label
    