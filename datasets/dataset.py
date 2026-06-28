"""
dataset.py
==========

PyTorch Dataset for ISTVT.

Loads sequences of six consecutive face images.
"""

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
        root=PROCESSED_DATA_DIR,
        transform=None,
    ):
        self.root = Path(root)

        self.transform = transform

        if self.transform is None:

            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        self.samples = []

        self._scan_videos()

    def _scan_videos(self):
        """
        Scan the processed dataset directory and build all valid
        six-frame sequences.

        Expected directory structure:

        processed/
            real/
                video_1/
                    000000.jpg
                    ...
            fake/
                video_2/
                    000000.jpg
                    ...
        """

        if not self.root.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {self.root}"
            )

        classes = {
            "real": 0,
            "fake": 1,
        }

        for class_name, label in classes.items():

            class_dir = self.root / class_name

            if not class_dir.exists():
                continue

            for video_dir in sorted(class_dir.iterdir()):

                if not video_dir.is_dir():
                    continue

                self._build_sequences(video_dir, label)


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
    