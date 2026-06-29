"""
build_manifest.py
=================

Create train/validation/test manifests
from the processed dataset.

Each row corresponds to ONE VIDEO.

Columns
-------
video,label

Example
-------
real/000,0
fake/Deepfakes_001,1
"""

from pathlib import Path
import random
import pandas as pd

from configs.config import (
    PROCESSED_DATA_DIR,
    MANIFEST_DIR,
    SEED,
)

# =============================================================================
# SPLITS
# =============================================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def collect_videos():
    """
    Collect all processed videos.

    Returns
    -------
    list[(str, int)]
        (relative_path, label)
    """

    samples = []

    # -----------------------------
    # REAL
    # -----------------------------

    real_root = PROCESSED_DATA_DIR / "real"

    if real_root.exists():

        for video in sorted(real_root.iterdir()):

            if video.is_dir():

                samples.append((
                    str(video.relative_to(PROCESSED_DATA_DIR)),
                    0,
                ))

    # -----------------------------
    # FAKE
    # -----------------------------

    fake_root = PROCESSED_DATA_DIR / "fake"

    if fake_root.exists():

        for video in sorted(fake_root.iterdir()):

            if video.is_dir():

                samples.append((
                    str(video.relative_to(PROCESSED_DATA_DIR)),
                    1,
                ))

    return samples


def split_dataset(samples):
    """
    Shuffle and split.
    """

    random.seed(SEED)

    random.shuffle(samples)

    n = len(samples)

    train_end = int(TRAIN_RATIO * n)
    val_end = train_end + int(VAL_RATIO * n)

    train = samples[:train_end]
    val = samples[train_end:val_end]
    test = samples[val_end:]

    return train, val, test


def save_manifest(samples, path):
    """
    Save CSV manifest.
    """

    df = pd.DataFrame(
        samples,
        columns=[
            "video",
            "label",
        ],
    )

    df.to_csv(
        path,
        index=False,
    )

    print(f"Saved {len(df)} samples -> {path}")


def main():

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = collect_videos()

    print(f"Found {len(samples)} videos.")

    train, val, test = split_dataset(samples)

    save_manifest(
        train,
        MANIFEST_DIR / "train.csv",
    )

    save_manifest(
        val,
        MANIFEST_DIR / "val.csv",
    )

    save_manifest(
        test,
        MANIFEST_DIR / "test.csv",
    )

    print("\nManifest generation complete.")


if __name__ == "__main__":
    main()