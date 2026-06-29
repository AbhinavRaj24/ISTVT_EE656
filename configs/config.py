"""
config.py
=========

Global configuration for the implementation of

ISTVT: Interpretable Spatial-Temporal Video Transformer
for Deepfake Detection
(IEEE TIFS, 2023)

Paper-defined parameters and implementation-specific
parameters are intentionally separated for clarity.
"""

from pathlib import Path
import torch

# =============================================================================
# PROJECT DIRECTORIES
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

# RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_DIR = Path("/content/drive/MyDrive/ISTVT/data/raw")
# PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR = Path("/content/drive/MyDrive/ISTVT/data/processed")
# PROCESSED_DATA_DIR = Path("/content/processed")
# MANIFEST_DIR = DATA_DIR / "manifests"
MANIFEST_DIR = Path("/content/drive/MyDrive/ISTVT/data/manifests")

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# =============================================================================
# RAW DATASET PATHS
# Update these according to your system.
# =============================================================================

# RAW_PATHS = {
#     "ff": "/path/to/FaceForensics++",
#     "celebdf": "/path/to/Celeb-DF-v2",
#     "dfdc": "/path/to/DFDC"
# }

RAW_PATHS = {
    "ff": "/content/drive/MyDrive/ISTVT/data/raw/FaceForensics++_C23",
}

# =============================================================================
# PARAMETERS EXPLICITLY SPECIFIED IN THE PAPER
# =============================================================================

# Input image size
IMAGE_SIZE = 300

# Face sequence length
SEQ_LEN = 6

# Patch size (feature map is split into 1×1 patches)
PATCH_SIZE = 1

# Xception Entry Flow output
FEATURE_GRID = 19          # 19 × 19 feature map
EMBED_DIM = 728            # Feature dimension

# Transformer
DEPTH = 12                 # Number of ST blocks (M)
NUM_HEADS = 8              # Number of attention heads (N)

# Paper:
# D = C / N
DIM_HEAD = EMBED_DIM // NUM_HEADS   # 728 / 8 = 91
# DIM_HEAD = 64
# Face preprocessing
FACE_CROP_SCALE = 1.25

# Training
LEARNING_RATE = 5e-4
EPOCHS = 100

# =============================================================================
# IMPLEMENTATION CHOICES
# (Not explicitly specified in the paper)
# =============================================================================

# Feed Forward Network expansion ratio
MLP_SCALE = 4
MLP_DIM = EMBED_DIM * MLP_SCALE

# Transformer dropout
DROPOUT = 0.1

# SGD weight decay
WEIGHT_DECAY = 1e-4

# Warm-up learning strategy (paper mentions warm-up but not exact settings)
USE_WARMUP = True
WARMUP_EPOCHS = 5

# =============================================================================
# TRAINING
# =============================================================================

BATCH_SIZE = 4

# Simulates batch size = 16
GRAD_ACCUM_STEPS = 4

NUM_WORKERS = 2

PIN_MEMORY = True

# =============================================================================
# DATASET SETTINGS
# =============================================================================

# Intra-dataset experiments
TRAIN_DATASET = "ff"

# Cross-dataset evaluation
CROSS_TEST_DATASETS = [
    "celebdf",
    "dfdc"
]


# =============================================================================
# MANIFEST FILES
# =============================================================================

TRAIN_MANIFEST = MANIFEST_DIR / "train.csv"
VAL_MANIFEST = MANIFEST_DIR / "val.csv"
TEST_MANIFEST = MANIFEST_DIR / "test.csv"

# =============================================================================
# HARDWARE
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# RANDOMNESS
# =============================================================================

SEED = 42

# =============================================================================
# CHECKPOINTING
# =============================================================================

SAVE_EVERY = 1

PRINT_EVERY = 20

# =============================================================================
# EVALUATION
# =============================================================================

THRESHOLD = 0.5

# =============================================================================
# VISUALIZATION
# =============================================================================

SAVE_HEATMAPS = True

NUM_VISUALIZATION_SAMPLES = 10