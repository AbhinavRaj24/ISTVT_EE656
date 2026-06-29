# ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection

> **PyTorch implementation of the IEEE TIFS 2023 paper**
>
> **ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection**

---

## Overview

Deepfake generation techniques have become increasingly realistic, making automated detection an important problem in digital forensics. While many deep learning approaches achieve high classification accuracy, most operate as black-box models and fail to explain **why** a video is classified as manipulated.

This project implements the **Interpretable Spatial-Temporal Video Transformer (ISTVT)** proposed in IEEE TIFS 2023. The model jointly models **spatial artifacts** within individual frames and **temporal inconsistencies** across consecutive frames using a transformer-based architecture while providing an interpretable explanation of its predictions through attention-based relevance visualization.

The implementation is developed in **PyTorch** and includes dataset preprocessing, face extraction, transformer architecture, training, evaluation, checkpointing and visualization modules.

---

# Highlights

* PyTorch implementation of ISTVT
* Xception Entry Flow backbone
* Spatial Self-Attention
* Temporal Residual Self-Attention (Self-Subtract)
* Spatial-Temporal Transformer Encoder
* FaceForensics++ preprocessing pipeline
* MTCNN face detection and cropping
* Sliding-window sequence generation
* Manifest-based dataset loading
* Model checkpointing
* Evaluation using Accuracy, Precision, Recall, F1 and ROC-AUC
* Interpretable Spatial-Temporal relevance visualization

---

# Repository Structure

```text
.
├── checkpoints/
├── configs/
│   └── config.py
├── datasets/
│   ├── preprocess.py
│   ├── build_manifest.py
│   ├── dataset.py
│   └── __init__.py
├── models/
│   ├── backbone.py
│   ├── attention.py
│   ├── transformer.py
│   ├── layers.py
│   ├── istvt.py
│   └── visualize.py
├── training/
│   ├── engine.py
│   ├── evaluate.py
│   └── train.py
├── tests/
├── outputs/
├── README.md
└── requirements.txt
```

---

# Method Pipeline

```text
Input Video
      │
      ▼
Frame Extraction
      │
      ▼
Face Detection (MTCNN)
      │
      ▼
300×300 Face Crop
      │
      ▼
Xception Entry Flow
      │
      ▼
19×19 Feature Maps
      │
      ▼
Patch Tokenization
      │
      ▼
Spatial Self-Attention
      │
      ▼
Temporal Residual Self-Attention
      │
      ▼
Transformer Encoder
      │
      ▼
Classification Head
      │
      ├────────────► Real / Fake Prediction
      │
      ▼
Spatial-Temporal Relevance Propagation
      │
      ▼
Interpretable Heatmap
```

---

# Architecture

The ISTVT architecture combines CNN-based feature extraction with Transformer-based spatial-temporal reasoning.

## 1. Face Preprocessing

Each video undergoes:

* Frame extraction
* Face detection using **MTCNN**
* Face cropping around the detected face
* Reflection padding when necessary
* Resize to **300 × 300**

Only facial regions are processed, reducing background noise and computational cost.

---

## 2. Xception Entry Flow

Each cropped face is passed through the Entry Flow of Xception.

Input

```
300 × 300 × 3
```

Output

```
728 × 19 × 19
```

Only the Entry Flow is used, as proposed in the original paper.

---

## 3. Patch Tokenization

Each spatial location becomes a transformer token.

```
19 × 19 = 361 tokens
```

Each token has

```
Embedding Dimension = 728
```

A learnable classification token is added to each frame before transformer processing.

---

## 4. Spatial Self-Attention

Spatial attention models relationships **within a single frame**.

For query **Q**, key **K**, and value **V**

[
Attention(Q,K,V)
================

Softmax
\left(
\frac{QK^T}{\sqrt{d}}
\right)V
]

Multi-head attention enables the network to attend to different facial regions simultaneously.

---

## 5. Temporal Residual Attention

Unlike conventional temporal attention, ISTVT introduces the **Self-Subtract mechanism**.

Residual features are computed as

[
R_t = X_t - X_{t-1}
]

These residual representations emphasize temporal inconsistencies introduced by deepfake manipulation.

Temporal attention is then computed over these residual tokens.

---

## 6. Transformer Encoder

Each encoder block consists of

```
LayerNorm
↓

Spatial Attention
↓

Residual Connection
↓

LayerNorm
↓

Temporal Attention
↓

Residual Connection
↓

Feed Forward Network
↓

Residual Connection
```

The transformer models both spatial and temporal dependencies.

---

## 7. Classification

The transformer output is aggregated through learnable classification tokens and passed through

```
LayerNorm

↓

Linear Layer

↓

Real / Fake
```

---

# Interpretability

One of the major contributions of ISTVT is **model interpretability**.

Instead of acting as a black-box classifier, the model generates spatial-temporal relevance maps indicating **where** and **when** the network focused while making its decision.

The visualization pipeline is

```
Video

↓

Transformer Attention

↓

Spatial Attention Rollout

↓

Temporal Attention Rollout

↓

19×19 Relevance Map

↓

Upsampling

↓

300×300 Heatmap

↓

Overlay on Face
```

The resulting visualization highlights manipulated facial regions, making predictions more transparent and easier to interpret.

---

# Dataset

Current implementation supports

* FaceForensics++ (C23)

Directory structure

```
FaceForensics++_C23/

├── original/

├── Deepfakes/

├── Face2Face/

├── FaceSwap/

├── FaceShifter/

├── NeuralTextures/

└── DeepFakeDetection/
```

---

# Preprocessing

Run

```bash
python -m datasets.preprocess
```

This performs

* Frame extraction
* Face detection
* Face cropping
* Image resizing
* Saving processed face images

---

# Manifest Generation

Generate dataset splits

```bash
python -m datasets.build_manifest
```

Three CSV files are created

```
train.csv

val.csv

test.csv
```

---

# Training

```bash
python -m training.train
```

Training includes

* SGD optimizer
* Warmup scheduler
* Checkpoint saving
* Validation after every epoch

---

# Evaluation

```bash
python -m training.evaluate
```

Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

---

# Configuration

All experiment parameters are defined in

```
configs/config.py
```

including

* Image size
* Sequence length
* Learning rate
* Batch size
* Transformer depth
* Number of heads
* Dataset paths
* Checkpoint locations

---

# Mathematical Summary

For each frame

[
F_i
===

Xception(I_i)
]

Spatial attention

[
S_i
===

Attention(F_i)
]

Temporal residual

[
R_t
===

F_t-F_{t-1}
]

Temporal attention

[
T
=

Attention(R)
]

Classification

[
y
=

MLP(T)
]

---

# Future Improvements

* Cross-dataset evaluation (Celeb-DF v2, DFDC)
* Mixed precision training
* Multi-GPU distributed training
* Larger transformer variants
* Faster preprocessing pipeline
* Real-time inference

---

# References

1. **ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection**, IEEE Transactions on Information Forensics and Security (TIFS), 2023.

2. FaceForensics++: Learning to Detect Manipulated Facial Images.

3. Xception: Deep Learning with Depthwise Separable Convolutions.

4. Attention Is All You Need.

---

# Acknowledgements

This implementation is based on the methodology proposed in the original ISTVT paper and was developed as part of the **EE656 Course Project**. The architecture and implementation were built from the paper and validated against the official implementation for correctness.

---

## License

This repository is intended for **academic and research purposes**.
