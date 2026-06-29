"""
train.py
========

Training script for ISTVT.
"""

import torch
import torch.nn as nn

from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

from configs.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    CHECKPOINT_DIR,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
)

from datasets.dataset import ISTVTDataset
from models.istvt import ISTVT

from training.engine import (
    train_one_epoch,
    evaluate,
    save_checkpoint,
)

from training.evaluate import (
    compute_metrics,
    print_metrics,
)


def main():

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_dataset = ISTVTDataset(TRAIN_MANIFEST)

    val_dataset = ISTVTDataset(VAL_MANIFEST)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train sequences : {len(train_dataset)}")
    print(f"Validation sequences : {len(val_dataset)}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = ISTVT()

    model.to(DEVICE)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------
    # Warmup Scheduler
    # --------------------------------------------------

    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=5,
    )

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------

    best_acc = 0.0

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
            scheduler,
        )

        val_loss, y_true, y_pred, y_score = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE,
        )

        metrics = compute_metrics(
            y_true,
            y_pred,
            y_score,
        )

        val_acc = metrics["accuracy"]

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")

        print_metrics(metrics)

        if val_acc > best_acc:

            best_acc = val_acc

            save_checkpoint(
                model,
                optimizer,
                epoch,
                CHECKPOINT_DIR / "best_model.pth",
            )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()