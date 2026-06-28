"""
engine.py
=========

Training and evaluation engine for ISTVT.

This module implements the standard PyTorch training loop used by the
ISTVT paper.
"""

from pathlib import Path
from sched import scheduler

import torch
from tqdm import tqdm
import numpy as np


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scheduler=None,
):
    """
    Train the model for one epoch.

    Returns
    -------
    float
        Average training loss.
    """

    model.train()

    running_loss = 0.0

    for frames, labels in tqdm(dataloader, desc="Training"):

        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(frames)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Evaluate the model.

    Returns
    -------
    tuple
        (
            average_loss,
            y_true,
            y_pred,
            y_score,
        )
    """

    model.eval()

    running_loss = 0.0

    y_true = []
    y_pred = []
    y_score = []

    for frames, labels in tqdm(dataloader, desc="Validation"):

        frames = frames.to(device)
        labels = labels.to(device)

        outputs = model(frames)

        loss = criterion(outputs, labels)

        running_loss += loss.item()

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )[:, 1]

        predictions = outputs.argmax(dim=1)

        y_true.extend(
            labels.cpu().numpy()
        )

        y_pred.extend(
            predictions.cpu().numpy()
        )

        y_score.extend(
            probabilities.cpu().numpy()
        )

    return (
        running_loss / len(dataloader),
        np.array(y_true),
        np.array(y_pred),
        np.array(y_score),
    )

def save_checkpoint(
    model,
    optimizer,
    epoch,
    path,
):
    """
    Save training checkpoint.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    model,
    optimizer,
    path,
    device,
):
    """
    Load checkpoint.

    Returns
    -------
    int
        Epoch number.
    """

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return checkpoint["epoch"]