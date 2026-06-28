"""
evaluate.py
===========

Evaluation metrics for ISTVT.

This module computes the metrics reported in the ISTVT paper.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(
    y_true,
    y_pred,
    y_score,
):
    """
    Compute evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    y_score : np.ndarray
        Probability of the fake class.

    Returns
    -------
    dict
    """

    metrics = {

        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "auc": roc_auc_score(
            y_true,
            y_score,
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ),
    }

    return metrics


def print_metrics(metrics):
    """
    Pretty-print evaluation metrics.
    """

    print("\nEvaluation Results")
    print("-" * 30)

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"ROC AUC  : {metrics['auc']:.4f}")

    print("\nConfusion Matrix")

    print(metrics["confusion_matrix"])