"""Evaluation metrics shared by both papers.

Threshold-free metrics come first because they are the honest way to compare
models on imbalanced data: accuracy on the ULB dataset is above 99.8% for a
classifier that predicts "legitimate" every single time.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Metrics where a larger value is better. Used by the reporting layer to decide
# which direction to bold in the LaTeX tables.
HIGHER_IS_BETTER = {
    "auprc": True,
    "roc_auc": True,
    "mcc": True,
    "f1": True,
    "precision": True,
    "recall": True,
    "g_mean": True,
    "ks": True,
    "balanced_accuracy": True,
    "precision_at_k": True,
    "recall_at_p90": True,
    "brier": False,
    "fpr_at_recall80": False,
}


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation, the standard credit scoring metric."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def g_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Geometric mean of sensitivity and specificity."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return float(np.sqrt(sensitivity * specificity))


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int | None = None) -> float:
    """Precision within the k highest-scoring cases.

    This mirrors how a fraud team actually works: only the top slice of alerts
    ever reaches a human investigator. When k is omitted it defaults to the
    number of true positives present, which makes the metric self-scaling.
    """
    if k is None:
        k = int(y_true.sum())
    k = max(int(k), 1)
    order = np.argsort(-y_score, kind="stable")[:k]
    return float(y_true[order].mean())


def recall_at_precision(
    y_true: np.ndarray, y_score: np.ndarray, min_precision: float = 0.90
) -> float:
    """Best recall achievable while holding precision at or above a floor.

    Returns 0.0 when no threshold reaches the requested precision.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    feasible = precision >= min_precision
    if not feasible.any():
        return 0.0
    return float(recall[feasible].max())


def fpr_at_recall(
    y_true: np.ndarray, y_score: np.ndarray, min_recall: float = 0.80
) -> float:
    """False positive rate at the operating point that first reaches a recall.

    The count of legitimate customers wrongly blocked is the cost side of the
    trade-off, so it belongs next to recall rather than buried in a curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    feasible = tpr >= min_recall
    if not feasible.any():
        return 1.0
    return float(fpr[feasible].min())


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold maximising F1, selected on validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one more point than thresholds.
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    k: int | None = None,
) -> dict[str, float]:
    """Score one set of predictions on every metric used in the papers."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        # Threshold-free.
        "auprc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
        "precision_at_k": precision_at_k(y_true, y_score, k),
        "recall_at_p90": recall_at_precision(y_true, y_score, 0.90),
        "fpr_at_recall80": fpr_at_recall(y_true, y_score, 0.80),
        # Threshold-dependent.
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "g_mean": g_mean(y_true, y_pred),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        # Raw counts, so any other metric can be recomputed from the records.
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# The metrics each paper leads with.
PAPER1_HEADLINE = ["auprc", "mcc", "f1", "precision_at_k", "recall_at_p90", "roc_auc"]
PAPER2_HEADLINE = ["roc_auc", "ks", "auprc", "f1", "g_mean", "brier"]
