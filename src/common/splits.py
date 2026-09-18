"""Data splitting protocols.

Two regimes are supported:

* Time-ordered splitting for the fraud study, where the test set must consist of
  transactions that occur strictly after everything the model was fitted on.
  Random splitting on that dataset lets a model see the future, and combined
  with oversampling it is the main reason published ULB numbers are optimistic.
* Stratified k-fold for the credit scoring datasets, which carry no usable
  timestamp and are conventionally evaluated that way.

Both regimes return a train / validation / test triple. The validation portion
exists so that early stopping and the decision threshold are never chosen on
test data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class Split:
    """Index arrays for one train / validation / test partition."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    fold: int = 0
    name: str = ""

    def sizes(self) -> dict[str, int]:
        return {
            "n_train": len(self.train_idx),
            "n_val": len(self.val_idx),
            "n_test": len(self.test_idx),
        }


def time_ordered_split(
    time_values: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.15,
) -> Split:
    """Single chronological cut: oldest -> train, then validation, then test."""
    order = np.argsort(np.asarray(time_values), kind="stable")
    n = len(order)
    n_test = int(round(n * test_size))
    n_remaining = n - n_test
    n_val = int(round(n_remaining * val_size))

    train_idx = order[: n_remaining - n_val]
    val_idx = order[n_remaining - n_val : n_remaining]
    test_idx = order[n_remaining:]
    return Split(train_idx, val_idx, test_idx, fold=0, name="time_ordered")


def rolling_origin_splits(
    time_values: np.ndarray,
    n_folds: int = 5,
    test_size: float = 0.1,
    val_size: float = 0.15,
    expanding: bool = True,
) -> list[Split]:
    """Rolling-origin evaluation: several chronological cuts over the stream.

    Fold i trains on everything before origin i and tests on the window that
    follows it, so each fold answers "how well would this model have worked had
    it been deployed at that moment".
    """
    order = np.argsort(np.asarray(time_values), kind="stable")
    n = len(order)
    window = int(round(n * test_size))
    if window < 1:
        raise ValueError("test_size too small for this dataset")

    # Leave enough history for the earliest fold to have a usable training set.
    first_origin = n - n_folds * window
    if first_origin < window:
        raise ValueError(
            f"cannot build {n_folds} folds of {window} samples from {n} rows"
        )

    splits: list[Split] = []
    for fold in range(n_folds):
        origin = first_origin + fold * window
        history = order[:origin] if expanding else order[max(0, origin - first_origin) : origin]
        n_val = int(round(len(history) * val_size))
        splits.append(
            Split(
                train_idx=history[: len(history) - n_val],
                val_idx=history[len(history) - n_val :],
                test_idx=order[origin : origin + window],
                fold=fold,
                name=f"rolling_origin_fold{fold}",
            )
        )
    return splits


def stratified_splits(
    y: np.ndarray,
    n_folds: int = 5,
    val_size: float = 0.15,
    seed: int = 0,
) -> list[Split]:
    """Stratified k-fold, with a stratified validation slice cut from each train fold."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits: list[Split] = []
    for fold, (train_full, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        train_idx, val_idx = train_test_split(
            train_full,
            test_size=val_size,
            stratify=y[train_full],
            random_state=seed,
        )
        splits.append(
            Split(
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                fold=fold,
                name=f"stratified_fold{fold}",
            )
        )
    return splits


def leaky_random_split(
    y: np.ndarray, test_size: float = 0.2, val_size: float = 0.15, seed: int = 0
) -> Split:
    """A deliberately naive random split.

    Used only as the control arm of the leakage ablation in Paper 1, to quantify
    how much of the headline performance in the literature comes from ignoring
    the temporal ordering.
    """
    idx = np.arange(len(y))
    train_full, test_idx = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_full, test_size=val_size, stratify=y[train_full], random_state=seed
    )
    return Split(train_idx, val_idx, test_idx, fold=0, name="leaky_random")
