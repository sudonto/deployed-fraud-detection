"""Feature construction for the ULB credit card dataset.

The raw frame has three kinds of column and each needs different treatment:

``Time``
    Seconds elapsed since the first transaction, spanning two days. Feeding it
    in raw is harmful under a chronological split, because every test value is
    larger than every training value and a tree simply learns the cut point. It
    is converted to a cyclical time-of-day encoding, which is the part of the
    signal that actually generalises (fraud concentrates in the small hours).
``Amount``
    Heavily right-skewed, so it is log-transformed before scaling.
``V1``-``V28``
    Already PCA projections, but with wildly different variances. Scaling
    matters here because both the resampler and the encoder rely on Euclidean
    distances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.common.io import Dataset, load_dataset

SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24.0


@dataclass
class FraudArrays:
    """Model-ready arrays plus the timestamps needed by the time-aware sampler."""

    X: np.ndarray
    y: np.ndarray
    time: np.ndarray
    feature_names: list[str]


def engineer_features(X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the model matrix and the raw timestamps, kept separate on purpose."""
    frame = X.copy()
    frame.columns = [str(c) for c in frame.columns]

    time_raw = frame["Time"].to_numpy(dtype=float)
    hours = (time_raw / SECONDS_PER_HOUR) % HOURS_PER_DAY
    angle = 2.0 * np.pi * hours / HOURS_PER_DAY

    features = frame.drop(columns=["Time"])
    features["log_amount"] = np.log1p(features["Amount"].to_numpy(dtype=float))
    features = features.drop(columns=["Amount"])
    features["tod_sin"] = np.sin(angle)
    features["tod_cos"] = np.cos(angle)

    return features, time_raw


def load_fraud_arrays(dataset: Dataset | None = None) -> FraudArrays:
    """Load ULB and turn it into arrays, without any fitting yet."""
    dataset = dataset or load_dataset("ulb_creditcard")
    features, time_raw = engineer_features(dataset.X)
    return FraudArrays(
        X=features.to_numpy(dtype=np.float32),
        y=dataset.y.astype(np.int8),
        time=time_raw,
        feature_names=list(features.columns),
    )


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit the scaler on training rows only. Called once per split, never before it."""
    return StandardScaler().fit(X_train)


def describe_split_drift(
    time: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> dict[str, float]:
    """Report the temporal gap between train and test.

    Included in the paper to make the chronological protocol auditable: under a
    correct split the maximum training timestamp is below the minimum test one.
    """
    return {
        "train_time_min_h": float(time[train_idx].min() / SECONDS_PER_HOUR),
        "train_time_max_h": float(time[train_idx].max() / SECONDS_PER_HOUR),
        "test_time_min_h": float(time[test_idx].min() / SECONDS_PER_HOUR),
        "test_time_max_h": float(time[test_idx].max() / SECONDS_PER_HOUR),
        "chronological": bool(time[train_idx].max() <= time[test_idx].min()),
    }
