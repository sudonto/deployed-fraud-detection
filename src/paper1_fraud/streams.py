"""Turn each fraud dataset into a chronologically ordered stream.

The feedback study replays a dataset the way a bank would have lived through
it, so every dataset has to be reduced to the same three things: a model matrix,
a label, and a round index saying which retraining cycle each row belongs to.
Where the rounds come from differs by dataset and is the one part that cannot be
shared, because the datasets record time very differently.

``baf_base``
    A ``month`` column with eight values, which are the rounds. Nothing to
    choose here, and the shift in fraud rate across those months (0.88% to
    1.48%) is documented by the dataset's authors.
``ieee_cis``
    ``TransactionDT`` in seconds from a fixed origin, spanning about six
    months. Cut into weeks, which gives a couple of dozen rounds with several
    hundred frauds each.
``ulb_creditcard``
    ``Time`` in seconds, spanning two days. Cut into equal blocks. Kept for
    continuity with the prior work that uses it, with the caveat that two days
    of data cannot show anything that needs to compound.

Categorical columns are ordinal-encoded rather than one-hot expanded. The
detector is a gradient boosted tree throughout this study, LightGBM handles
categorical splits natively, and one-hot expanding IEEE-CIS would turn 455
columns into several thousand for no gain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.io import load_dataset

SECONDS_PER_DAY = 86_400.0


@dataclass
class Stream:
    """A dataset in the form the replay needs."""

    key: str
    X: np.ndarray
    y: np.ndarray
    round_index: np.ndarray
    feature_names: list[str]
    categorical_indices: list[int]

    @property
    def n_rounds(self) -> int:
        return int(self.round_index.max()) + 1

    def round_table(self) -> pd.DataFrame:
        frame = pd.DataFrame({"round": self.round_index, "y": self.y})
        table = frame.groupby("round")["y"].agg(["size", "sum", "mean"])
        table.columns = ["rows", "frauds", "rate"]
        return table


def _encode(X: pd.DataFrame, categorical: list[str]) -> tuple[np.ndarray, list[int]]:
    """Ordinal-encode the categoricals and coerce the rest to float."""
    frame = X.copy()
    indices = []
    for position, column in enumerate(frame.columns):
        if column in categorical:
            codes = pd.Categorical(frame[column].astype(str)).codes
            frame[column] = codes.astype(np.float32)
            indices.append(position)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    return frame.to_numpy(dtype=np.float32), indices


def _equal_frequency_rounds(order_key: np.ndarray, n_rounds: int) -> np.ndarray:
    """Assign rounds so each holds a similar number of rows.

    Equal-width blocks over a timestamp leave some rounds nearly empty when the
    volume is uneven, and a round with a handful of transactions in it produces
    a review budget of zero and an undefined precision.
    """
    ranks = np.argsort(np.argsort(order_key, kind="stable"), kind="stable")
    return np.minimum((ranks * n_rounds) // len(ranks), n_rounds - 1).astype(int)


def load_stream(
    key: str, n_rounds: int | None = None, subsample: float | None = None,
    seed: int = 0,
) -> Stream:
    """Load one dataset as a chronologically ordered stream of rounds.

    ``subsample`` keeps a stratified share of the rows, for checking that a
    change to the simulation still runs before spending an hour on the full
    million. It is never used for a result that goes in the paper.
    """
    dataset = load_dataset(key)
    X, y = dataset.X, dataset.y

    if subsample is not None and 0 < subsample < 1:
        rng = np.random.default_rng(seed)
        keep = np.zeros(len(y), dtype=bool)
        for label in (0, 1):
            index = np.flatnonzero(y == label)
            chosen = rng.choice(index, size=int(len(index) * subsample), replace=False)
            keep[chosen] = True
        X, y = X.loc[keep].reset_index(drop=True), y[keep]

    if key.startswith("baf"):
        rounds = X["month"].to_numpy().astype(int)
        features = X.drop(columns=["month"])
        categorical = [c for c in dataset.categorical_columns if c != "month"]
        order = np.argsort(rounds, kind="stable")

    elif key == "ieee_cis":
        time = pd.to_numeric(X["TransactionDT"], errors="coerce").to_numpy(dtype=float)
        order = np.argsort(time, kind="stable")
        # Weeks, because that is roughly how often a fraud team retrains, and it
        # leaves several hundred frauds per round across the six-month span.
        weeks = ((time - np.nanmin(time)) // (7 * SECONDS_PER_DAY)).astype(int)
        rounds = weeks
        features = X.drop(columns=["TransactionDT"])
        categorical = [c for c in dataset.categorical_columns if c != "TransactionDT"]

    elif key == "ulb_creditcard":
        time = X["Time"].to_numpy(dtype=float)
        order = np.argsort(time, kind="stable")
        rounds = _equal_frequency_rounds(time, n_rounds or 10)
        features = X.drop(columns=["Time"])
        categorical = []

    else:
        raise ValueError(f"{key!r} has no stream definition")

    matrix, categorical_indices = _encode(features, categorical)
    rounds = rounds[order]
    # Rounds are renumbered from zero with no gaps, so an empty week in the
    # middle of IEEE-CIS does not leave a round that nothing can be trained on.
    _, rounds = np.unique(rounds, return_inverse=True)

    return Stream(
        key=key,
        X=matrix[order],
        y=y[order].astype(np.int8),
        round_index=rounds.astype(int),
        feature_names=list(features.columns),
        categorical_indices=categorical_indices,
    )
