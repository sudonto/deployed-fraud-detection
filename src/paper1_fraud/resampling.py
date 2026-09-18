"""Time-Aware Stratified Resampling (TASR).

Standard SMOTE treats the minority class as one undifferentiated cloud. On
transaction data that assumption is wrong twice over:

1. Fraud arrives in bursts. Interpolating between a fraud at 03:00 on day one
   and a fraud at 14:00 on day two produces a point that corresponds to no
   plausible transaction, because the two came from different attack episodes.
2. The synthetic points inherit no timestamp, so the resampled training set
   silently loses the temporal structure the rest of the pipeline relies on.

TASR fixes both. The training window is cut into equal-count temporal blocks;
synthetic minority points are interpolated only between fraud cases inside the
same block, and each one is given an interpolated timestamp of its own. The
synthetic budget is distributed across blocks in proportion to how much fraud
each block actually contains, so the resampled data keeps the original burst
profile instead of flattening it.

A cleaning pass then removes majority points that sit on the decision boundary.
Both cleaning rules are anchored on the minority class, which is exact for Tomek
links (every Tomek pair has one member of each class) and is what makes the pass
affordable on 200k+ rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class ResampleResult:
    X: np.ndarray
    y: np.ndarray
    time: np.ndarray
    stats: dict = field(default_factory=dict)


class TimeAwareStratifiedResampler:
    """The TASR operator.

    Parameters
    ----------
    target_ratio:
        Desired minority-to-majority ratio after oversampling. 0.10 means one
        synthetic-or-real fraud for every ten legitimate transactions. Full
        balancing (1.0) is deliberately not the default; on a dataset with an
        imbalance ratio of 578:1 it would fabricate 99.8% of the training set.
    n_blocks:
        Number of equal-count temporal blocks. The default of 24 gives roughly
        hour-sized blocks on the two-day ULB window.
    k_neighbors:
        Neighbourhood size for interpolation, capped per block by how many
        minority cases that block contains.
    cleaning:
        ``"tomek"``, ``"enn"`` or ``None``.
    """

    def __init__(
        self,
        target_ratio: float = 0.10,
        n_blocks: int = 24,
        k_neighbors: int = 5,
        cleaning: str | None = "tomek",
        jitter_scale: float = 0.05,
        random_state: int = 0,
    ) -> None:
        if not 0 < target_ratio <= 1.0:
            raise ValueError("target_ratio must be in (0, 1]")
        if cleaning not in {"tomek", "enn", None}:
            raise ValueError("cleaning must be 'tomek', 'enn' or None")
        self.target_ratio = target_ratio
        self.n_blocks = n_blocks
        self.k_neighbors = k_neighbors
        self.cleaning = cleaning
        self.jitter_scale = jitter_scale
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    def fit_resample(
        self, X: np.ndarray, y: np.ndarray, time: np.ndarray
    ) -> ResampleResult:
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int8)
        time = np.asarray(time, dtype=float)

        n_minority = int(y.sum())
        n_majority = int(len(y) - n_minority)
        if n_minority == 0:
            raise ValueError("no minority samples in the training fold")

        n_desired = int(round(self.target_ratio * n_majority))
        n_synthetic = max(0, n_desired - n_minority)

        blocks = self._assign_blocks(time)
        per_block = self._allocate_budget(blocks, y, n_synthetic, rng)

        synth_X, synth_t = [], []
        block_report = []
        for block_id, budget in per_block.items():
            if budget <= 0:
                continue
            member = np.flatnonzero((blocks == block_id) & (y == 1))
            new_X, new_t, mode = self._interpolate_within_block(
                X[member], time[member], budget, rng
            )
            synth_X.append(new_X)
            synth_t.append(new_t)
            block_report.append(
                {"block": int(block_id), "n_fraud": len(member), "n_synth": int(budget), "mode": mode}
            )

        if synth_X:
            X_out = np.vstack([X, np.vstack(synth_X)])
            t_out = np.concatenate([time, np.concatenate(synth_t)])
            y_out = np.concatenate([y, np.ones(sum(len(a) for a in synth_X), dtype=np.int8)])
        else:
            X_out, y_out, t_out = X, y, time

        stats = {
            "n_before": int(len(y)),
            "n_minority_before": n_minority,
            "n_synthetic": int(X_out.shape[0] - X.shape[0]),
            "n_blocks_used": len(block_report),
            "blocks": block_report,
        }

        if self.cleaning:
            keep = self._clean(X_out, y_out)
            removed = int((~keep).sum())
            X_out, y_out, t_out = X_out[keep], y_out[keep], t_out[keep]
            stats["n_cleaned"] = removed

        stats["n_after"] = int(len(y_out))
        stats["n_minority_after"] = int(y_out.sum())
        stats["ratio_after"] = float(y_out.sum() / max(1, len(y_out) - y_out.sum()))

        # Restore chronological order so downstream code can still assume it.
        order = np.argsort(t_out, kind="stable")
        return ResampleResult(X_out[order], y_out[order], t_out[order], stats)

    # ------------------------------------------------------------------ #
    def _assign_blocks(self, time: np.ndarray) -> np.ndarray:
        """Equal-count temporal blocks, so each block carries comparable volume."""
        order = np.argsort(time, kind="stable")
        blocks = np.empty(len(time), dtype=np.int32)
        edges = np.array_split(order, self.n_blocks)
        for block_id, chunk in enumerate(edges):
            blocks[chunk] = block_id
        return blocks

    def _allocate_budget(
        self, blocks: np.ndarray, y: np.ndarray, n_synthetic: int, rng
    ) -> dict[int, int]:
        """Split the synthetic budget across blocks proportional to fraud density."""
        counts = {}
        for block_id in np.unique(blocks):
            counts[int(block_id)] = int(((blocks == block_id) & (y == 1)).sum())

        total = sum(counts.values())
        if total == 0 or n_synthetic == 0:
            return {b: 0 for b in counts}

        exact = {b: n_synthetic * c / total for b, c in counts.items()}
        budget = {b: int(np.floor(v)) for b, v in exact.items()}

        # Hand out the rounding remainder to the blocks with the largest fraction.
        shortfall = n_synthetic - sum(budget.values())
        if shortfall > 0:
            remainders = sorted(
                ((exact[b] - budget[b], b) for b in counts if counts[b] > 0), reverse=True
            )
            for _, b in remainders[:shortfall]:
                budget[b] += 1
        return budget

    def _interpolate_within_block(
        self, X_block: np.ndarray, t_block: np.ndarray, budget: int, rng
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Generate synthetic frauds using only same-block neighbours."""
        n = len(X_block)

        if n == 1:
            # A single fraud cannot be interpolated; perturb it instead, using a
            # scale tied to the feature spread so the jitter stays plausible.
            base = np.repeat(X_block, budget, axis=0)
            scale = self.jitter_scale * (np.abs(X_block).mean() + 1e-6)
            new_X = base + rng.normal(0.0, scale, size=base.shape).astype(np.float32)
            return new_X, np.repeat(t_block, budget), "jitter"

        k = int(min(self.k_neighbors, n - 1))
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_block)
        _, neighbours = nn.kneighbors(X_block)
        neighbours = neighbours[:, 1:]  # drop self

        base_idx = rng.integers(0, n, size=budget)
        pick = rng.integers(0, k, size=budget)
        mate_idx = neighbours[base_idx, pick]
        lam = rng.random(size=(budget, 1)).astype(np.float32)

        new_X = X_block[base_idx] + lam * (X_block[mate_idx] - X_block[base_idx])
        # The synthetic point sits between two real frauds in feature space, so
        # it is given the matching position in time.
        new_t = t_block[base_idx] + lam[:, 0] * (t_block[mate_idx] - t_block[base_idx])
        return new_X.astype(np.float32), new_t, "smote"

    # ------------------------------------------------------------------ #
    def _clean(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.cleaning == "tomek":
            return _tomek_keep_mask(X, y)
        return _enn_keep_mask(X, y)


def _tomek_keep_mask(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Drop the majority member of every Tomek link.

    Anchoring on the minority class is exact rather than an approximation: a
    Tomek link is a mutual nearest-neighbour pair from opposite classes, so every
    such pair contains exactly one minority point.
    """
    minority = np.flatnonzero(y == 1)
    nn = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(X)

    _, nbr_of_min = nn.kneighbors(X[minority])
    candidates = nbr_of_min[:, 1]
    is_majority = y[candidates] == 0
    minority, candidates = minority[is_majority], candidates[is_majority]
    if len(candidates) == 0:
        return np.ones(len(y), dtype=bool)

    # Mutual check: the majority point must point back at the same minority point.
    _, nbr_of_maj = nn.kneighbors(X[candidates])
    mutual = nbr_of_maj[:, 1] == minority

    keep = np.ones(len(y), dtype=bool)
    keep[np.unique(candidates[mutual])] = False
    return keep


def _enn_keep_mask(X: np.ndarray, y: np.ndarray, k: int = 3) -> np.ndarray:
    """Edited nearest neighbours, restricted to removing majority points."""
    minority = np.flatnonzero(y == 1)
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)

    # Only majority points that are near a fraud can be misclassified by their
    # own neighbourhood in a way that matters, so those are the ones we test.
    _, near_min = nn.kneighbors(X[minority])
    suspects = np.unique(near_min[:, 1:].ravel())
    suspects = suspects[y[suspects] == 0]
    if len(suspects) == 0:
        return np.ones(len(y), dtype=bool)

    _, nbr = nn.kneighbors(X[suspects])
    votes = y[nbr[:, 1:]].mean(axis=1)

    keep = np.ones(len(y), dtype=bool)
    keep[suspects[votes > 0.5]] = False
    return keep


# --------------------------------------------------------------------------- #
# Reference samplers for the comparison table.
# --------------------------------------------------------------------------- #
def build_reference_sampler(name: str, target_ratio: float = 0.10, random_state: int = 0):
    """Return an imbalanced-learn sampler, or ``None`` for the untouched baseline."""
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    name = name.lower()
    if name in {"none", "baseline"}:
        return None

    common = {"sampling_strategy": target_ratio, "random_state": random_state}
    if name == "smote":
        return SMOTE(k_neighbors=5, **common)
    if name == "borderline_smote":
        return BorderlineSMOTE(k_neighbors=5, **common)
    if name == "adasyn":
        return ADASYN(n_neighbors=5, **common)
    if name == "random_over":
        return RandomOverSampler(**common)
    if name == "random_under":
        return RandomUnderSampler(sampling_strategy=target_ratio, random_state=random_state)
    if name == "smote_tomek":
        return SMOTETomek(smote=SMOTE(k_neighbors=5, **common), random_state=random_state)
    if name == "smote_enn":
        return SMOTEENN(smote=SMOTE(k_neighbors=5, **common), random_state=random_state)
    raise ValueError(f"unknown sampler {name!r}")


REFERENCE_SAMPLERS = [
    "none",
    "random_over",
    "random_under",
    "smote",
    "borderline_smote",
    "adasyn",
    "smote_tomek",
    "smote_enn",
]
