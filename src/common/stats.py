"""Statistical tests for model comparison.

A table of point estimates does not establish that one model beats another, so
every headline claim in both papers is backed by one of these:

* bootstrap confidence intervals and a paired bootstrap test, which work for any
  metric including AUPRC (DeLong does not);
* the DeLong test, for ROC-AUC specifically, because reviewers expect it;
* Friedman followed by post-hoc Nemenyi, for ranking many models across the
  several datasets in Paper 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
@dataclass
class Interval:
    point: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "ci_low": self.low, "ci_high": self.high}


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Stratified bootstrap interval for any metric of (y_true, y_score).

    Resampling is done within each class so that a bootstrap replicate of an
    extremely imbalanced dataset cannot end up with zero positives.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    point = float(metric_fn(y_true, y_score))
    values = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ]
        )
        values[b] = metric_fn(y_true[idx], y_score[idx])

    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return Interval(point, float(low), float(high))


def paired_bootstrap_test(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Two-sided paired bootstrap test for metric(A) - metric(B).

    Both models are scored on the identical resampled rows, which removes the
    variance caused by the sample itself and tests only the model difference.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    observed = float(metric_fn(y_true, score_a)) - float(metric_fn(y_true, score_b))
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ]
        )
        diffs[b] = metric_fn(y_true[idx], score_a[idx]) - metric_fn(y_true[idx], score_b[idx])

    # p-value from the proportion of centred replicates at least as extreme.
    centred = diffs - diffs.mean()
    p_value = float((np.abs(centred) >= abs(observed)).mean())
    low, high = np.quantile(diffs, [0.025, 0.975])
    return {
        "observed_diff": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": max(p_value, 1.0 / n_boot),
    }


# --------------------------------------------------------------------------- #
# DeLong test for ROC-AUC
# --------------------------------------------------------------------------- #
def _midrank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, vectorised.

    The obvious implementation walks the sorted array in Python to find runs of
    equal values, which costs seconds on the hundred-thousand-row pooled score
    vectors this is called on. Marking the run boundaries with a comparison and
    resolving them with two cumulative reductions does the same work in numpy.
    """
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)

    # start[i] is the index where the run containing i begins; end[i] where it ends.
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    np.not_equal(sorted_x[1:], sorted_x[:-1], out=is_new[1:])
    group = np.cumsum(is_new) - 1
    starts = np.flatnonzero(is_new)
    ends = np.append(starts[1:], n) - 1

    ranks = 0.5 * (starts[group] + ends[group]) + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _fast_delong(scores: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Sun & Xu (2014) fast DeLong. ``scores`` is (n_models, n_samples), positives first."""
    m, n = n_pos, scores.shape[1] - n_pos
    positives, negatives = scores[:, :m], scores[:, m:]
    k = scores.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _midrank(positives[r])
        ty[r] = _midrank(negatives[r])
        tz[r] = _midrank(scores[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    covariance = sx / m + sy / n
    return aucs, np.atleast_2d(covariance)


def delong_test(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray
) -> dict[str, float]:
    """DeLong test for two correlated ROC curves on the same samples."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind="stable")  # positives first
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    scores = np.vstack([np.asarray(score_a)[order], np.asarray(score_b)[order]])

    aucs, cov = _fast_delong(scores, n_pos)
    contrast = np.array([[1.0, -1.0]])
    variance = float(contrast @ cov @ contrast.T)
    if variance <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": 0.0, "p_value": 1.0}

    z = float((aucs[0] - aucs[1]) / np.sqrt(variance))
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "diff": float(aucs[0] - aucs[1]),
        "z": z,
        "p_value": p,
    }


# --------------------------------------------------------------------------- #
# Multi-model, multi-dataset comparison
# --------------------------------------------------------------------------- #
def friedman_nemenyi(scores: dict[str, list[float]]) -> dict:
    """Friedman test plus post-hoc Nemenyi over models measured on several datasets.

    ``scores`` maps a model name to its metric on each dataset, with the dataset
    order identical across models. Returns average ranks (rank 1 = best), the
    Friedman p-value, the critical difference at alpha=0.05, and pairwise
    Nemenyi p-values when the library is available.
    """
    names = list(scores)
    matrix = np.asarray([scores[n] for n in names], dtype=float)  # (models, datasets)
    n_models, n_datasets = matrix.shape
    if n_models < 3 or n_datasets < 2:
        raise ValueError("Friedman needs at least 3 models and 2 datasets")

    # Rank per dataset, 1 = best (metrics here are all higher-is-better).
    ranks = np.apply_along_axis(stats.rankdata, 0, -matrix)
    avg_ranks = ranks.mean(axis=1)

    # friedmanchisquare expects one array per model, measured across datasets.
    statistic, p_value = stats.friedmanchisquare(*[matrix[i] for i in range(n_models)])

    # Nemenyi critical difference (Demsar 2006).
    q_alpha = _nemenyi_q_alpha(n_models)
    cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6.0 * n_datasets))

    result = {
        "models": names,
        "average_ranks": {n: float(r) for n, r in zip(names, avg_ranks)},
        "friedman_statistic": float(statistic),
        "friedman_p_value": float(p_value),
        "critical_difference": float(cd),
        "n_datasets": n_datasets,
    }

    try:
        import scikit_posthocs as sp
        import pandas as pd

        long = pd.DataFrame(matrix.T, columns=names)
        posthoc = sp.posthoc_nemenyi_friedman(long.to_numpy())
        posthoc.index = posthoc.columns = names
        result["nemenyi_p_values"] = posthoc.to_dict()
    except ImportError:
        result["nemenyi_p_values"] = None

    return result


def _nemenyi_q_alpha(n_models: int) -> float:
    """Studentised range critical values at alpha=0.05 (Demsar 2006, Table 5)."""
    table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
        9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
        15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
    }
    if n_models in table:
        return table[n_models]
    return table[max(table)]


def wilcoxon_holm(scores: dict[str, list[float]], reference: str) -> dict[str, dict]:
    """Wilcoxon signed-rank of every model against a reference, Holm-corrected."""
    others = [n for n in scores if n != reference]
    raw = {}
    for name in others:
        a = np.asarray(scores[reference], dtype=float)
        b = np.asarray(scores[name], dtype=float)
        if np.allclose(a, b):
            raw[name] = 1.0
            continue
        raw[name] = float(stats.wilcoxon(a, b).pvalue)

    ordered = sorted(raw, key=raw.get)
    corrected, running = {}, 0.0
    for rank, name in enumerate(ordered):
        adjusted = min(1.0, raw[name] * (len(ordered) - rank))
        running = max(running, adjusted)
        corrected[name] = running

    return {
        name: {"p_raw": raw[name], "p_holm": corrected[name]} for name in others
    }
