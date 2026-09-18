"""Significance testing on the stored per-sample predictions.

Comparing models on five fold-level numbers gives a test almost no power. Since
every run also stores its test-set scores, the comparisons here are made on the
predictions themselves: a stratified paired bootstrap for AUPRC, which is the
headline metric for fraud, and DeLong's test for ROC-AUC, which has an exact
variance estimator for correlated ROC curves on shared samples.

Folds are concatenated in a fixed order before testing. Both models in a pair
see exactly the same rows in the same order, so the pairing that both tests rely
on is preserved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common.io import PROJECT_ROOT, load_scores
from src.common.stats import bootstrap_ci, delong_test, paired_bootstrap_test


def _fingerprint(*parts) -> str:
    """A short key that changes whenever any input to a test changes."""
    digest = hashlib.blake2b(digest_size=12)
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(np.ascontiguousarray(part, dtype=np.float64).tobytes())
        else:
            digest.update(repr(part).encode("utf-8"))
    return digest.hexdigest()


def _cached(paper: int, key: str, compute):
    """Memoise one test to disk.

    A paired bootstrap over five thousand resamples of a hundred and seventy
    thousand pooled predictions takes minutes, and the analysis is rerun after
    every stage of a sweep that lasts days. The key includes a digest of the
    score arrays, so a rerun experiment invalidates its own cached tests.
    """
    path = PROJECT_ROOT / "results" / f"paper{paper}" / "cache" / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


@dataclass
class Comparison:
    reference: str
    comparator: str
    n_samples: int
    n_positives: int
    reference_auprc: float
    comparator_auprc: float
    delta_auprc: float
    auprc_ci: tuple[float, float]
    auprc_p: float
    reference_auc: float
    comparator_auc: float
    delta_auc: float
    delong_p: float

    def as_dict(self) -> dict:
        return {
            "reference": self.reference,
            "comparator": self.comparator,
            "n_samples": self.n_samples,
            "n_positives": self.n_positives,
            "reference_auprc": self.reference_auprc,
            "comparator_auprc": self.comparator_auprc,
            "delta_auprc": self.delta_auprc,
            "auprc_ci_low": self.auprc_ci[0],
            "auprc_ci_high": self.auprc_ci[1],
            "auprc_p": self.auprc_p,
            "reference_auc": self.reference_auc,
            "comparator_auc": self.comparator_auc,
            "delta_auc": self.delta_auc,
            "delong_p": self.delong_p,
        }


def pooled_scores(
    paper: int, stage: str, arm: str, folds: list[int], seed: int = 0
) -> tuple[np.ndarray, np.ndarray] | None:
    """Concatenate a model's test predictions over folds, in fold order."""
    truths, predictions = [], []
    for fold in folds:
        loaded = load_scores(paper, stage, arm, fold, seed)
        if loaded is None:
            return None
        y_true, scores = loaded
        truths.append(np.asarray(y_true))
        predictions.append(np.asarray(scores))
    if not truths:
        return None
    return np.concatenate(truths), np.concatenate(predictions)


def compare(
    paper: int,
    reference: tuple[str, str],
    comparators: list[tuple[str, str]],
    folds: list[int],
    seed: int = 0,
    n_boot: int = 5000,
) -> list[Comparison]:
    """Test the reference model against each comparator on pooled predictions."""
    reference_loaded = pooled_scores(paper, reference[0], reference[1], folds, seed)
    if reference_loaded is None:
        return []
    y_true, reference_scores = reference_loaded

    results = []
    for stage, arm in comparators:
        if (stage, arm) == reference:
            continue
        loaded = pooled_scores(paper, stage, arm, folds, seed)
        if loaded is None:
            continue
        comparator_truth, comparator_scores = loaded
        if len(comparator_truth) != len(y_true) or not np.array_equal(comparator_truth, y_true):
            # Different rows would silently invalidate the pairing.
            continue

        key = _fingerprint(
            "compare", paper, reference, (stage, arm), n_boot, seed,
            y_true, reference_scores, comparator_scores,
        )
        outcome = _cached(paper, key, lambda: {
            "bootstrap": paired_bootstrap_test(
                y_true, reference_scores, comparator_scores,
                average_precision_score, n_boot=n_boot, seed=seed,
            ),
            "delong": delong_test(y_true, reference_scores, comparator_scores),
        })
        bootstrap, delong = outcome["bootstrap"], outcome["delong"]

        results.append(
            Comparison(
                reference=reference[1],
                comparator=arm,
                n_samples=int(len(y_true)),
                n_positives=int(y_true.sum()),
                reference_auprc=float(average_precision_score(y_true, reference_scores)),
                comparator_auprc=float(average_precision_score(y_true, comparator_scores)),
                delta_auprc=float(bootstrap["observed_diff"]),
                auprc_ci=(float(bootstrap["ci_low"]), float(bootstrap["ci_high"])),
                auprc_p=float(bootstrap["p_value"]),
                reference_auc=float(roc_auc_score(y_true, reference_scores)),
                comparator_auc=float(roc_auc_score(y_true, comparator_scores)),
                delta_auc=float(delong.get("auc_a", np.nan) - delong.get("auc_b", np.nan)),
                delong_p=float(delong["p_value"]),
            )
        )
    return results


def reference_interval(
    paper: int, reference: tuple[str, str], folds: list[int], seed: int = 0, n_boot: int = 5000
) -> dict:
    """Bootstrap interval for the proposed model's own AUPRC."""
    loaded = pooled_scores(paper, reference[0], reference[1], folds, seed)
    if loaded is None:
        return {}
    y_true, scores = loaded

    key = _fingerprint("interval", paper, reference, n_boot, seed, y_true, scores)

    def compute() -> dict:
        interval = bootstrap_ci(y_true, scores, average_precision_score,
                                n_boot=n_boot, seed=seed)
        return {
            "auprc": float(average_precision_score(y_true, scores)),
            "ci_low": float(interval.low),
            "ci_high": float(interval.high),
            "n_samples": int(len(y_true)),
            "n_positives": int(y_true.sum()),
        }

    return _cached(paper, key, compute)


def paired_arms(
    rounds: pd.DataFrame,
    reference: str,
    comparators: list[str],
    metric: str = "auprc",
    n_boot: int = 10_000,
    seed: int = 0,
) -> list[dict]:
    """Compare feedback arms on a per-round metric, paired by seed and round.

    The feedback study has no held-out test set to bootstrap over: each arm is a
    whole simulated deployment, and its outcome is a short series of rounds. So
    the pairing is done where it actually exists. Every arm replays the same
    stream, is cut into the same rounds, and is run under the same seeds, so the
    round of one arm and the same round of another are measurements of the same
    stretch of traffic under two different bookkeeping rules. Differencing
    within a (seed, round) cell removes the round-to-round variation in fraud
    volume, which is far larger than the effect being tested.

    Reported are the mean paired difference with a bootstrap interval over
    cells, and a Wilcoxon signed-rank p-value, which assumes nothing about the
    shape of the per-round differences.
    """
    from scipy import stats as scipy_stats

    if rounds.empty or metric not in rounds.columns:
        return []

    keys = ["seed", "round_index"]
    reference_rows = rounds[rounds["arm"] == reference].set_index(keys)[metric]
    if reference_rows.empty:
        return []

    generator = np.random.default_rng(seed)
    results = []
    for arm in comparators:
        if arm == reference:
            continue
        comparator_rows = rounds[rounds["arm"] == arm].set_index(keys)[metric]
        if comparator_rows.empty:
            continue

        aligned = pd.concat(
            {"reference": reference_rows, "comparator": comparator_rows}, axis=1
        ).dropna()
        if len(aligned) < 3:
            continue

        differences = (aligned["comparator"] - aligned["reference"]).to_numpy(dtype=float)
        draws = generator.integers(0, len(differences), size=(n_boot, len(differences)))
        means = differences[draws].mean(axis=1)

        # Wilcoxon is undefined when every difference is zero, which happens for
        # an arm compared against itself under a different name.
        if np.allclose(differences, 0):
            p_value = 1.0
        else:
            p_value = float(scipy_stats.wilcoxon(differences).pvalue)

        results.append({
            "reference": reference,
            "comparator": arm,
            "metric": metric,
            "n_pairs": int(len(differences)),
            "reference_mean": float(aligned["reference"].mean()),
            "comparator_mean": float(aligned["comparator"].mean()),
            "delta": float(differences.mean()),
            "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)),
            "p_value": p_value,
        })
    return results


def stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""
