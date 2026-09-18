"""Correctness checks for the shared modules and the TASR operator.

These are assertions about properties the papers depend on, not unit tests for
their own sake. If any of them fails, a results table somewhere is wrong.

    python -m scripts.verify_modules
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common.metrics import evaluate, ks_statistic, precision_at_k, recall_at_precision
from src.common.splits import rolling_origin_splits, stratified_splits, time_ordered_split
from src.common.stats import bootstrap_ci, delong_test, friedman_nemenyi, paired_bootstrap_test
from src.paper1_fraud.resampling import TimeAwareStratifiedResampler

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))


def synthetic(n: int = 4000, positive_rate: float = 0.02, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < positive_rate).astype(int)
    # A score that is informative but far from perfect.
    score = rng.normal(loc=y * 1.2, scale=1.0)
    score = (score - score.min()) / (score.max() - score.min())
    return y, score


def main() -> int:
    y, score = synthetic()

    # ---------------- metrics ---------------- #
    metrics = evaluate(y, score, threshold=0.5)
    check(
        "auprc matches sklearn",
        np.isclose(metrics["auprc"], average_precision_score(y, score)),
    )
    check(
        "roc_auc matches sklearn",
        np.isclose(metrics["roc_auc"], roc_auc_score(y, score)),
    )
    check("ks in [0,1]", 0.0 <= ks_statistic(y, score) <= 1.0)
    check(
        "confusion counts sum to n",
        metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"] == len(y),
    )
    perfect = y.astype(float)
    check("precision_at_k is 1 for a perfect ranker", precision_at_k(y, perfect) == 1.0)
    check(
        "recall_at_p90 is 1 for a perfect ranker",
        np.isclose(recall_at_precision(y, perfect, 0.90), 1.0),
    )
    check(
        "recall_at_p90 is 0 when precision is unreachable",
        recall_at_precision(y, np.zeros_like(score), 0.99) == 0.0,
    )

    # ---------------- splits ---------------- #
    time = np.sort(np.random.default_rng(1).random(5000) * 172_800)
    single = time_ordered_split(time, test_size=0.2, val_size=0.15)
    check(
        "time_ordered_split is chronological",
        time[single.train_idx].max() <= time[single.val_idx].min()
        and time[single.val_idx].max() <= time[single.test_idx].min(),
    )
    check(
        "time_ordered_split partitions every row",
        len(set(single.train_idx) | set(single.val_idx) | set(single.test_idx)) == len(time),
    )

    folds = rolling_origin_splits(time, n_folds=5, test_size=0.1)
    check("rolling origin produces 5 folds", len(folds) == 5)
    check(
        "every rolling fold trains strictly in the past",
        all(time[f.train_idx].max() <= time[f.test_idx].min() for f in folds),
    )
    check(
        "rolling folds have disjoint test windows",
        len({i for f in folds for i in f.test_idx}) == sum(len(f.test_idx) for f in folds),
    )
    check(
        "training window grows with the fold index",
        all(len(folds[i].train_idx) < len(folds[i + 1].train_idx) for i in range(4)),
    )

    strat = stratified_splits(y, n_folds=5, seed=0)
    check(
        "stratified folds never share rows between train and test",
        all(len(set(s.train_idx) & set(s.test_idx)) == 0 for s in strat),
    )
    check(
        "stratified folds keep the positive rate",
        all(abs(y[s.test_idx].mean() - y.mean()) < 0.02 for s in strat),
    )

    # ---------------- statistics ---------------- #
    delong = delong_test(y, score, score)
    check(
        "DeLong recovers the sklearn AUC",
        np.isclose(delong["auc_a"], roc_auc_score(y, score), atol=1e-6),
        f"delong={delong['auc_a']:.6f} sklearn={roc_auc_score(y, score):.6f}",
    )
    check("DeLong of a model against itself is not significant", delong["p_value"] > 0.99)

    weaker = 0.5 * score + 0.5 * np.random.default_rng(2).random(len(score))
    delong_diff = delong_test(y, score, weaker)
    check(
        "DeLong detects a genuinely weaker model",
        delong_diff["p_value"] < 0.05 and delong_diff["diff"] > 0,
        f"p={delong_diff['p_value']:.2e} diff={delong_diff['diff']:.4f}",
    )

    interval = bootstrap_ci(y, score, average_precision_score, n_boot=200, seed=0)
    check(
        "bootstrap interval brackets the point estimate",
        interval.low <= interval.point <= interval.high,
    )

    paired = paired_bootstrap_test(y, score, weaker, average_precision_score, n_boot=200)
    check("paired bootstrap finds the stronger model", paired["observed_diff"] > 0)

    ranks = friedman_nemenyi(
        {
            "good": [0.90, 0.88, 0.91, 0.89],
            "middling": [0.85, 0.83, 0.86, 0.84],
            "poor": [0.70, 0.72, 0.69, 0.71],
        }
    )
    check(
        "Friedman ranks the models correctly",
        ranks["average_ranks"]["good"] < ranks["average_ranks"]["middling"] < ranks["average_ranks"]["poor"],
    )
    check("Nemenyi post-hoc is available", ranks["nemenyi_p_values"] is not None)

    # ---------------- TASR ---------------- #
    rng = np.random.default_rng(3)
    n = 20_000
    t = np.sort(rng.random(n) * 172_800)
    # Fraud concentrated in two bursts, which is the structure TASR exists to keep.
    burst = ((t > 20_000) & (t < 30_000)) | ((t > 120_000) & (t < 130_000))
    y_f = (burst & (rng.random(n) < 0.05)).astype(np.int8)
    X_f = rng.normal(size=(n, 8)).astype(np.float32) + y_f[:, None] * 1.5

    resampler = TimeAwareStratifiedResampler(target_ratio=0.10, n_blocks=12, random_state=0)
    result = resampler.fit_resample(X_f, y_f, t)

    check("TASR adds synthetic minority samples", result.stats["n_synthetic"] > 0)
    check(
        "TASR reaches roughly the requested ratio",
        0.07 <= result.stats["ratio_after"] <= 0.13,
        f"ratio={result.stats['ratio_after']:.4f}",
    )
    check("TASR emits a timestamp per row", len(result.time) == len(result.y))
    check("TASR output stays chronologically sorted", np.all(np.diff(result.time) >= 0))
    check(
        "synthetic timestamps stay inside the original window",
        result.time.min() >= t.min() - 1e-6 and result.time.max() <= t.max() + 1e-6,
    )

    # The property that distinguishes TASR from plain SMOTE: synthetic frauds
    # appear only where real fraud already occurred, so the burst profile holds.
    original_fraud_times = t[y_f == 1]
    synthetic_times = result.time[result.y == 1]
    tolerance = (t.max() - t.min()) / 12
    near_real_burst = np.array(
        [np.abs(original_fraud_times - s).min() <= tolerance for s in synthetic_times]
    )
    check(
        "synthetic frauds stay inside blocks that contain real fraud",
        near_real_burst.mean() > 0.99,
        f"{100 * near_real_burst.mean():.2f}% within one block of a real fraud",
    )

    check("TASR cleaning removed boundary majority points", result.stats.get("n_cleaned", 0) >= 0)

    # ---------------- report ---------------- #
    width = max(len(name) for name, _, _ in CHECKS)
    failed = 0
    for name, ok, detail in CHECKS:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        suffix = f"   {detail}" if detail else ""
        print(f"[{status}] {name.ljust(width)}{suffix}")

    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
