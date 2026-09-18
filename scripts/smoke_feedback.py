"""Quick check that the feedback simulation behaves the way the design assumes.

Run before committing the sweep to the server. It uses a short stream and few
rounds, so the numbers are noisy, but the structural claims it checks are the
ones that would invalidate the study if they were wrong: that the review policy
spends exactly its budget, that propensities are positive wherever inverse
weighting needs them to be, that the naive arm really does poison its own pool,
and that the oracle really does not.
"""

from __future__ import annotations

import numpy as np

from src.paper1_fraud.data import load_fraud_arrays
from src.paper1_fraud.feedback import (
    FeedbackConfig,
    _propensities,
    _select_reviews,
    run_simulation,
    summarise,
)

PASS, FAIL = "[PASS]", "[FAIL]"
results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append(condition)
    print(f"{PASS if condition else FAIL} {name:<58} {detail}")


def main() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(10_000)
    budget = 100

    for policy, rule in [
        ("topk", "assume_negative"),
        ("uniform", "assume_negative"),
        ("banded", "assume_negative"),
        ("topk", "ips"),
        ("banded", "ips"),
    ]:
        config = FeedbackConfig(review_policy=policy, pool_rule=rule, exploration=0.1)
        reviewed, propensity = _select_reviews(scores, budget, config, rng)
        label = f"{policy}+{rule}"
        check(f"{label} spends its budget", len(reviewed) == budget, f"n={len(reviewed)}")
        check(
            f"{label} propensities sum to the budget",
            abs(propensity.sum() - budget) < 1.0,
            f"sum={propensity.sum():.2f}",
        )
        if rule == "ips":
            check(
                f"{label} leaves no zero propensity",
                bool((propensity > 0).all()),
                f"min={propensity.min():.2e}",
            )

    config = FeedbackConfig(review_policy="banded", pool_rule="assume_negative", exploration=0.1)
    propensity = _propensities(scores, budget, config)
    order = np.argsort(-scores, kind="stable")
    band_end = int(round(budget * (1 - 0.1))) + int(round(budget * config.band_width))
    check(
        "banded exploration stays inside the band",
        bool((propensity[order[band_end + 1:]] == 0).all()),
        f"band ends at rank {band_end}",
    )

    arrays = load_fraud_arrays()
    order = np.argsort(arrays.time, kind="stable")
    X, y = arrays.X[order], arrays.y[order]

    print()
    print(f"stream: {len(y)} transactions, {int(y.sum())} frauds")
    print()

    n_rounds = 5
    edges = np.linspace(0, len(y), n_rounds + 1, dtype=int)
    round_index = np.empty(len(y), dtype=int)
    for index in range(n_rounds):
        round_index[edges[index]:edges[index + 1]] = index

    summaries = {}
    for arm, config in [
        ("oracle", FeedbackConfig("topk", "full_labels")),
        ("naive", FeedbackConfig("topk", "assume_negative")),
        ("proposed", FeedbackConfig("banded", "ips", exploration=0.1)),
    ]:
        outcomes = run_simulation(
            X, y, round_index, config,
            warmup_rounds=1, budget_fraction=0.01, seed=0,
        )
        summary = summarise(outcomes)
        summaries[arm] = summary
        print(
            f"  {arm:<10} caught {summary['frauds_caught']:>3}/{summary['frauds_present']:<3}"
            f"  AUPRC {summary['auprc_first']:.4f} -> {summary['auprc_final']:.4f}"
            f"  pool errors {summary['pool_mislabelled_final']}"
        )

    print()
    check(
        "the oracle pool holds no wrong labels",
        summaries["oracle"]["pool_mislabelled_final"] == 0,
        f"errors={summaries['oracle']['pool_mislabelled_final']}",
    )
    check(
        "the naive pool accumulates wrong labels",
        summaries["naive"]["pool_mislabelled_final"] > 0,
        f"errors={summaries['naive']['pool_mislabelled_final']}",
    )
    check(
        "the proposed pool fabricates no labels",
        summaries["proposed"]["pool_mislabelled_final"] == 0,
        f"errors={summaries['proposed']['pool_mislabelled_final']}",
    )
    check(
        "every arm reviews the same number of transactions",
        len({s["reviews_spent"] for s in summaries.values()}) == 1,
        f"spent={ {k: v['reviews_spent'] for k, v in summaries.items()} }",
    )

    print()
    print(f"{sum(results)}/{len(results)} checks passed")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
