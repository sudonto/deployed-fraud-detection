"""Fast end-to-end exercise of the Paper 1 pipeline on a subsample.

Runs every code path the full study uses - each baseline family, each resampler,
the encoder, and the hybrid head - with tiny budgets, so that a crash surfaces in
a minute instead of two hours into the real run.

    python -m scripts.smoke_paper1
"""

from __future__ import annotations

import sys
import time

import numpy as np

from src.common.seeds import set_seed
from src.common.splits import rolling_origin_splits
from src.paper1_fraud.baselines import AutoencoderAnomaly, DeepBaseline, DeepConfig, build_shallow_baseline
from src.paper1_fraud.data import describe_split_drift, load_fraud_arrays
from src.paper1_fraud.encoder import SCAEConfig
from src.paper1_fraud.pipeline import HybridConfig, HybridFraudDetector
from src.paper1_fraud.run_experiments import prepare_split
from src.common.metrics import best_f1_threshold, evaluate

SUBSAMPLE = 40_000


def subsample(arrays, n: int, seed: int = 0):
    """Keep a contiguous recent window plus every fraud, so positives survive."""
    rng = np.random.default_rng(seed)
    order = np.argsort(arrays.time, kind="stable")
    tail = order[-n:]
    frauds = np.flatnonzero(arrays.y == 1)
    keep = np.union1d(tail, frauds)
    rng.shuffle(keep)
    keep = keep[np.argsort(arrays.time[keep], kind="stable")]

    arrays.X, arrays.y, arrays.time = arrays.X[keep], arrays.y[keep], arrays.time[keep]
    return arrays


def main() -> int:
    set_seed(0)
    arrays = subsample(load_fraud_arrays(), SUBSAMPLE)
    print(f"subsample: {len(arrays.y)} rows, {int(arrays.y.sum())} frauds "
          f"({100 * arrays.y.mean():.3f}%)")

    split = rolling_origin_splits(arrays.time, n_folds=3, test_size=0.15)[-1]
    print("split sizes:", split.sizes())
    print("drift:", describe_split_drift(arrays.time, split.train_idx, split.test_idx))
    data = prepare_split(arrays, split)

    failures = []

    def attempt(name: str, fn):
        started = time.perf_counter()
        try:
            auprc = fn()
            print(f"  OK   {name:<34} AUPRC={auprc:.4f}  ({time.perf_counter() - started:.1f}s)")
        except Exception as exc:  # noqa: BLE001 - the point is to collect every failure
            failures.append((name, repr(exc)))
            print(f"  FAIL {name:<34} {exc!r}")

    def shallow(name):
        def run():
            model = build_shallow_baseline(name, seed=0)
            model.fit(data["X_train"], data["y_train"])
            val = model.predict_proba(data["X_val"])[:, 1]
            test = model.predict_proba(data["X_test"])[:, 1]
            return evaluate(data["y_test"], test, best_f1_threshold(data["y_val"], val))["auprc"]
        return run

    def deep(name):
        def run():
            config = DeepConfig(seed=0, max_epochs=3, patience=2)
            model = AutoencoderAnomaly(config) if name == "autoencoder" else DeepBaseline(name, config)
            model.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
            val = model.predict_proba(data["X_val"])
            test = model.predict_proba(data["X_test"])
            return evaluate(data["y_test"], test, best_f1_threshold(data["y_val"], val))["auprc"]
        return run

    def hybrid(config):
        def run():
            model = HybridFraudDetector(config)
            model.fit(data["X_train"], data["y_train"], data["t_train"], data["X_val"], data["y_val"])
            return model.score_test(data["X_test"], data["y_test"])["auprc"]
        return run

    print("\nshallow baselines")
    for name in ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]:
        attempt(name, shallow(name))

    print("\ndeep baselines")
    for name in ["mlp", "cnn1d", "bilstm", "autoencoder"]:
        attempt(name, deep(name))

    print("\nresamplers (LightGBM head, no encoder)")
    for sampler in ["none", "random_over", "random_under", "smote", "borderline_smote",
                    "adasyn", "smote_tomek", "smote_enn", "tasr"]:
        attempt(sampler, hybrid(HybridConfig(sampler=sampler, use_encoder=False, seed=0)))

    print("\nhybrid variants")
    quick = SCAEConfig(seed=0, max_epochs=3, patience=2)
    attempt("tasr+scae+lgbm", hybrid(HybridConfig(sampler="tasr", use_encoder=True, scae=quick, seed=0)))
    attempt("tasr+scae+xgb", hybrid(HybridConfig(sampler="tasr", use_encoder=True, scae=quick, head="xgboost", seed=0)))
    attempt("tasr+ae+lgbm", hybrid(HybridConfig(
        sampler="tasr", use_encoder=True,
        scae=SCAEConfig(seed=0, max_epochs=3, patience=2, contrastive_weight=0.0), seed=0)))
    attempt("tasr+scae emb-only", hybrid(HybridConfig(
        sampler="tasr", use_encoder=True, use_raw_features=False, scae=quick, seed=0)))

    print()
    if failures:
        print(f"{len(failures)} failure(s):")
        for name, error in failures:
            print(f"  {name}: {error}")
        return 1
    print("all smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
