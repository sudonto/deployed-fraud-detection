"""Experiment driver for Paper 1.

Runs are grouped into stages so the study can be built up incrementally, and
every completed run is written to its own JSON file. Re-running a stage skips
work that already has a result file, which makes the whole thing resumable.

Stages
------
baselines   Classical, ensemble and deep detectors, no resampling.
sampling    LightGBM head under every resampling operator, including TASR.
ablation    Component-by-component teardown of the proposed pipeline.
proposed    The full model, both heads, plus a seed-stability check.
leakage     The protocol ablation: correct vs. leaky resampling and splitting.

Usage
-----
    python -m src.paper1_fraud.run_experiments --stage all
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from src.common.io import PROJECT_ROOT, load_scores, safe_name, save_scores, write_json
from src.common.metrics import best_f1_threshold, evaluate
from src.common.seeds import set_seed
from src.common.splits import Split, leaky_random_split, rolling_origin_splits
from src.common.torch_utils import release_memory, run_resilient
from src.paper1_fraud.data import describe_split_drift, fit_scaler, load_fraud_arrays
from src.paper1_fraud.encoder import SCAEConfig
from src.paper1_fraud.pipeline import HybridConfig, HybridFraudDetector
from src.paper1_fraud.resampling import (
    TimeAwareStratifiedResampler,
    build_reference_sampler,
)

LOGGER = logging.getLogger("paper1")
RESULTS_DIR = PROJECT_ROOT / "results" / "paper1" / "raw_runs"

# Fold geometry chosen from the fraud timeline rather than by convention: the
# 492 frauds are spread very unevenly across the 48-hour window, and a 12% test
# window is the smallest that still leaves at least 34 frauds in every fold
# (scripts/inspect_fraud_timeline.py). A 10% window drops one fold to 22.
N_FOLDS = 5
TEST_SIZE = 0.12

SHALLOW_BASELINES = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
]
DEEP_BASELINES = ["mlp", "cnn1d", "bilstm", "autoencoder"]
SAMPLERS = [
    "none",
    "random_over",
    "random_under",
    "smote",
    "borderline_smote",
    "adasyn",
    "smote_tomek",
    "smote_enn",
    "tasr",
]


# --------------------------------------------------------------------------- #
def prepare_split(arrays, split: Split):
    """Scale using training rows only, then hand back the three partitions."""
    scaler = fit_scaler(arrays.X[split.train_idx])
    return {
        "X_train": scaler.transform(arrays.X[split.train_idx]).astype(np.float32),
        "y_train": arrays.y[split.train_idx],
        "t_train": arrays.time[split.train_idx],
        "X_val": scaler.transform(arrays.X[split.val_idx]).astype(np.float32),
        "y_val": arrays.y[split.val_idx],
        "X_test": scaler.transform(arrays.X[split.test_idx]).astype(np.float32),
        "y_test": arrays.y[split.test_idx],
    }


def result_path(stage: str, arm: str, fold: int, seed: int) -> Path:
    return RESULTS_DIR / f"{stage}__{safe_name(arm)}__fold{fold}__seed{seed}.json"


def already_done(stage: str, arm: str, fold: int, seed: int, force: bool) -> bool:
    """True only when both the metrics and the predictions behind them exist.

    Checking the metrics file alone would let a resumed run skip an arm that was
    recorded before per-transaction scores were being saved, leaving a result
    that no significance test can use.
    """
    if force:
        return False
    metrics_written = result_path(stage, arm, fold, seed).exists()
    return metrics_written and load_scores(1, stage, arm, fold, seed) is not None


def record(
    stage: str,
    arm: str,
    fold: int,
    seed: int,
    metrics: dict,
    extra: dict | None = None,
) -> dict:
    payload = {
        "paper": 1,
        "dataset": "ulb_creditcard",
        "stage": stage,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    return payload


def store(
    stage: str,
    arm: str,
    fold: int,
    seed: int,
    metrics: dict,
    extra: dict,
    y_true: np.ndarray,
    scores: np.ndarray,
) -> None:
    """Write the summary record and the raw predictions that back it.

    Also the point where one arm ends and the next begins, so it is where the
    GPU allocator is asked to give its cached blocks back.
    """
    write_json(
        result_path(stage, arm, fold, seed),
        record(stage, arm, fold, seed, metrics, extra),
    )
    save_scores(1, stage, arm, fold, seed, y_true, scores)
    release_memory()


# --------------------------------------------------------------------------- #
def run_baseline(name: str, data: dict, seed: int) -> tuple[dict, dict]:
    """Fit one baseline on unresampled data and score it on the test partition."""
    from src.paper1_fraud.baselines import (
        AutoencoderAnomaly,
        DeepBaseline,
        DeepConfig,
        build_shallow_baseline,
    )

    started = time.perf_counter()
    if name in DEEP_BASELINES:
        config = DeepConfig(seed=seed, max_epochs=40 if name == "bilstm" else 60)

        def fit_and_score():
            model = (
                AutoencoderAnomaly(config)
                if name == "autoencoder"
                else DeepBaseline(name, config)
            )
            model.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
            return model.predict_proba(data["X_val"]), model.predict_proba(data["X_test"])

        val_scores, test_scores = run_resilient(fit_and_score)
    else:
        model = build_shallow_baseline(name, seed=seed)
        model.fit(data["X_train"], data["y_train"])
        val_scores = model.predict_proba(data["X_val"])[:, 1]
        test_scores = model.predict_proba(data["X_test"])[:, 1]
    elapsed = time.perf_counter() - started

    threshold = best_f1_threshold(data["y_val"], val_scores)
    metrics = evaluate(data["y_test"], test_scores, threshold=threshold)
    return metrics, {"fit_seconds": elapsed, "threshold": threshold}, test_scores


def run_hybrid(config: HybridConfig, data: dict) -> tuple[dict, dict, np.ndarray]:
    def fit_and_score():
        detector = HybridFraudDetector(config)
        detector.fit(
            data["X_train"],
            data["y_train"],
            data["t_train"],
            data["X_val"],
            data["y_val"],
        )
        return detector, detector.score_test(data["X_test"], data["y_test"])

    model, metrics = run_resilient(fit_and_score)
    extra = {
        "timings": model.timings_,
        "resample_stats": {
            k: v for k, v in model.resample_stats_.items() if k != "blocks"
        },
        "threshold": model.threshold_,
        "config": {
            "sampler": config.sampler,
            "target_ratio": config.target_ratio,
            "use_encoder": config.use_encoder,
            "use_raw_features": config.use_raw_features,
            "contrastive_weight": config.scae.contrastive_weight,
            "head": config.head,
        },
    }
    return metrics, extra, model.test_scores_


# --------------------------------------------------------------------------- #
def stage_baselines(arrays, splits, args) -> None:
    names = SHALLOW_BASELINES + ([] if args.skip_deep else DEEP_BASELINES)
    for split in splits:
        data = prepare_split(arrays, split)
        for name in names:
            if already_done("baselines", name, split.fold, args.seed, args.force):
                continue
            set_seed(args.seed)
            LOGGER.info("baselines | %s | fold %d", name, split.fold)
            metrics, extra, scores = run_baseline(name, data, args.seed)
            store("baselines", name, split.fold, args.seed, metrics, extra,
                  data["y_test"], scores)
            LOGGER.info("  AUPRC=%.4f MCC=%.4f", metrics["auprc"], metrics["mcc"])


def stage_sampling(arrays, splits, args) -> None:
    for split in splits:
        data = prepare_split(arrays, split)
        for sampler in SAMPLERS:
            arm = f"lgbm_{sampler}"
            if already_done("sampling", arm, split.fold, args.seed, args.force):
                continue
            set_seed(args.seed)
            LOGGER.info("sampling | %s | fold %d", arm, split.fold)
            config = HybridConfig(
                sampler=sampler, use_encoder=False, head="lightgbm", seed=args.seed
            )
            metrics, extra, scores = run_hybrid(config, data)
            store("sampling", arm, split.fold, args.seed, metrics, extra,
                  data["y_test"], scores)
            LOGGER.info("  AUPRC=%.4f MCC=%.4f", metrics["auprc"], metrics["mcc"])


def ablation_arms(seed: int) -> dict[str, HybridConfig]:
    """Each arm removes exactly one ingredient from the full model."""
    full = lambda **kw: HybridConfig(  # noqa: E731
        sampler="tasr",
        use_encoder=True,
        use_raw_features=True,
        head="lightgbm",
        seed=seed,
        scae=SCAEConfig(seed=seed),
        **kw,
    )
    arms = {
        "full": full(),
        "no_tasr": HybridConfig(
            sampler="none", use_encoder=True, head="lightgbm", seed=seed,
            scae=SCAEConfig(seed=seed),
        ),
        "smote_instead_of_tasr": HybridConfig(
            sampler="smote", use_encoder=True, head="lightgbm", seed=seed,
            scae=SCAEConfig(seed=seed),
        ),
        "no_encoder": HybridConfig(
            sampler="tasr", use_encoder=False, head="lightgbm", seed=seed
        ),
        "no_contrastive": HybridConfig(
            sampler="tasr", use_encoder=True, head="lightgbm", seed=seed,
            scae=SCAEConfig(seed=seed, contrastive_weight=0.0),
        ),
        "no_reconstruction": HybridConfig(
            sampler="tasr", use_encoder=True, head="lightgbm", seed=seed,
            scae=SCAEConfig(seed=seed, reconstruction_weight=0.0),
        ),
        "embedding_only": HybridConfig(
            sampler="tasr", use_encoder=True, use_raw_features=False,
            head="lightgbm", seed=seed, scae=SCAEConfig(seed=seed),
        ),
        "no_cleaning": HybridConfig(
            sampler="tasr", cleaning=None, use_encoder=True, head="lightgbm",
            seed=seed, scae=SCAEConfig(seed=seed),
        ),
        "one_block": HybridConfig(
            sampler="tasr", n_blocks=1, use_encoder=True, head="lightgbm",
            seed=seed, scae=SCAEConfig(seed=seed),
        ),
    }
    return arms


def stage_ablation(arrays, splits, args) -> None:
    for split in splits:
        data = prepare_split(arrays, split)
        for arm, config in ablation_arms(args.seed).items():
            if already_done("ablation", arm, split.fold, args.seed, args.force):
                continue
            set_seed(args.seed)
            LOGGER.info("ablation | %s | fold %d", arm, split.fold)
            metrics, extra, scores = run_hybrid(config, data)
            store("ablation", arm, split.fold, args.seed, metrics, extra,
                  data["y_test"], scores)
            LOGGER.info("  AUPRC=%.4f MCC=%.4f", metrics["auprc"], metrics["mcc"])


def stage_proposed(arrays, splits, args) -> None:
    for split in splits:
        data = prepare_split(arrays, split)
        for head in ("lightgbm", "xgboost"):
            arm = f"proposed_{head}"
            if already_done("proposed", arm, split.fold, args.seed, args.force):
                continue
            set_seed(args.seed)
            LOGGER.info("proposed | %s | fold %d", arm, split.fold)
            config = HybridConfig(
                sampler="tasr", use_encoder=True, head=head, seed=args.seed,
                scae=SCAEConfig(seed=args.seed),
            )
            metrics, extra, scores = run_hybrid(config, data)
            store("proposed", arm, split.fold, args.seed, metrics, extra,
                  data["y_test"], scores)
            LOGGER.info("  AUPRC=%.4f MCC=%.4f", metrics["auprc"], metrics["mcc"])

    # Seed stability on the most recent fold: does the conclusion survive a
    # different random initialisation of the encoder?
    last = splits[-1]
    data = prepare_split(arrays, last)
    for seed in (1, 2, 3, 4):
        if already_done("proposed", "proposed_lightgbm_seedcheck", last.fold, seed, args.force):
            continue
        set_seed(seed)
        LOGGER.info("proposed | seed check | seed %d", seed)
        config = HybridConfig(
            sampler="tasr", use_encoder=True, head="lightgbm", seed=seed,
            scae=SCAEConfig(seed=seed),
        )
        metrics, extra, scores = run_hybrid(config, data)
        store("proposed", "proposed_lightgbm_seedcheck", last.fold, seed, metrics, extra,
              data["y_test"], scores)


def stage_leakage(arrays, args) -> None:
    """Quantify how much published performance comes from an unsound protocol.

    Four arms crossed over two factors: how the data is split (chronological vs
    random) and where the oversampling happens (inside the training fold vs on
    the whole dataset before splitting).
    """
    correct_split = rolling_origin_splits(arrays.time, N_FOLDS, TEST_SIZE)[-1]
    random_split = leaky_random_split(arrays.y, test_size=TEST_SIZE, seed=args.seed)

    for split_name, split in (("chronological", correct_split), ("random", random_split)):
        for resample_when in ("in_fold", "before_split"):
            arm = f"{split_name}__{resample_when}"
            if already_done("leakage", arm, 0, args.seed, args.force):
                continue
            set_seed(args.seed)
            LOGGER.info("leakage | %s", arm)

            if resample_when == "in_fold":
                data = prepare_split(arrays, split)
                config = HybridConfig(
                    sampler="smote", use_encoder=False, head="lightgbm", seed=args.seed
                )
                metrics, extra, scores = run_hybrid(config, data)
                y_test = data["y_test"]
            else:
                metrics, extra, y_test, scores = _run_leaky_before_split(
                    arrays, split, args.seed
                )

            extra["split_drift"] = describe_split_drift(
                arrays.time, split.train_idx, split.test_idx
            )
            store("leakage", arm, 0, args.seed, metrics, extra, y_test, scores)
            LOGGER.info("  AUPRC=%.4f MCC=%.4f", metrics["auprc"], metrics["mcc"])


def _run_leaky_before_split(
    arrays, split: Split, seed: int
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """Reproduce the common mistake: oversample the whole dataset, then split.

    Synthetic minority points interpolated from test-set frauds end up in the
    training set, so the model is partly memorising its own test data.
    """
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(arrays.X)  # also fitted on test rows, as is common
    X_all = scaler.transform(arrays.X).astype(np.float32)

    sampler = build_reference_sampler("smote", target_ratio=0.10, random_state=seed)
    X_res, y_res = sampler.fit_resample(X_all, arrays.y)
    X_res = np.asarray(X_res, dtype=np.float32)
    y_res = np.asarray(y_res).astype(np.int8)

    # Original rows keep their index, so the intended test rows can still be
    # identified inside the resampled matrix.
    n_original = len(arrays.y)
    test_mask = np.zeros(len(y_res), dtype=bool)
    test_mask[split.test_idx] = True
    val_mask = np.zeros(len(y_res), dtype=bool)
    val_mask[split.val_idx] = True
    train_mask = ~test_mask & ~val_mask
    train_mask[n_original:] = True  # every synthetic point goes into training

    model = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63, n_jobs=-1,
        verbose=-1, random_state=seed,
    )
    model.fit(X_res[train_mask], y_res[train_mask])
    val_scores = model.predict_proba(X_res[val_mask])[:, 1]
    test_scores = model.predict_proba(X_res[test_mask])[:, 1]

    threshold = best_f1_threshold(y_res[val_mask], val_scores)
    metrics = evaluate(y_res[test_mask], test_scores, threshold=threshold)
    return (
        metrics,
        {"note": "resampling and scaling applied before the split"},
        y_res[test_mask],
        test_scores,
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 1 experiments")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "baselines", "sampling", "ablation", "proposed", "leakage"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--force", action="store_true", help="recompute finished runs")
    parser.add_argument("--skip-deep", action="store_true", help="skip deep baselines")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    arrays = load_fraud_arrays()
    LOGGER.info(
        "ULB loaded: %d rows, %d features, %d frauds (%.4f%%)",
        len(arrays.y), arrays.X.shape[1], int(arrays.y.sum()), 100 * arrays.y.mean(),
    )

    splits = rolling_origin_splits(arrays.time, args.folds, TEST_SIZE)
    for split in splits:
        drift = describe_split_drift(arrays.time, split.train_idx, split.test_idx)
        LOGGER.info(
            "fold %d: train %d / val %d / test %d, chronological=%s",
            split.fold, *split.sizes().values(), drift["chronological"],
        )

    stage = args.stage
    if stage in ("all", "baselines"):
        stage_baselines(arrays, splits, args)
    if stage in ("all", "sampling"):
        stage_sampling(arrays, splits, args)
    if stage in ("all", "ablation"):
        stage_ablation(arrays, splits, args)
    if stage in ("all", "proposed"):
        stage_proposed(arrays, splits, args)
    if stage in ("all", "leakage"):
        stage_leakage(arrays, args)

    LOGGER.info("done; %d result files", len(list(RESULTS_DIR.glob("*.json"))))


if __name__ == "__main__":
    main()
