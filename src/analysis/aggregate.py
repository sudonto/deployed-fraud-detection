"""Read raw run files into tidy frames and summarise them across folds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import PROJECT_ROOT

RESULTS_ROOT = PROJECT_ROOT / "results"

# How each metric should be rendered, and which direction is better.
METRIC_LABELS = {
    "auprc": ("AUPRC", 4, "max"),
    "roc_auc": ("ROC-AUC", 4, "max"),
    "mcc": ("MCC", 4, "max"),
    "f1": ("F1", 4, "max"),
    "precision": ("Precision", 4, "max"),
    "recall": ("Recall", 4, "max"),
    "ks": ("KS", 4, "max"),
    "g_mean": ("G-mean", 4, "max"),
    "brier": ("Brier", 4, "min"),
    "balanced_accuracy": ("Bal. Acc.", 4, "max"),
    "precision_at_k": ("P@k", 4, "max"),
    "recall_at_precision_90": ("R@P90", 4, "max"),
    "average_precision_gain": ("APG", 4, "max"),
    # Feedback-loop study. Frauds caught is the operational bottom line, so it
    # leads; the rest describe how the pool and the ranking got there.
    "catch_rate": ("Fraud caught (\\%)", 1, "max"),
    "frauds_caught": ("Frauds caught", 0, "max"),
    "review_precision": ("Review precision", 4, "max"),
    "auprc_mean": ("AUPRC (mean)", 4, "max"),
    "auprc_final": ("AUPRC (final round)", 4, "max"),
    "auprc_drift": ("AUPRC drift", 4, "max"),
    "roc_auc_mean": ("ROC-AUC (mean)", 4, "max"),
    "recall_at_budget_mean": ("Recall@budget", 4, "max"),
    "pool_error_rate": ("Pool label error", 4, "min"),
    "capacity_ratio": ("Reviews per fraud", 2, "max"),
}

PAPER1_HEADLINE = ["auprc", "mcc", "f1", "precision", "recall", "roc_auc"]
PAPER2_HEADLINE = ["roc_auc", "ks", "f1", "g_mean", "brier"]


def load_runs(paper: int, stage: str | None = None) -> pd.DataFrame:
    """Flatten every result file for a paper into one row per run."""
    directory = RESULTS_ROOT / f"paper{paper}" / "raw_runs"
    if not directory.exists():
        return pd.DataFrame()

    rows = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if stage is not None and payload.get("stage") != stage:
            continue

        row = {
            "dataset": payload.get("dataset"),
            "stage": payload.get("stage"),
            "arm": payload.get("arm"),
            "fold": payload.get("fold"),
            "seed": payload.get("seed"),
            "fit_seconds": payload.get("fit_seconds"),
            "source_file": path.name,
        }
        row.update(payload.get("metrics", {}))
        for key in ("epochs_run", "best_val_auc", "n_synthetic", "n_removed", "view"):
            if key in payload:
                row[key] = payload[key]
        rows.append(row)

    return pd.DataFrame(rows)


def load_feedback(stage: str = "main", directory: Path | None = None) -> pd.DataFrame:
    """One row per feedback simulation, with the arm's configuration attached.

    Kept separate from ``load_runs`` because a feedback run is a whole simulated
    deployment rather than a cross-validation fold: it is identified by its
    review budget and censoring width as much as by its arm, and those live in
    the config block rather than in the metrics.
    """
    directory = directory or RESULTS_ROOT / "paper1" / "raw_runs"
    if not directory.exists():
        return pd.DataFrame()

    rows = []
    for path in sorted(directory.glob(f"feedback_{stage}__*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        config = payload.get("config", {})
        row = {
            "dataset": payload.get("dataset"),
            "stage": payload.get("stage"),
            "arm": payload.get("arm"),
            "seed": payload.get("seed"),
            "budget_fraction": config.get("budget_fraction"),
            "censor_width": config.get("censor_width"),
            "exploration": config.get("exploration"),
            "review_policy": config.get("review_policy"),
            "pool_rule": config.get("pool_rule"),
            "n_rounds": config.get("n_rounds"),
            "detector": config.get("detector"),
            "detector_tag": config.get("detector_tag"),
            "runtime_s": payload.get("runtime_s"),
            "source_file": path.name,
        }
        row.update(payload.get("metrics", {}))
        # The share of the final pool carrying a label the bank got wrong. This
        # is the mechanism the paper argues about, and it is only interpretable
        # as a rate; the raw count scales with how many rows the rule keeps.
        size = row.get("pool_size_final") or 0
        row["pool_error_rate"] = (row.get("pool_mislabelled_final", 0) / size) if size else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def load_feedback_rounds(stage: str = "main", directory: Path | None = None) -> pd.DataFrame:
    """One row per round per simulation, for the degradation curves."""
    directory = directory or RESULTS_ROOT / "paper1" / "raw_runs"
    if not directory.exists():
        return pd.DataFrame()

    rows = []
    for path in sorted(directory.glob(f"feedback_{stage}__*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        config = payload.get("config", {})
        for record in payload.get("rounds", []):
            row = dict(record)
            row.update({
                "dataset": payload.get("dataset"),
                "arm": payload.get("arm"),
                "seed": payload.get("seed"),
                "budget_fraction": config.get("budget_fraction"),
                "censor_width": config.get("censor_width"),
            })
            size = row.get("pool_size") or 0
            row["pool_error_rate"] = (row.get("pool_mislabelled", 0) / size) if size else np.nan
            rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        # Cumulative frauds caught is the quantity a fraud team actually cares
        # about, and it has to be accumulated within a single simulation.
        keys = ["dataset", "arm", "seed", "budget_fraction"]
        frame = frame.sort_values(keys + ["round_index"])
        frame["cumulative_caught"] = frame.groupby(keys)["frauds_caught"].cumsum()
        frame["cumulative_frauds"] = frame.groupby(keys)["n_frauds"].cumsum()
    return frame


def summarise(
    frame: pd.DataFrame,
    metrics: list[str],
    group: list[str] | None = None,
) -> pd.DataFrame:
    """Mean and standard deviation of each metric across folds and seeds."""
    if frame.empty:
        return pd.DataFrame()
    group = group or ["arm"]
    available = [m for m in metrics if m in frame.columns]

    aggregated = frame.groupby(group)[available].agg(["mean", "std", "count"])
    aggregated.columns = [f"{metric}_{statistic}" for metric, statistic in aggregated.columns]
    return aggregated.reset_index()


def format_mean_std(mean: float, std: float, decimals: int = 4, bold: bool = False) -> str:
    """Render "0.8123 ± 0.0210" for LaTeX, tolerating a missing deviation."""
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    body = f"{mean:.{decimals}f}"
    if std is not None and not (isinstance(std, float) and np.isnan(std)):
        body += f" $\\pm$ {std:.{decimals}f}"
    return f"\\textbf{{{body}}}" if bold else body


def best_arm(summary: pd.DataFrame, metric: str) -> str | None:
    """Which arm wins on a metric, respecting whether lower or higher is better."""
    column = f"{metric}_mean"
    if summary.empty or column not in summary.columns:
        return None
    direction = METRIC_LABELS.get(metric, (metric, 4, "max"))[2]
    index = summary[column].idxmin() if direction == "min" else summary[column].idxmax()
    if pd.isna(index):
        return None
    return summary.loc[index, "arm"]


def per_fold_matrix(frame: pd.DataFrame, metric: str, arms: list[str] | None = None) -> dict:
    """Metric values per arm, aligned by fold, for the paired statistical tests."""
    if frame.empty or metric not in frame.columns:
        return {}
    arms = arms or sorted(frame["arm"].dropna().unique())
    matrix = {}
    for arm in arms:
        subset = frame[frame["arm"] == arm].sort_values(["seed", "fold"])
        values = subset[metric].to_numpy(dtype=float)
        if len(values):
            matrix[arm] = values
    if not matrix:
        return {}
    shortest = min(len(v) for v in matrix.values())
    return {arm: values[:shortest].tolist() for arm, values in matrix.items()}


def pretty_arm(arm: str) -> str:
    """Human-readable model names for the tables."""
    replacements = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "knn": "k-NN",
        "svm": "SVM (RBF)",
        "mlp": "MLP",
        "cnn1d": "1D-CNN",
        "bilstm": "BiLSTM",
        "autoencoder": "Autoencoder (anomaly)",
        "tabnet_reference": "TabNet (reference impl.)",
        "tabnet_vanilla": "TabNet (vanilla)",
        "gma_full": "GMA-TabNet (proposed)",
        "gma_single_head": "GMA-TabNet, single head",
        "gma_fixed_temperature": "GMA-TabNet, fixed temperature",
        "gma_no_interaction_gate": "GMA-TabNet, no interaction gate",
        "gma_no_focal": "GMA-TabNet, no focal loss",
        "gma_pretrained": "GMA-TabNet, self-supervised pretraining",
        "stack_tabnet_vanilla": "TabNet-Stacking (published design)",
        "stack_gma": "GMA-TabNet-Stacking (proposed)",
        "stack_gma_nested": "GMA-TabNet-Stacking (nested extractor)",
        "stack_no_extractor": "Stacking without extractor",
        "proposed_lightgbm": "TASR + SCAE + LightGBM (proposed)",
        "proposed_xgboost": "TASR + SCAE + XGBoost (proposed)",
        "full": "Full pipeline",
        "no_contrastive": "without contrastive term",
        "no_reconstruction": "without reconstruction term",
        "no_encoder": "without encoder (TASR only)",
        "no_tasr": "without TASR (SCAE only)",
        "embedding_only": "embedding only, raw features dropped",
        "one_block": "single encoder block",
        "no_cleaning": "without Tomek/ENN cleaning",
        "smote_instead_of_tasr": "SMOTE in place of TASR",
        "random": "Random search",
        "tpe": "Bayesian (TPE)",
        "pso": "Particle swarm",
        "ga": "Genetic algorithm",
        "oracle_full_labels": "Oracle (all labels revealed)",
        "naive_assume_negative": "Deployed practice (unreviewed $\\rightarrow$ legitimate)",
        "reviewed_only": "Reviewed rows only",
        "explore_banded": "Banded exploration",
        "explore_uniform": "Uniform exploration",
        "ips": "Inverse propensity weighting",
        "ips_augmented": "IPS-augmented pool",
        "censor": "Hard censoring",
        "soft_censor": "Soft censoring",
        "censor_ips": "Hard censoring + IPS",
        "explore_censor": "Banded exploration + hard censoring",
        "explore_soft_censor": "Banded exploration + soft censoring",
    }
    if arm in replacements:
        return replacements[arm]
    if arm.startswith("lgbm_"):
        sampler = arm.removeprefix("lgbm_")
        names = {
            "none": "No resampling",
            "smote": "SMOTE",
            "adasyn": "ADASYN",
            "borderline_smote": "Borderline-SMOTE",
            "smote_tomek": "SMOTE-Tomek",
            "smote_enn": "SMOTE-ENN",
            "random_under": "Random undersampling",
            "tasr": "TASR (proposed)",
        }
        return names.get(sampler, sampler.replace("_", " ").title())
    return arm.replace("_", " ")


DATASET_LABELS = {
    "ulb_creditcard": "ULB Credit Card",
    "baf_base": "BAF (base)",
    "baf_variant3": "BAF (variant III)",
    "ieee_cis": "IEEE-CIS",
    "uci_dccc": "UCI Taiwan DCCC",
    "german_credit": "German Credit",
    "give_me_some_credit": "Give Me Some Credit",
}


def dataset_label(key: str) -> str:
    return DATASET_LABELS.get(key, key.replace("_", " ").title())
