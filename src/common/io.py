"""Dataset acquisition and caching.

Every dataset is pulled from OpenML so the whole study is reproducible without
a Kaggle account. Each one has been used in prior published work, which is what
makes the baseline comparisons in both papers meaningful.

After the first download the frames are cached as Parquet under ``data/raw`` and
subsequent loads never touch the network.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass
class DatasetSpec:
    """Everything needed to fetch one dataset and normalise its label."""

    key: str
    openml_id: int
    positive_label: str
    description: str
    # Expected shape and imbalance, verified on load so a silently changed
    # upstream version cannot corrupt the results.
    expected_rows: int
    expected_features: int
    expected_minority_pct: float
    time_column: str | None = None
    drop_columns: tuple[str, ...] = ()


DATASETS: dict[str, DatasetSpec] = {
    "ulb_creditcard": DatasetSpec(
        key="ulb_creditcard",
        openml_id=42175,
        positive_label="1",
        description=(
            "ULB / Worldline European credit card transactions, September 2013. "
            "Features V1-V28 are PCA projections of the withheld raw attributes."
        ),
        expected_rows=284_807,
        expected_features=30,
        expected_minority_pct=0.1727,
        time_column="Time",
    ),
    "baf_base": DatasetSpec(
        key="baf_base",
        openml_id=46793,
        positive_label="1",
        description=(
            "Bank Account Fraud, base variant (Jesus et al., NeurIPS 2022 Datasets "
            "and Benchmarks). One million online account-opening applications over "
            "eight months, generated from a real Feedzai portfolio under a privacy "
            "preserving generative model. The month column is the temporal axis and "
            "carries a documented distribution shift."
        ),
        expected_rows=1_000_000,
        expected_features=31,
        expected_minority_pct=1.1029,
        time_column="month",
    ),
    "baf_variant3": DatasetSpec(
        key="baf_variant3",
        openml_id=46796,
        positive_label="1",
        description=(
            "Bank Account Fraud, variant III. Same generator as the base variant "
            "with a higher prevalence of fraud in one applicant group, used here as "
            "a robustness check rather than a second headline dataset."
        ),
        expected_rows=1_000_000,
        expected_features=31,
        expected_minority_pct=1.1030,
        time_column="month",
        # The OpenML copy of this variant carries two unnamed extra columns that
        # the base variant does not. Keeping them would mean the robustness
        # check ran on a different feature space from the headline result, so
        # the comparison would confound the shift being tested with a change in
        # what the model gets to see.
        drop_columns=("x1", "x2"),
    ),
    "ieee_cis": DatasetSpec(
        key="ieee_cis",
        openml_id=46858,
        positive_label="1",
        description=(
            "IEEE-CIS Fraud Detection (Vesta Corporation, 2019). 590k e-commerce "
            "transactions over roughly six months, with TransactionDT giving the "
            "seconds elapsed from a fixed reference. The most widely used large "
            "public card-fraud benchmark after ULB."
        ),
        expected_rows=590_540,
        expected_features=455,
        expected_minority_pct=3.4990,
        time_column="TransactionDT",
    ),
    "uci_dccc": DatasetSpec(
        key="uci_dccc",
        openml_id=42477,
        positive_label="1",
        description=(
            "UCI Default of Credit Card Clients, Taiwan, April-September 2005 "
            "(Yeh & Lien, 2009). 30k clients, next-month default as the label."
        ),
        expected_rows=30_000,
        expected_features=23,
        expected_minority_pct=22.12,
    ),
    "german_credit": DatasetSpec(
        key="german_credit",
        openml_id=31,
        positive_label="bad",
        description="Statlog German Credit Data (UCI). 1000 applicants, 20 attributes.",
        expected_rows=1_000,
        expected_features=20,
        expected_minority_pct=30.0,
    ),
    "give_me_some_credit": DatasetSpec(
        key="give_me_some_credit",
        openml_id=46929,
        positive_label="Yes",
        description=(
            "Kaggle Give Me Some Credit. 150k borrowers, serious delinquency "
            "within two years as the label."
        ),
        expected_rows=150_000,
        expected_features=10,
        expected_minority_pct=6.68,
    ),
}


@dataclass
class Dataset:
    """A loaded, label-normalised dataset."""

    key: str
    X: pd.DataFrame
    y: np.ndarray
    categorical_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    time_column: str | None = None
    description: str = ""

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def minority_pct(self) -> float:
        return float(self.y.mean() * 100.0)

    @property
    def imbalance_ratio(self) -> float:
        """Majority-to-minority ratio, the usual way this is reported."""
        pos = int(self.y.sum())
        return float((len(self.y) - pos) / max(pos, 1))

    def summary(self) -> dict:
        return {
            "key": self.key,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_positive": int(self.y.sum()),
            "minority_pct": round(self.minority_pct, 4),
            "imbalance_ratio": round(self.imbalance_ratio, 1),
            "n_categorical": len(self.categorical_columns),
            "n_numeric": len(self.numeric_columns),
        }


def _cache_paths(key: str) -> tuple[Path, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / f"{key}.parquet", RAW_DIR / f"{key}.meta.json"


def _download(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    from sklearn.datasets import fetch_openml

    LOGGER.info("Downloading %s from OpenML (id=%s)", spec.key, spec.openml_id)
    bunch = fetch_openml(
        data_id=spec.openml_id,
        as_frame=True,
        parser="auto",
    )
    X = bunch.data.copy()
    y = bunch.target.copy()

    if spec.drop_columns:
        X = X.drop(columns=[c for c in spec.drop_columns if c in X.columns])

    # OpenML sometimes carries a redundant row-id column.
    for candidate in ("ID", "id", "Id", "Unnamed: 0", "index"):
        if candidate in X.columns and X[candidate].is_unique:
            X = X.drop(columns=[candidate])

    return X, y


def _binarise(y: pd.Series, positive_label: str) -> np.ndarray:
    """Map the target to {0, 1} with 1 = the event of interest (fraud/default)."""
    as_str = y.astype(str).str.strip()

    # OpenML hands some targets back as floats, so the same class arrives as
    # "1.0" on one dataset and "1" on another. Normalise before comparing.
    numeric = pd.to_numeric(as_str, errors="coerce")
    if numeric.notna().all():
        as_str = numeric.astype(int).astype(str)

    uniques = sorted(as_str.unique())
    if positive_label not in uniques:
        raise ValueError(
            f"positive label {positive_label!r} not found; observed {uniques}"
        )
    if len(uniques) != 2:
        raise ValueError(f"expected a binary target, observed {uniques}")
    return (as_str == positive_label).to_numpy().astype(np.int8)


def _split_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical, numeric = [], []
    for col in X.columns:
        if isinstance(X[col].dtype, pd.CategoricalDtype) or X[col].dtype == object:
            categorical.append(col)
        else:
            numeric.append(col)
    return categorical, numeric


def _verify(spec: DatasetSpec, dataset: Dataset) -> None:
    """Fail loudly if the upstream data no longer matches the published shape."""
    problems = []
    if dataset.n_samples != spec.expected_rows:
        problems.append(f"rows {dataset.n_samples} != expected {spec.expected_rows}")
    if dataset.n_features != spec.expected_features:
        problems.append(
            f"features {dataset.n_features} != expected {spec.expected_features}"
        )
    if abs(dataset.minority_pct - spec.expected_minority_pct) > 0.05:
        problems.append(
            f"minority {dataset.minority_pct:.4f}% != expected "
            f"{spec.expected_minority_pct:.4f}%"
        )
    if problems:
        raise ValueError(f"{spec.key} failed verification: " + "; ".join(problems))


def load_dataset(key: str, use_cache: bool = True) -> Dataset:
    """Load one of the four study datasets, downloading it once if needed."""
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; available: {sorted(DATASETS)}")
    spec = DATASETS[key]
    frame_path, meta_path = _cache_paths(key)

    downloaded = False
    if use_cache and frame_path.exists():
        frame = pd.read_parquet(frame_path)
        y = frame.pop("__target__").to_numpy().astype(np.int8)
        X = frame
    else:
        X, raw_y = _download(spec)
        y = _binarise(raw_y, spec.positive_label)
        downloaded = True

    categorical, numeric = _split_column_types(X)
    dataset = Dataset(
        key=key,
        X=X,
        y=y,
        categorical_columns=categorical,
        numeric_columns=numeric,
        time_column=spec.time_column,
        description=spec.description,
    )
    # Verified before anything is cached. Writing first meant that a download
    # which failed its shape check still left a bad parquet behind, and the next
    # run read that instead of downloading, so the fix to the specification had
    # no effect and the same error repeated.
    _verify(spec, dataset)

    if downloaded:
        to_cache = X.copy()
        to_cache["__target__"] = y
        to_cache.to_parquet(frame_path, index=False)

    if not meta_path.exists():
        meta = dataset.summary() | {
            "openml_id": spec.openml_id,
            "description": spec.description,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return dataset


def load_all(keys: list[str] | None = None) -> dict[str, Dataset]:
    return {k: load_dataset(k) for k in (keys or list(DATASETS))}


def save_scores(
    paper: int,
    stage: str,
    arm: str,
    fold: int,
    seed: int,
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Path:
    """Persist test-set predictions alongside the summary metrics.

    Fold-level metrics give only five numbers per model, which is far too few to
    test a difference on. Keeping the per-transaction scores lets the analysis
    run DeLong's test on the pooled predictions, where the sample size is the
    number of transactions rather than the number of folds.
    """
    directory = PROJECT_ROOT / "results" / f"paper{paper}" / "scores"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stage}__{safe_name(arm)}__fold{fold}__seed{seed}.npz"
    np.savez_compressed(
        path,
        y_true=np.asarray(y_true, dtype=np.int8),
        scores=np.asarray(scores, dtype=np.float32),
    )
    return path


def load_scores(paper: int, stage: str, arm: str, fold: int, seed: int = 0):
    """Read back saved predictions, or None if that run did not store any."""
    path = (
        PROJECT_ROOT / "results" / f"paper{paper}" / "scores"
        / f"{stage}__{safe_name(arm)}__fold{fold}__seed{seed}.npz"
    )
    if not path.exists():
        return None
    with np.load(path) as payload:
        return payload["y_true"], payload["scores"]


def safe_name(arm: str) -> str:
    """Filesystem-safe form of an arm name, shared by every writer and reader."""
    return arm.replace("+", "_").replace("=", "-").replace("/", "-").replace(",", "-")


def write_json(path: Path | str, payload: dict) -> None:
    """Persist a result record, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def read_runs(directory: Path | str) -> list[dict]:
    """Read every raw run record in a directory, for table generation."""
    directory = Path(directory)
    if not directory.exists():
        return []
    records = []
    for path in sorted(directory.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records
