"""The Paper 1 hybrid detector: TASR -> SCAE -> gradient boosting head.

Every component is switchable from the constructor, so the ablation table in the
paper is produced by the same code path as the full model rather than by a
separate re-implementation. Turning everything off yields a plain LightGBM on
raw features, which is exactly the strongest published baseline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.common.metrics import best_f1_threshold, evaluate
from src.paper1_fraud.encoder import SCAEConfig, SCAETrainer
from src.paper1_fraud.resampling import (
    TimeAwareStratifiedResampler,
    build_reference_sampler,
)


@dataclass
class HybridConfig:
    """Configuration for one arm of the experiment."""

    # Resampling: "tasr", "none", or any name accepted by build_reference_sampler.
    sampler: str = "tasr"
    target_ratio: float = 0.10
    n_blocks: int = 24
    k_neighbors: int = 5
    cleaning: str | None = "tomek"

    # Representation module.
    use_encoder: bool = True
    use_raw_features: bool = True
    encoder_on_resampled: bool = True
    scae: SCAEConfig = field(default_factory=SCAEConfig)

    # Classification head.
    head: str = "lightgbm"
    head_params: dict = field(default_factory=dict)
    scale_pos_weight: bool = False

    seed: int = 0

    def label(self) -> str:
        parts = [f"sampler={self.sampler}"]
        if self.use_encoder:
            parts.append("scae" if self.scae.contrastive_weight > 0 else "ae")
            if not self.use_raw_features:
                parts.append("embonly")
        parts.append(self.head)
        return "+".join(parts)


DEFAULT_LGBM = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}

DEFAULT_XGB = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "eval_metric": "aucpr",
}


def fit_lgbm(model, X, y, X_val, y_val, stopping_rounds: int = 100):
    """Fit a LightGBM model with early stopping, across API versions.

    LightGBM 4.7 renamed the validation arguments; supporting both keeps the
    project runnable on the version range pinned in requirements.txt.
    """
    from lightgbm import early_stopping, log_evaluation

    callbacks = [early_stopping(stopping_rounds, verbose=False), log_evaluation(0)]
    try:
        model.fit(
            X, y, eval_X=X_val, eval_y=y_val,
            eval_metric="average_precision", callbacks=callbacks,
        )
    except TypeError:
        model.fit(
            X, y, eval_set=[(X_val, y_val)],
            eval_metric="average_precision", callbacks=callbacks,
        )
    return model


def _reference_sampler_times(sampler, t: np.ndarray, n_out: int) -> np.ndarray:
    """Recover a timestamp per row after an imbalanced-learn sampler has run.

    Reference samplers are not time aware, which is exactly the gap TASR fills.
    Under-samplers expose the rows they kept, so their timestamps are exact;
    over-samplers append synthetic rows with no time of their own and are given
    the training median.
    """
    kept = getattr(sampler, "sample_indices_", None)
    if kept is not None and len(kept) == n_out:
        return t[np.asarray(kept)]

    times = np.full(n_out, float(np.median(t)))
    if n_out >= len(t):
        times[: len(t)] = t
    return times


class HybridFraudDetector:
    """Fit-once, predict-many wrapper around the full pipeline."""

    def __init__(self, config: HybridConfig | None = None) -> None:
        self.config = config or HybridConfig()
        self.encoder_: SCAETrainer | None = None
        self.head_ = None
        self.threshold_: float = 0.5
        self.timings_: dict[str, float] = {}
        self.resample_stats_: dict = {}

    # ------------------------------------------------------------------ #
    def _resample(self, X: np.ndarray, y: np.ndarray, t: np.ndarray):
        cfg = self.config
        if cfg.sampler in {"none", None}:
            self.resample_stats_ = {"n_before": len(y), "n_after": len(y), "n_synthetic": 0}
            return X, y, t

        if cfg.sampler == "tasr":
            resampler = TimeAwareStratifiedResampler(
                target_ratio=cfg.target_ratio,
                n_blocks=cfg.n_blocks,
                k_neighbors=cfg.k_neighbors,
                cleaning=cfg.cleaning,
                random_state=cfg.seed,
            )
            result = resampler.fit_resample(X, y, t)
            self.resample_stats_ = result.stats
            return result.X, result.y, result.time

        sampler = build_reference_sampler(cfg.sampler, cfg.target_ratio, cfg.seed)
        X_res, y_res = sampler.fit_resample(X, y)
        t_res = _reference_sampler_times(sampler, t, len(y_res))
        self.resample_stats_ = {
            "n_before": int(len(y)),
            "n_after": int(len(y_res)),
            "n_synthetic": int(len(y_res) - len(y)),
            "n_minority_after": int(y_res.sum()),
        }
        return np.asarray(X_res, dtype=np.float32), np.asarray(y_res).astype(np.int8), t_res

    def _build_head(self, y_train: np.ndarray):
        cfg = self.config
        if cfg.head == "lightgbm":
            from lightgbm import LGBMClassifier

            params = {**DEFAULT_LGBM, **cfg.head_params, "random_state": cfg.seed}
            if cfg.scale_pos_weight:
                params["scale_pos_weight"] = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
            return LGBMClassifier(**params)

        if cfg.head == "xgboost":
            from xgboost import XGBClassifier

            params = {**DEFAULT_XGB, **cfg.head_params, "random_state": cfg.seed}
            if cfg.scale_pos_weight:
                params["scale_pos_weight"] = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
            return XGBClassifier(**params)

        raise ValueError(f"unknown head {cfg.head!r}")

    def _features(self, X: np.ndarray) -> np.ndarray:
        if self.encoder_ is None:
            return X
        embedding = self.encoder_.transform(X)
        if self.config.use_raw_features:
            return np.hstack([X, embedding])
        return embedding

    # ------------------------------------------------------------------ #
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        t_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "HybridFraudDetector":
        cfg = self.config

        started = time.perf_counter()
        X_res, y_res, _ = self._resample(X_train, y_train, t_train)
        self.timings_["resample_s"] = time.perf_counter() - started

        if cfg.use_encoder:
            started = time.perf_counter()
            encoder_X, encoder_y = (
                (X_res, y_res) if cfg.encoder_on_resampled else (X_train, y_train)
            )
            self.encoder_ = SCAETrainer(cfg.scae).fit(encoder_X, encoder_y, X_val, y_val)
            self.timings_["encoder_s"] = time.perf_counter() - started

        started = time.perf_counter()
        train_features = self._features(X_res)
        val_features = self._features(X_val)

        self.head_ = self._build_head(y_res)
        if cfg.head == "lightgbm":
            fit_lgbm(self.head_, train_features, y_res, val_features, y_val)
        else:
            self.head_.set_params(early_stopping_rounds=100)
            self.head_.fit(train_features, y_res, eval_set=[(val_features, y_val)], verbose=False)
        self.timings_["head_s"] = time.perf_counter() - started

        # The operating threshold is chosen on validation data and then frozen.
        val_scores = self.head_.predict_proba(val_features)[:, 1]
        self.threshold_ = best_f1_threshold(y_val, val_scores)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.head_ is not None, "call fit() first"
        return self.head_.predict_proba(self._features(X))[:, 1]

    def score_test(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        started = time.perf_counter()
        scores = self.predict_proba(X_test)
        self.timings_["inference_s"] = time.perf_counter() - started
        self.test_scores_ = scores
        return evaluate(y_test, scores, threshold=self.threshold_)
