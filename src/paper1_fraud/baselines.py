"""Baseline detectors reproduced under the same protocol as the proposed model.

Two families are covered:

* classical and ensemble learners, which are the strongest published performers
  on this dataset;
* deep classifiers as they are actually used in the fraud literature, where the
  30 features are treated as a length-30 sequence so that convolutional and
  recurrent layers have something to slide over.

Reproducing them here rather than quoting published numbers is the point. Every
paper on ULB uses a different split, a different resampling step and a different
threshold, so their tables are not comparable to each other. Ours are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.torch_utils import (
    BalancedBatches,
    DeviceArrays,
    resolve_device,
    sequential_batches,
)

SKLEARN_BASELINES = ["logistic_regression", "decision_tree", "random_forest", "knn"]
BOOSTING_BASELINES = ["xgboost", "lightgbm", "catboost"]
DEEP_BASELINES = ["mlp", "cnn1d", "bilstm", "autoencoder"]
ALL_BASELINES = SKLEARN_BASELINES + BOOSTING_BASELINES + DEEP_BASELINES


def build_shallow_baseline(name: str, seed: int = 0, class_weight: bool = True):
    """Instantiate a scikit-learn style baseline with sensible published settings."""
    name = name.lower()
    weight = "balanced" if class_weight else None

    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000, class_weight=weight, random_state=seed)
    if name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=20, class_weight=weight, random_state=seed
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample" if class_weight else None,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            class_weight=weight,
            n_jobs=-1,
            verbose=-1,
            random_state=seed,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="Balanced" if class_weight else None,
            verbose=0,
            allow_writing_files=False,
            random_seed=seed,
        )
    raise ValueError(f"unknown baseline {name!r}")


# --------------------------------------------------------------------------- #
# Deep baselines
# --------------------------------------------------------------------------- #
@dataclass
class DeepConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    # A training step on this GPU costs the same 8 ms whether the batch holds
    # 512 rows or 8192 (measured in scripts/profile_training_step.py): the cost
    # is kernel launch latency, not arithmetic. Larger batches therefore buy an
    # order of magnitude in epoch time for free. The positive fraction is held
    # at the same 6.25% so the loss sees the same class mix per step.
    batch_size: int = 2048
    positives_per_batch: int = 128
    max_epochs: int = 100
    patience: int = 12
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    use_focal: bool = True
    hidden: int = 128
    dropout: float = 0.2
    seed: int = 0


def focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float, alpha: float
) -> torch.Tensor:
    """Binary focal loss (Lin et al., 2017), the standard remedy in this literature."""
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1 - probability) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (alpha_t * (1 - p_t).pow(gamma) * bce).mean()


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _CNN1D(nn.Module):
    """Features are read as a length-d, single-channel signal.

    This is how convolutional fraud detectors on ULB are normally set up. The
    ordering of PCA components carries no real adjacency, which is precisely the
    limitation the results section discusses.
    """

    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1))).squeeze(-1)


class _BiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden // 2, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        output, _ = self.lstm(x.unsqueeze(-1))
        return self.head(output[:, -1, :]).squeeze(-1)


class DeepBaseline:
    """Shared training loop for the supervised deep baselines."""

    # Recurrent inference materialises a hidden state for every timestep, so the
    # 16k batch that suits the feed-forward models exhausts 4 GB of VRAM here.
    INFERENCE_BATCH = {"bilstm": 2048}
    DEFAULT_INFERENCE_BATCH = 16384

    def __init__(self, kind: str, config: DeepConfig | None = None, device=None) -> None:
        self.kind = kind
        self.config = config or DeepConfig()
        self.device = resolve_device(device)
        self.model: nn.Module | None = None

    def _build(self, input_dim: int) -> nn.Module:
        cfg = self.config
        if self.kind == "mlp":
            return _MLP(input_dim, cfg.hidden, cfg.dropout)
        if self.kind == "cnn1d":
            return _CNN1D(input_dim, cfg.hidden, cfg.dropout)
        if self.kind == "bilstm":
            return _BiLSTM(input_dim, cfg.hidden, cfg.dropout)
        raise ValueError(f"unknown deep baseline {self.kind!r}")

    def fit(self, X_train, y_train, X_val, y_val) -> "DeepBaseline":
        from sklearn.metrics import average_precision_score

        cfg = self.config
        torch.manual_seed(cfg.seed)
        self.model = self._build(X_train.shape[1]).to(self.device)
        optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        pos_weight = torch.tensor(
            [(y_train == 0).sum() / max(1, (y_train == 1).sum())],
            device=self.device,
            dtype=torch.float32,
        )
        train = DeviceArrays(X_train, y_train, self.device)
        batches = BalancedBatches(
            y_train, cfg.batch_size, cfg.positives_per_batch, self.device, cfg.seed
        )
        X_val_t = torch.as_tensor(
            np.ascontiguousarray(X_val, dtype=np.float32), device=self.device
        )

        best, best_state, stale = -np.inf, None, 0
        for _ in range(cfg.max_epochs):
            self.model.train()
            for idx in batches:
                xb, yb = train.X[idx], train.y[idx]
                logits = self.model(xb)
                if cfg.use_focal:
                    loss = focal_loss(logits, yb, cfg.focal_gamma, cfg.focal_alpha)
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimiser.step()

            score = float(average_precision_score(y_val, self._raw_scores(X_val_t)))
            if score > best + 1e-5:
                best, stale = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_val_auprc = float(best)
        return self

    @torch.no_grad()
    def _raw_scores(self, X_t: torch.Tensor, batch_size: int | None = None) -> np.ndarray:
        assert self.model is not None
        batch_size = batch_size or self.INFERENCE_BATCH.get(
            self.kind, self.DEFAULT_INFERENCE_BATCH
        )
        was_training = self.model.training
        self.model.eval()
        chunks = []
        for start, stop in sequential_batches(len(X_t), batch_size):
            chunks.append(torch.sigmoid(self.model(X_t[start:stop])).cpu().numpy())
        self.model.train(was_training)
        return np.concatenate(chunks)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.as_tensor(
            np.ascontiguousarray(X, dtype=np.float32), device=self.device
        )
        return self._raw_scores(X_t)


class AutoencoderAnomaly:
    """Unsupervised baseline: fit on legitimate traffic, score by reconstruction error.

    Included because it is the most common deep approach in the fraud literature
    that needs no labels, and because it sets the floor that any supervised
    representation has to clear.
    """

    def __init__(self, config: DeepConfig | None = None, device=None) -> None:
        self.config = config or DeepConfig()
        self.device = resolve_device(device)
        self.model: nn.Module | None = None

    def fit(self, X_train, y_train, X_val, y_val) -> "AutoencoderAnomaly":
        cfg = self.config
        torch.manual_seed(cfg.seed)
        input_dim = X_train.shape[1]
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        ).to(self.device)

        optimiser = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)
        tensor = torch.as_tensor(
            np.ascontiguousarray(X_train[y_train == 0], dtype=np.float32), device=self.device
        )
        val_normal = np.ascontiguousarray(X_val[y_val == 0], dtype=np.float32)

        best, best_state, stale = np.inf, None, 0
        generator = torch.Generator(device=self.device.type).manual_seed(cfg.seed)
        for _ in range(cfg.max_epochs):
            self.model.train()
            perm = torch.randperm(len(tensor), generator=generator, device=self.device)
            for start, stop in sequential_batches(len(tensor), cfg.batch_size):
                batch = tensor[perm[start:stop]]
                loss = F.mse_loss(self.model(batch), batch)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()

            val_loss = float(self._reconstruction_error(val_normal).mean())
            if val_loss < best - 1e-6:
                best, stale = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _reconstruction_error(self, X: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        assert self.model is not None
        was_training = self.model.training
        self.model.eval()
        tensor = torch.as_tensor(
            np.ascontiguousarray(X, dtype=np.float32), device=self.device
        )
        errors = []
        for start, stop in sequential_batches(len(tensor), batch_size):
            batch = tensor[start:stop]
            errors.append(((self.model(batch) - batch) ** 2).mean(dim=1).cpu().numpy())
        self.model.train(was_training)
        return np.concatenate(errors)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Squash the reconstruction error into [0, 1] so it can be ranked and thresholded."""
        error = self._reconstruction_error(X)
        # Rank-based normalisation keeps the ordering, which is all the
        # threshold-free metrics depend on, without assuming an error scale.
        order = error.argsort().argsort()
        return (order / max(1, len(order) - 1)).astype(float)
