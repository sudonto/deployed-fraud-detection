"""Supervised Contrastive Autoencoder (SCAE).

The deep half of the Paper 1 hybrid. It learns a representation rather than a
decision, and the actual classification is left to a gradient boosting head.
That division of labour is deliberate: on this dataset boosted trees beat every
end-to-end deep classifier reported in the literature, so the useful question is
not "can a network replace the tree" but "can a network hand the tree a better
feature space".

The training objective has two terms:

reconstruction
    Keeps the embedding faithful to the input and stops it from collapsing onto
    whatever separates the 492 frauds in the training window, which would not
    survive contact with a later time period.
supervised contrastive
    Pulls frauds towards other frauds and away from legitimate transactions.
    Unlike a cross-entropy head it uses every fraud pair in the batch, which is
    what makes it usable when a batch contains a handful of positives.

Because a uniformly sampled batch of 512 rows from ULB contains 0.9 frauds on
average, batches are drawn by a sampler that guarantees a fixed number of
positives. Without it the contrastive term is undefined for most steps.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

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


@dataclass
class SCAEConfig:
    hidden_dims: tuple[int, ...] = (128, 64)
    embedding_dim: int = 32
    dropout: float = 0.15
    temperature: float = 0.1
    contrastive_weight: float = 1.0
    reconstruction_weight: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    # Sized like the deep baselines: on this GPU a step costs the same at 512 or
    # 8192 rows, so the larger batch is close to free. It also gives the
    # contrastive term many more positive pairs to work with per step.
    batch_size: int = 2048
    positives_per_batch: int = 128
    max_epochs: int = 100
    patience: int = 12
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    """SupCon (Khosla et al., 2020) over L2-normalised embeddings."""
    z = F.normalize(embeddings, dim=1)
    similarity = z @ z.T / temperature

    # Numerical stabilisation: subtracting the row max leaves softmax unchanged.
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()

    n = z.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask

    exp_sim = torch.exp(similarity).masked_fill(self_mask, 0.0)
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    n_positives = positive_mask.sum(dim=1)
    valid = n_positives > 0
    if not valid.any():
        return embeddings.sum() * 0.0

    mean_log_prob = (positive_mask * log_prob).sum(dim=1)[valid] / n_positives[valid]
    return -mean_log_prob.mean()


class SCAE(nn.Module):
    """Encoder / decoder pair. The encoder output is what the GBDT head consumes."""

    def __init__(self, input_dim: int, config: SCAEConfig) -> None:
        super().__init__()
        self.config = config

        encoder_layers: list[nn.Module] = []
        prev = input_dim
        for width in config.hidden_dims:
            encoder_layers += [
                nn.Linear(prev, width),
                nn.BatchNorm1d(width),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ]
            prev = width
        encoder_layers.append(nn.Linear(prev, config.embedding_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        prev = config.embedding_dim
        for width in reversed(config.hidden_dims):
            decoder_layers += [
                nn.Linear(prev, width),
                nn.BatchNorm1d(width),
                nn.GELU(),
            ]
            prev = width
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.decoder(z)


class SCAETrainer:
    """Fit an SCAE and expose it as a feature transformer."""

    def __init__(self, config: SCAEConfig | None = None, device: str | None = None) -> None:
        self.config = config or SCAEConfig()
        self.device = resolve_device(device)
        self.model: SCAE | None = None
        self.history: list[dict] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "SCAETrainer":
        cfg = self.config
        torch.manual_seed(cfg.seed)

        self.model = SCAE(X_train.shape[1], cfg).to(self.device)
        optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.max_epochs)

        train = DeviceArrays(X_train, y_train, self.device)
        val = DeviceArrays(X_val, y_val, self.device)
        train_batches = BalancedBatches(
            y_train, cfg.batch_size, cfg.positives_per_batch, self.device, cfg.seed
        )
        val_batches = BalancedBatches(
            y_val, cfg.batch_size, cfg.positives_per_batch, self.device, cfg.seed
        )

        best_loss, best_state, stale = float("inf"), None, 0
        for epoch in range(cfg.max_epochs):
            train_loss = self._run_epoch(train, train_batches, optimiser)
            val_loss = self._run_epoch(val, val_batches, optimiser=None)
            scheduler.step()
            self.history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )

            if val_loss < best_loss - 1e-5:
                best_loss, stale = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_val_loss = best_loss
        self.epochs_run = len(self.history)
        return self

    def _run_epoch(self, arrays: DeviceArrays, batches: BalancedBatches, optimiser) -> float:
        assert self.model is not None
        training = optimiser is not None
        self.model.train(training)
        cfg = self.config

        totals, n_batches = 0.0, 0
        with torch.set_grad_enabled(training):
            for idx in batches:
                xb, yb = arrays.X[idx], arrays.y[idx]
                z, reconstruction = self.model(xb)
                loss = cfg.reconstruction_weight * F.mse_loss(reconstruction, xb)
                if cfg.contrastive_weight > 0:
                    loss = loss + cfg.contrastive_weight * supervised_contrastive_loss(
                        z, yb.long(), cfg.temperature
                    )

                if training:
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimiser.step()

                totals += float(loss.detach())
                n_batches += 1
        return totals / max(n_batches, 1)

    @torch.no_grad()
    def transform(self, X: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        """Map inputs to the learned embedding space."""
        assert self.model is not None, "call fit() first"
        self.model.eval()
        tensor = torch.as_tensor(
            np.ascontiguousarray(X, dtype=np.float32), device=self.device
        )
        chunks = []
        for start, stop in sequential_batches(len(tensor), batch_size):
            chunks.append(self.model.encoder(tensor[start:stop]).cpu().numpy())
        return np.vstack(chunks).astype(np.float32)
