"""GPU batching helpers shared by every torch model in the project.

All four datasets fit comfortably in 4 GB of VRAM (the largest training matrix
is roughly 30 MB), so the tensors are uploaded once and batches are formed by
indexing them in place. Going through a ``DataLoader`` instead costs about two
orders of magnitude here, because it fetches one row at a time in Python and
then collates - a per-epoch cost of tens of seconds on a model that needs
milliseconds of actual compute.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def cuda_usable() -> bool:
    """Whether CUDA is available and actually responding.

    ``torch.cuda.is_available()`` returns True on a device the driver has since
    lost, and the failure then surfaces as an out-of-memory error on a two
    megabyte allocation. Touching the device once at startup turns that into an
    answer we can act on.
    """
    if _FORCE_CPU or not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda").sum().item()
        return True
    except Exception:
        return False


def release_memory() -> None:
    """Return cached blocks to the driver between models.

    The allocator normally holds freed blocks for reuse, which is the right
    default inside one training run. Across a long sweep of differently shaped
    models on a 4 GB card that also drives the display, the retained-but-unused
    blocks fragment the heap until an allocation that should fit does not. Called
    between arms, this keeps a sweep from failing on its fiftieth model.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class DeviceArrays:
    """Feature and label tensors held on the training device."""

    def __init__(self, X: np.ndarray, y: np.ndarray, device: torch.device) -> None:
        self.X = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32), device=device)
        self.y = torch.as_tensor(np.ascontiguousarray(y, dtype=np.float32), device=device)
        self.device = device

    def __len__(self) -> int:
        return int(self.X.shape[0])


class BalancedBatches:
    """Index generator that guarantees a fixed number of positives per batch.

    A uniformly drawn batch of 512 rows from the ULB training window contains
    fewer than one fraud on average, which leaves a contrastive objective with no
    positive pairs and a focal loss with almost no signal. Positives are drawn
    with replacement, negatives without, so one epoch remains one pass over the
    legitimate transactions.
    """

    def __init__(
        self,
        y: np.ndarray,
        batch_size: int,
        positives_per_batch: int,
        device: torch.device,
        seed: int = 0,
    ) -> None:
        y = np.asarray(y)
        self.pos = torch.as_tensor(np.flatnonzero(y == 1), device=device)
        self.neg = torch.as_tensor(np.flatnonzero(y == 0), device=device)
        self.n_pos = max(1, min(positives_per_batch, batch_size // 2))
        self.n_neg = max(1, batch_size - self.n_pos)
        self.device = device
        self.generator = torch.Generator(device=device.type).manual_seed(seed)
        self.n_batches = max(1, int(len(self.neg)) // self.n_neg)

    def __iter__(self):
        order = self.neg[torch.randperm(len(self.neg), generator=self.generator, device=self.device)]
        for b in range(self.n_batches):
            negatives = order[b * self.n_neg : (b + 1) * self.n_neg]
            picks = torch.randint(
                0, len(self.pos), (self.n_pos,), generator=self.generator, device=self.device
            )
            yield torch.cat([self.pos[picks], negatives])

    def __len__(self) -> int:
        return self.n_batches


def sequential_batches(n: int, batch_size: int):
    """Contiguous index ranges, for inference and reconstruction passes."""
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


# Set FORCE_CPU=1 in the environment to run a whole sweep on the CPU. Useful
# when the GPU is unavailable or unreliable; every model here is small enough
# that the CPU is a workable, if slower, alternative.
_FORCE_CPU = os.environ.get("FORCE_CPU", "").strip() in {"1", "true", "True"}


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if _FORCE_CPU:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def force_cpu():
    """Make every model built inside this block run on the CPU.

    Used to retry an arm that hit an out-of-memory error. Model classes ask for
    their device through ``resolve_device``, so overriding it here redirects the
    whole arm without threading a device argument through the call stack.
    """
    global _FORCE_CPU
    previous = _FORCE_CPU
    _FORCE_CPU = True
    try:
        yield
    finally:
        _FORCE_CPU = previous


def run_resilient(fn, *args, retries: int = 3, wait_seconds: float = 20.0, **kwargs):
    """Run ``fn``, retrying on the GPU before giving up and using the CPU.

    This 4 GB card also drives the display, so free memory moves while a sweep
    is running and an allocation of a few megabytes can be refused with
    gigabytes nominally free. The condition is usually transient, and the CPU
    fallback is expensive for the recurrent models, where a fold takes minutes
    instead of seconds. So we wait for the pressure to pass and try again a few
    times, and only fall back to the CPU when the GPU keeps refusing.
    """
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except torch.OutOfMemoryError:
            release_memory()
            if attempt < retries - 1:
                LOGGER.warning(
                    "  CUDA out of memory; waiting %.0fs and retrying on GPU (%d/%d)",
                    wait_seconds, attempt + 1, retries - 1,
                )
                time.sleep(wait_seconds)

    LOGGER.warning("  GPU still refusing allocations; running this arm on the CPU")
    with force_cpu():
        return fn(*args, **kwargs)
