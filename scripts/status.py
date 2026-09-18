"""Show what has been computed so far, per paper and stage.

Useful after an interruption to see exactly what needs rerunning.

    python -m scripts.status
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.common.io import PROJECT_ROOT


def survey(paper: int) -> None:
    root = PROJECT_ROOT / "results" / f"paper{paper}"
    raw = root / "raw_runs"
    scores = root / "scores"

    run_files = sorted(raw.glob("*.json")) if raw.exists() else []
    score_files = sorted(scores.glob("*.npz")) if scores.exists() else []

    print(f"\npaper {paper}: {len(run_files)} runs, {len(score_files)} score files")
    if not run_files:
        return

    by_stage: Counter = Counter()
    by_stage_dataset: Counter = Counter()
    for path in run_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  corrupt: {path.name}")
            continue
        stage = payload.get("stage", "?")
        by_stage[stage] += 1
        by_stage_dataset[(payload.get("dataset", "?"), stage)] += 1

    if paper == 1:
        for stage, count in sorted(by_stage.items()):
            print(f"  {stage:<12} {count:3d} runs")
    else:
        for (dataset, stage), count in sorted(by_stage_dataset.items()):
            print(f"  {dataset:<22} {stage:<14} {count:3d} runs")

    tuning = root / "tuning"
    if tuning.exists():
        studies = sorted(tuning.glob("*__hpo__*.json"))
        configs = sorted(tuning.glob("*__best_config__*.json"))
        print(f"  tuning: {len(studies)} optimiser studies, {len(configs)} chosen configs")


def main() -> None:
    import torch

    from src.common.torch_utils import cuda_usable

    usable = cuda_usable()
    print(f"torch {torch.__version__}, cuda usable: {usable}")
    if usable:
        free, total = torch.cuda.mem_get_info()
        print(
            f"  {torch.cuda.get_device_name(0)}: "
            f"{free / 1024**3:.2f} GB free of {total / 1024**3:.2f} GB"
        )
    elif torch.cuda.is_available():
        print("  a CUDA device is reported but does not respond; running on CPU")
    survey(1)
    survey(2)


if __name__ == "__main__":
    main()
