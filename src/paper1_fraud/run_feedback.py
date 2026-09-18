"""Experiment driver for the Paper 1 feedback-loop study.

Every arm is replayed over the same stream with the same rounds and the same
analyst budget, so the only thing that varies is how the simulated bank records
what it learns. Each run writes its own JSON file and is skipped on re-run,
which keeps the sweep resumable.

Stages
------
main        Every bookkeeping rule, on every dataset, at the default capacity.
capacity    The three arms the paper leads with, swept across analyst capacity.
censoring   The censored-region width, which is the proposed rule's one knob.
detector    LightGBM hyperparameters around the default, on the four headline arms.

Usage
-----
    python -m src.paper1_fraud.run_feedback --stage all
    python -m src.paper1_fraud.run_feedback --stage main --datasets baf_base
"""

from __future__ import annotations

import argparse
import logging
import time

from dataclasses import replace

from src.common.io import PROJECT_ROOT, safe_name, write_json
from src.common.seeds import set_seed
from src.paper1_fraud.feedback import FeedbackConfig, run_simulation, summarise
from src.paper1_fraud.streams import load_stream

LOGGER = logging.getLogger("paper1.feedback")
RESULTS_DIR = PROJECT_ROOT / "results" / "paper1" / "raw_runs"

# BAF leads because its eight months are real retraining cycles with over a
# thousand frauds each. IEEE-CIS is the second dataset and the harder one, at
# 455 features over six months. ULB is kept because the prior work uses it, and
# because it shows what happens when a stream is too short for a loop to form.
DATASETS = ["baf_base", "ieee_cis", "ulb_creditcard"]
ROBUSTNESS_DATASET = "baf_variant3"

BUDGET = 0.005
EXPLORATION = 0.10
CENSOR_WIDTH = 20.0
SEEDS = [0, 1, 2, 3, 4]

# Five seeds on the headline dataset, three on the others. The variance between
# seeds comes from the stochastic review draw and the detector's subsampling,
# and on a stream with over ten thousand frauds it is small; three replicates
# are enough to show the ordering holds, and the saving pays for the capacity
# sweep, which is where the paper's argument actually lives.
SEEDS_BY_DATASET = {"baf_base": SEEDS, "ieee_cis": SEEDS[:3], "ulb_creditcard": SEEDS[:3]}

# The loop has two distinct failure modes and the arms are laid out as a 2x2
# over them, so the ablation reads off the table directly. One failure is
# informational: reviewing only the top of the ranking means the stream is never
# sampled anywhere else, so the model never learns it is wrong. The other is
# corruption: the unreviewed rows are written down as legitimate, so the model
# is actively taught that its blind spot is empty. Banded exploration addresses
# the first, soft censoring the second, and the proposed arm is both.
ARMS: dict[str, FeedbackConfig] = {
    "oracle_full_labels": FeedbackConfig("topk", "full_labels"),
    "naive_assume_negative": FeedbackConfig("topk", "assume_negative"),
    "reviewed_only": FeedbackConfig("topk", "reviewed_only"),
    "explore_banded": FeedbackConfig("banded", "assume_negative", exploration=EXPLORATION),
    "explore_uniform": FeedbackConfig("uniform", "assume_negative", exploration=EXPLORATION),
    "ips": FeedbackConfig("banded", "ips", exploration=EXPLORATION),
    "ips_augmented": FeedbackConfig("banded", "ips_augmented", exploration=EXPLORATION),
    "censor": FeedbackConfig("topk", "censor", censor_width=CENSOR_WIDTH),
    "soft_censor": FeedbackConfig("topk", "soft_censor"),
    "censor_ips": FeedbackConfig(
        "banded", "censor_ips", exploration=EXPLORATION, censor_width=CENSOR_WIDTH
    ),
    # The two ways of combining the halves. Both are reported; which one the
    # paper recommends is decided by the full results rather than here, and the
    # table carries both so the choice is visible.
    "explore_censor": FeedbackConfig(
        "banded", "censor", exploration=EXPLORATION, censor_width=CENSOR_WIDTH
    ),
    "explore_soft_censor": FeedbackConfig("banded", "soft_censor", exploration=EXPLORATION),
}

# The cells of the 2x2 over the loop's two failure modes, plus the oracle that
# bounds them: neither remedy, exploration alone, censoring alone, and both.
HEADLINE_ARMS = [
    "oracle_full_labels", "naive_assume_negative", "explore_banded",
    "censor", "soft_censor", "explore_censor", "explore_soft_censor",
]

# Capacity is expressed as a share of transactions reviewed. On BAF, whose fraud
# rate is 1.1%, this grid runs from about a tenth of a review per fraud to two
# per fraud, which brackets what a real team can afford.
CAPACITY_GRID = [0.001, 0.002, 0.005, 0.01, 0.02]
# The operating point of the main study is 20 budgets; the rest of the grid
# brackets it so the sweep contains the width every other number uses.
CENSOR_GRID = [2.0, 5.0, 20.0, 50.0, 100.0]

# Coordinate sweep around DETECTOR_PARAMS. Each setting moves one knob; the
# recovered gap, not the absolute AUPRC, is what the table has to show is
# stable.
DETECTOR_GRID = {
    "trees200": {"n_estimators": 200},
    "trees1000": {"n_estimators": 1000},
    "lr002": {"learning_rate": 0.02},
    "lr010": {"learning_rate": 0.10},
    "leaves31": {"num_leaves": 31},
    "leaves127": {"num_leaves": 127},
}
DETECTOR_ARMS = (
    "oracle_full_labels",
    "naive_assume_negative",
    "censor",
    "soft_censor",
)

# Streams are expensive to build and every arm replays the same one, so they are
# loaded once per dataset rather than once per run.
_STREAMS: dict[str, object] = {}

# Set only by the smoke test. A subsampled run is not a result, and writing one
# into the results directory would make it indistinguishable from a real one, so
# the destination moves with it.
SUBSAMPLE: float | None = None


def stream_for(key: str):
    if key not in _STREAMS:
        stream = load_stream(key, subsample=SUBSAMPLE)
        LOGGER.info(
            "%s: %d rows, %d frauds, %d rounds",
            key, len(stream.y), int(stream.y.sum()), stream.n_rounds,
        )
        _STREAMS[key] = stream
    return _STREAMS[key]


def result_path(stage: str, dataset: str, arm: str, seed: int, tag: str = ""):
    suffix = f"__{safe_name(tag)}" if tag else ""
    return RESULTS_DIR / (
        f"feedback_{stage}__{safe_name(dataset)}__{safe_name(arm)}{suffix}__seed{seed}.json"
    )


def execute(
    stage: str,
    dataset: str,
    arm: str,
    config: FeedbackConfig,
    seed: int,
    budget: float = BUDGET,
    tag: str = "",
) -> None:
    path = result_path(stage, dataset, arm, seed, tag)
    if path.exists():
        return

    stream = stream_for(dataset)
    LOGGER.info("%s | %s | %s | seed %d%s", stage, dataset, arm, seed, f" | {tag}" if tag else "")
    set_seed(seed)
    started = time.perf_counter()
    outcomes = run_simulation(
        stream.X, stream.y, stream.round_index, config,
        warmup_rounds=1, budget_fraction=budget, seed=seed,
        categorical=stream.categorical_indices,
    )
    elapsed = time.perf_counter() - started
    summary = summarise(outcomes)

    LOGGER.info(
        "  caught %d/%d (%.1f%%) at %.2f reviews per fraud | AUPRC %.4f -> %.4f | %.0fs",
        summary["frauds_caught"], summary["frauds_present"], 100 * summary["catch_rate"],
        summary["capacity_ratio"], summary["auprc_first"], summary["auprc_final"], elapsed,
    )

    write_json(path, {
        "paper": 1,
        "stage": f"feedback_{stage}",
        "dataset": dataset,
        "arm": arm,
        "seed": seed,
        "config": {
            "review_policy": config.review_policy,
            "pool_rule": config.pool_rule,
            "exploration": config.exploration,
            "band_width": config.band_width,
            "censor_width": config.censor_width,
            "propensity_floor": config.propensity_floor,
            "weight_clip": config.weight_clip,
            "budget_fraction": budget,
            "warmup_rounds": 1,
            "n_rounds": stream.n_rounds,
            "detector": config.detector,
            "detector_tag": tag or None,
        },
        "metrics": summary,
        "rounds": [vars(o) for o in outcomes],
        "runtime_s": elapsed,
    })


def seeds_for(dataset: str) -> list[int]:
    return SEEDS_BY_DATASET.get(dataset, SEEDS[:3])


def stage_main(datasets: list[str]) -> None:
    # Headline arms first, so a sweep that has to be cut short still has the
    # comparison the paper is built on rather than five seeds of one ablation.
    ordered = HEADLINE_ARMS + [a for a in ARMS if a not in HEADLINE_ARMS]
    for dataset in datasets:
        for arm in ordered:
            for seed in seeds_for(dataset):
                execute("main", dataset, arm, ARMS[arm], seed)


def stage_capacity(datasets: list[str]) -> None:
    # Capacity decides how hard the loop bites: analysts who can review a large
    # multiple of the fraud volume confirm nearly all of it and the pool stays
    # clean, while analysts who see a fraction of it record the rest as
    # legitimate. This is the sweep the paper leads with.
    for dataset in datasets[:1]:
        for budget in CAPACITY_GRID:
            for arm in HEADLINE_ARMS:
                for seed in SEEDS[:3]:
                    execute("capacity", dataset, arm, ARMS[arm], seed, budget, f"b{budget:g}")


def stage_censoring(datasets: list[str]) -> None:
    for dataset in datasets[:1]:
        for width in CENSOR_GRID:
            for seed in SEEDS[:3]:
                config = FeedbackConfig("topk", "censor", censor_width=width)
                execute("censoring", dataset, "censor", config, seed, tag=f"c{width:g}")


def stage_detector(datasets: list[str]) -> None:
    # Bank Account Fraud only: the headline claim is measured there, and a
    # full-stream sweep would spend the page budget on a second table the
    # reviewer did not ask for.
    for tag, overrides in DETECTOR_GRID.items():
        for arm in DETECTOR_ARMS:
            for seed in SEEDS[:3]:
                config = replace(ARMS[arm], detector=overrides)
                execute("detector", "baf_base", arm, config, seed, tag=tag)


def stage_robustness(datasets: list[str]) -> None:
    for arm in HEADLINE_ARMS:
        for seed in SEEDS[:3]:
            execute("robustness", ROBUSTNESS_DATASET, arm, ARMS[arm], seed)


STAGES = {
    "main": stage_main,
    "capacity": stage_capacity,
    "censoring": stage_censoring,
    "detector": stage_detector,
    "robustness": stage_robustness,
}


def main() -> None:
    global SUBSAMPLE, RESULTS_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=[*STAGES, "all"])
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument(
        "--subsample", type=float, default=None,
        help="stratified share of rows to keep, for smoke tests only; "
             "results are written to results/paper1/smoke instead",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    if args.subsample:
        SUBSAMPLE = args.subsample
        RESULTS_DIR = PROJECT_ROOT / "results" / "paper1" / "smoke"
        LOGGER.warning("subsampling to %.1f%%; writing to %s", 100 * SUBSAMPLE, RESULTS_DIR)

    for name in ([args.stage] if args.stage != "all" else list(STAGES)):
        STAGES[name](args.datasets)

    LOGGER.info("done; %d feedback result files",
                len(list(RESULTS_DIR.glob("feedback_*.json"))))


if __name__ == "__main__":
    main()
