"""Download and verify every dataset used by the two papers.

Run this once before any experiment:

    python -m scripts.fetch_datasets
    python -m scripts.fetch_datasets baf_base ieee_cis

Each dataset is checked against the row count, feature count and class balance
recorded in its specification. A mismatch aborts the run rather than silently
producing results that cannot be compared with the published literature.

Naming datasets on the command line restricts the run to those, which matters
for the million-row ones: they take minutes to fetch and parse, and there is no
reason to touch them again once they are cached.
"""

from __future__ import annotations

import logging
import sys

from src.common.io import DATASETS, PROJECT_ROOT, load_dataset, write_json


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summaries = {}
    failures = []

    requested = sys.argv[1:] or list(DATASETS)
    unknown = [k for k in requested if k not in DATASETS]
    if unknown:
        logging.error("unknown dataset(s): %s", ", ".join(unknown))
        return 1

    for key in requested:
        spec = DATASETS[key]
        try:
            dataset = load_dataset(key)
        except Exception as exc:  # noqa: BLE001 - reported, then re-raised as a summary
            failures.append((key, str(exc)))
            logging.error("%s FAILED: %s", key, exc)
            continue

        summary = dataset.summary()
        summary["openml_id"] = spec.openml_id
        summaries[key] = summary
        logging.info(
            "%-22s %7d rows  %3d features  %6d positives (%.4f%%)  IR 1:%.0f",
            key,
            summary["n_samples"],
            summary["n_features"],
            summary["n_positive"],
            summary["minority_pct"],
            summary["imbalance_ratio"],
        )

    write_json(PROJECT_ROOT / "data" / "dataset_summary.json", summaries)

    if failures:
        logging.error("%d dataset(s) failed verification", len(failures))
        return 1
    logging.info("all %d datasets verified", len(summaries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
