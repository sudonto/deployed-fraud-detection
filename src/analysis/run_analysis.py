"""Regenerate Paper 1 tables, figures and macros from the stored runs.

    python -m src.analysis.run_analysis --paper 1
"""

from __future__ import annotations

import argparse
import json
import logging

from src.analysis.aggregate import PAPER1_HEADLINE, load_runs, summarise
from src.analysis.feedback_report import build as build_feedback_report
from src.analysis.figures import build_paper1_figures
from src.analysis.macros import build_paper1_macros
from src.analysis.tables import BEEI, build_paper1_tables, use_layout
from src.common.io import PROJECT_ROOT, write_json

LOGGER = logging.getLogger("analysis")


def attempt(description, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:  # noqa: BLE001
        LOGGER.error("%s failed: %s: %s", description, type(error).__name__, error)
        return None


def paths_for(paper: int):
    root = PROJECT_ROOT / "results" / f"paper{paper}"
    return root / "tables", root / "figures", root / "summary"


def summarise_paper(paper: int) -> dict:
    runs = load_runs(paper)
    if runs.empty:
        return {}
    digest = {"n_runs": int(len(runs)), "stages": {}}
    for stage, frame in runs.groupby("stage"):
        summary = summarise(frame, PAPER1_HEADLINE, group=["arm"])
        digest["stages"][str(stage)] = json.loads(summary.to_json(orient="records"))
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate Paper 1 artefacts")
    parser.add_argument("--paper", type=int, choices=[1], default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tables_dir, figures_dir, summary_dir = paths_for(args.paper)
    with use_layout(BEEI):
        tables = attempt("paper 1 tables", build_paper1_tables, tables_dir) or {}
        feedback = attempt(
            "paper 1 feedback report", build_feedback_report, tables_dir, figures_dir,
        ) or {}
        tables.update(feedback)
    written = [name for name, content in tables.items() if content]
    LOGGER.info("paper 1: %d tables -> %s", len(written), tables_dir)
    attempt("paper 1 figures", build_paper1_figures, figures_dir)
    attempt("paper 1 macros", build_paper1_macros, tables_dir / "macros.tex")
    digest = attempt("paper 1 digest", summarise_paper, args.paper)
    if digest:
        write_json(summary_dir / "summary.json", digest)


if __name__ == "__main__":
    main()
