"""Tables and figures for the Paper 1 feedback-loop study.

Kept apart from the general reporting code because a feedback run is a different
kind of object from a cross-validation fold. There is no held-out test set and no
single metric: each run is a simulated deployment, and what it produced is a
count of fraud caught over a stretch of time, a ranking quality that moves from
round to round, and a training pool whose labels drift away from the truth. The
tables and figures here are built around those three things.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis.aggregate import (  # noqa: E402
    METRIC_LABELS,
    dataset_label,
    load_feedback,
    load_feedback_rounds,
    pretty_arm,
)
from src.analysis.figures import PALETTE, apply_style, save  # noqa: E402
from src.analysis.significance import paired_arms, stars  # noqa: E402
from src.analysis.tables import (  # noqa: E402
    BEEI,
    current_layout,
    fit_body,
    latex_table,
    use_layout,
)

ORACLE = "oracle_full_labels"
NAIVE = "naive_assume_negative"
PROPOSED = "explore_censor"

# The order the arms are presented in: the two bounds first, then the ablation
# of the combined rule, then the alternatives it is being argued against.
ARM_ORDER = [
    ORACLE,
    NAIVE,
    "reviewed_only",
    "explore_uniform",
    "explore_banded",
    "soft_censor",
    "censor",
    "explore_soft_censor",
    "explore_censor",
    "censor_ips",
    "ips",
    "ips_augmented",
]

# The arms the curves are drawn for. More than this and the figures become a
# thicket; these are the 2x2 and the bound.
CURVE_ARMS = [ORACLE, NAIVE, "explore_banded", "censor", PROPOSED]

HEADLINE_METRICS = ["catch_rate", "auprc_mean", "auprc_final", "pool_error_rate"]


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort by the presentation order, keeping anything unrecognised at the end."""
    rank = {arm: position for position, arm in enumerate(ARM_ORDER)}
    return frame.assign(_rank=frame["arm"].map(lambda a: rank.get(a, len(rank))))\
                .sort_values(["_rank", "arm"]).drop(columns="_rank")


def _cell(mean: float, std: float, decimals: int, bold: bool = False) -> str:
    if mean is None or (isinstance(mean, float) and not np.isfinite(mean)):
        return "--"
    body = f"{mean:.{decimals}f}"
    if std is not None and np.isfinite(std):
        body += f" $\\pm$ {std:.{decimals}f}"
    return f"\\textbf{{{body}}}" if bold else body


def _recovery(value: float, naive: float, oracle: float) -> float:
    """Share of the oracle-to-naive gap an arm closes.

    The absolute numbers are governed by how much analyst capacity the scenario
    grants, which is a property of the scenario rather than of the method. What
    a mitigation can be judged on is how much of the loss it recovers, and that
    is comparable across capacities and datasets.
    """
    gap = oracle - naive
    if not np.isfinite(gap) or abs(gap) < 1e-12:
        return np.nan
    return (value - naive) / gap


# --------------------------------------------------------------------------- #
# Tables


def main_table(runs: pd.DataFrame, dataset: str, rounds: pd.DataFrame | None = None) -> str:
    """The headline comparison: every rule, one dataset, fixed capacity.

    The paired test against deployed practice used to be a second table with the
    same rows in the same order. Carrying it as three more columns here means a
    reader checking whether a difference is real does not have to find it.
    """
    subset = runs[runs["dataset"] == dataset]
    if subset.empty:
        return ""

    metrics = [m for m in HEADLINE_METRICS if m in subset.columns]
    grouped = subset.groupby("arm")[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]
    grouped = _ordered(grouped.reset_index())

    reference = grouped.set_index("arm")
    naive_catch = reference.loc[NAIVE, "catch_rate_mean"] if NAIVE in reference.index else np.nan
    oracle_catch = reference.loc[ORACLE, "catch_rate_mean"] if ORACLE in reference.index else np.nan

    # The oracle is an upper bound, not a competitor, so it is excluded from the
    # search for the best achievable cell.
    achievable = grouped[grouped["arm"] != ORACLE]
    winners = {
        metric: (
            achievable[f"{metric}_mean"].idxmin()
            if METRIC_LABELS.get(metric, ("", 4, "max"))[2] == "min"
            else achievable[f"{metric}_mean"].idxmax()
        )
        for metric in metrics
        if f"{metric}_mean" in achievable.columns and achievable[f"{metric}_mean"].notna().any()
    }

    # The paired test, keyed by arm, so the columns line up with the rows above.
    tests: dict[str, dict] = {}
    pairs = 0
    if rounds is not None and not rounds.empty:
        block = rounds[rounds["dataset"] == dataset]
        if not block.empty:
            arms = [a for a in ARM_ORDER if a in set(block["arm"]) and a != NAIVE]
            for outcome in paired_arms(block, NAIVE, arms, metric="auprc"):
                tests[str(outcome["comparator"])] = outcome
                pairs = outcome["n_pairs"]

    rows = []
    for index, record in grouped.iterrows():
        arm = str(record["arm"])
        cells = [pretty_arm(arm)]
        for metric in metrics:
            decimals = METRIC_LABELS.get(metric, (metric, 4, "max"))[1]
            mean = record.get(f"{metric}_mean")
            std = record.get(f"{metric}_std")
            # Fraud caught is a share of the stream; every table in the
            # manuscript reports it as a percentage so the three floats agree.
            if metric == "catch_rate" and mean is not None and np.isfinite(mean):
                mean = 100 * mean
                std = 100 * std if std is not None and np.isfinite(std) else std
                decimals = 1
            cells.append(_cell(mean, std, decimals, bold=winners.get(metric) == index))
        recovered = _recovery(record.get("catch_rate_mean"), naive_catch, oracle_catch)
        cells.append("--" if not np.isfinite(recovered) else f"{100 * recovered:.0f}\\%")
        if tests:
            outcome = tests.get(arm)
            if outcome is None:
                cells += ["--", "--", "--"]
            else:
                cells += [
                    f"{outcome['delta']:+.4f}",
                    f"[{outcome['ci_low']:+.4f}, {outcome['ci_high']:+.4f}]",
                    f"{outcome['p_value']:.4f}{stars(outcome['p_value'])}",
                ]
        rows.append(cells)

    headers = ["Bookkeeping rule"] + [METRIC_LABELS[m][0] for m in metrics] + ["Gap closed"]
    if tests:
        headers += ["$\\Delta$ AUPRC", "95\\% CI", "$p$"]
    n_seeds = int(subset.groupby("arm")["seed"].nunique().max())
    budget = subset["budget_fraction"].dropna().iloc[0] if subset["budget_fraction"].notna().any() else np.nan

    # Carried in the caption rather than in a note block: a threeparttable
    # cannot be scaled to the text width, and with the test columns attached
    # this table needs to be.
    testing = "" if not tests else (
        " The last three columns test per-round AUPRC against deployed practice, "
        f"paired within each (seed, round) cell over {pairs} cells; intervals are "
        "bootstrap percentiles of the mean paired difference and $p$-values come "
        "from a Wilcoxon signed-rank test. "
        "$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$."
    )

    return latex_table(
        headers,
        rows,
        caption=(
            f"Effect of the label bookkeeping rule on {dataset_label(dataset)}, "
            f"replayed over {int(subset['n_rounds'].dropna().max() or 0)} rounds with "
            f"{100 * budget:.1f}\\% of transactions reviewed per round, "
            f"mean $\\pm$ standard deviation over {n_seeds} seeds. "
            "The oracle sees every label and bounds what any rule can achieve; "
            "the deployed-practice row is what a bank gets by recording "
            "unreviewed transactions as legitimate. Gap closed is the share of "
            "the oracle-to-practice gap in fraud caught that the rule recovers."
            + testing
        ),
        label=f"tab:p1-feedback-{dataset.replace('_', '-')}",
    )


DETECTOR_ORDER = [
    "default", "trees200", "trees1000", "lr002", "lr010", "leaves31", "leaves127",
]
DETECTOR_LABELS = {
    "default": "Default (500, 0.05, 63)",
    "trees200": "200 trees",
    "trees1000": "1{,}000 trees",
    "lr002": "Learning rate 0.02",
    "lr010": "Learning rate 0.10",
    "leaves31": "31 leaves",
    "leaves127": "127 leaves",
}


def _closed_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    pct = 100 * value
    body = f"{pct:.0f}\\%"
    return f"${pct:.0f}\\%$" if pct < 0 else body


def _gap_closed(frame: pd.DataFrame) -> dict[str, float]:
    """Share of the oracle-to-practice gap each arm closes, from means."""
    means = frame.groupby("arm")["catch_rate"].mean()
    naive = means.get(NAIVE, np.nan)
    oracle = means.get(ORACLE, np.nan)
    return {arm: _recovery(value, naive, oracle) for arm, value in means.items()}


def _detector_panel(
    detector: pd.DataFrame | None,
    default_runs: pd.DataFrame | None,
    columns: int,
) -> list[list[str]]:
    """Rows of panel (c): recovered gap at each LightGBM setting."""
    if detector is None or detector.empty or "detector_tag" not in detector.columns:
        return []
    blocks: dict[str, pd.DataFrame] = {}
    if default_runs is not None and not default_runs.empty:
        blocks["default"] = default_runs
    for tag, block in detector.groupby("detector_tag"):
        if tag:
            blocks[str(tag)] = block
    if len(blocks) <= (1 if "default" in blocks else 0):
        return []

    headers = ["Setting", "Hard catch (\\%)", "Hard closed",
               "Soft catch (\\%)", "Soft closed"]
    pad = [""] * max(columns - len(headers), 0)
    rows = [headers[:columns] + pad]
    for tag in DETECTOR_ORDER:
        if tag not in blocks:
            continue
        frame = blocks[tag]
        closed = _gap_closed(frame)
        catch = frame.groupby("arm")["catch_rate"].agg(["mean", "std"])
        hard = catch.loc["censor"] if "censor" in catch.index else None
        soft = catch.loc["soft_censor"] if "soft_censor" in catch.index else None
        cells = [
            DETECTOR_LABELS.get(tag, tag),
            "--" if hard is None else _cell(100 * hard["mean"], 100 * hard["std"], 1),
            _closed_pct(closed.get("censor", np.nan)),
            "--" if soft is None else _cell(100 * soft["mean"], 100 * soft["std"], 1),
            _closed_pct(closed.get("soft_censor", np.nan)),
        ]
        rows.append(cells[:columns] + pad)
    return rows if len(rows) > 1 else []


def sweep_table(
    capacity: pd.DataFrame,
    censoring: pd.DataFrame,
    detector: pd.DataFrame | None = None,
    default_runs: pd.DataFrame | None = None,
) -> str:
    """Both censoring sweeps in one float, as two panels sharing a column grid.

    They were separate tables and are read together: panel (a) asks how much
    review capacity the rule needs before it stops helping, panel (b) asks how
    wide the censored band should be. Assembled by hand rather than through
    ``latex_table`` because the two panels label their columns differently,
    which is the one thing that function does not express.
    """
    capacity = capacity.dropna(subset=["budget_fraction"]) if not capacity.empty else capacity
    if capacity.empty:
        return ""

    budgets = sorted(capacity["budget_fraction"].unique())
    arms = [a for a in ARM_ORDER if a in set(capacity["arm"])]
    grouped = capacity.groupby(["arm", "budget_fraction"])["catch_rate"].agg(["mean", "std"])
    ratios = capacity.groupby("budget_fraction")["capacity_ratio"].mean()

    layout = current_layout()
    columns = 1 + len(budgets)

    # The rows are collected as cells before being joined, so the tabular can be
    # measured column by column and sized to the target page. The two panels
    # share a column grid, and panel (b)'s cells are the wider ones, so a
    # row-wise glance at panel (a) alone understates the table badly.
    grid: list[list[str]] = [
        ["Bookkeeping rule"] + [f"{ratios.get(b, float('nan')):.2f}" for b in budgets]
    ]
    for arm in arms:
        cells = [pretty_arm(arm)]
        for budget in budgets:
            if (arm, budget) not in grouped.index:
                cells.append("--")
                continue
            record = grouped.loc[(arm, budget)]
            cells.append(_cell(100 * record["mean"], 100 * record["std"], 1))
        grid.append(cells)

    panel_a = len(grid)

    widths = censoring.dropna(subset=["censor_width"]) if (
        not censoring.empty and "censor_width" in censoring.columns
    ) else pd.DataFrame()
    if not widths.empty:
        summary = widths.groupby("censor_width")[
            ["catch_rate", "auprc_mean", "pool_error_rate", "pool_size_final"]
        ].agg(["mean", "std"])
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]

        panel_headers = ["Width ($\\times$ budget)", "Fraud caught (\\%)", "AUPRC",
                         "Pool label error", "Pool rows"]
        # Pad to the grid set by panel (a), which has as many columns as the
        # capacity sweep has budgets.
        pad = [""] * max(columns - len(panel_headers), 0)
        grid.append(panel_headers[:columns] + pad)
        for width, record in summary.iterrows():
            cells = [
                f"{width:g}",
                _cell(100 * record["catch_rate_mean"], 100 * record["catch_rate_std"], 1),
                _cell(record["auprc_mean_mean"], record["auprc_mean_std"], 4),
                _cell(record["pool_error_rate_mean"], record["pool_error_rate_std"], 4),
                f"{record['pool_size_final_mean']:,.0f}",
            ]
            grid.append(cells[:columns] + pad)

    panel_b = len(grid)

    detector_rows = _detector_panel(detector, default_runs, columns)
    if detector_rows:
        grid.extend(detector_rows)

    sizing = fit_body(grid)
    boxed = any(directive.startswith("\\resizebox") for directive in sizing)

    caption = (
        "Sweeps over the censoring family and the detector. (a) Fraud caught "
        "against review capacity in reviews per fraud; budgets are "
        + ", ".join(f"{100 * b:.1f}\\%" for b in budgets)
        + " of transactions per round. (b) Hard censoring against censored-band "
        "width, as a multiple of the review budget."
        + (
            " (c) Share of the oracle-to-practice gap recovered by hard and "
            "soft censoring as one LightGBM hyperparameter moves around the "
            "default of 500 trees, learning rate 0.05 and 63 leaves."
            if detector_rows else ""
        )
        + " Mean $\\pm$ standard deviation over three seeds throughout."
    )

    lines = [
        f"\\begin{{{layout.wide_environment}}}[{layout.placement}]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:p1-feedback-sweeps}",
        *(sizing or ["\\small"]),
        "\\begin{tabular}{l" + "c" * (columns - 1) + "}",
        "\\toprule",
        f"\\multicolumn{{{columns}}}{{l}}{{\\textit{{(a) Fraud caught (\\%) by "
        f"review capacity, reviews per fraud}}}} \\\\",
        "\\midrule",
        " & ".join(grid[0]) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(row) + " \\\\" for row in grid[1:panel_a]]
    if panel_b > panel_a:
        lines += [
            "\\midrule",
            f"\\multicolumn{{{columns}}}{{l}}{{\\textit{{(b) Hard censoring by "
            f"width of the censored band}}}} \\\\",
            "\\midrule",
            " & ".join(grid[panel_a]) + " \\\\",
            "\\midrule",
        ]
        lines += [" & ".join(row) + " \\\\" for row in grid[panel_a + 1:panel_b]]
    if detector_rows:
        lines += [
            "\\midrule",
            f"\\multicolumn{{{columns}}}{{l}}{{\\textit{{(c) Gap closed (\\%) by "
            f"detector setting}}}} \\\\",
            "\\midrule",
            " & ".join(grid[panel_b]) + " \\\\",
            "\\midrule",
        ]
        lines += [" & ".join(row) + " \\\\" for row in grid[panel_b + 1:]]

    lines += ["\\bottomrule", "\\end{tabular}"]
    if boxed:
        lines.append("}")
    lines.append(f"\\end{{{layout.wide_environment}}}")
    return "\n".join(lines) + "\n"


def cross_dataset_table(runs: pd.DataFrame) -> str:
    """Every rule on every stream, which is what shows the ordering is unstable.

    This replaced three per-stream tables of the same shape. The full metric
    set with its spreads stays on Bank Account Fraud in
    Table~\\ref{tab:p1-feedback-baf-base}; here the two outcomes that carry the
    argument are enough, and putting the streams beside each other is what makes
    the reversals visible at all.
    """
    if runs.empty:
        return ""
    arms = [a for a in ARM_ORDER if a in set(runs["arm"])]
    datasets = [d for d in ["baf_base", "ieee_cis", "ulb_creditcard", "baf_variant3"]
                if d in set(runs["dataset"])]
    if not datasets:
        return ""

    grouped = runs.groupby(["arm", "dataset"])[["catch_rate", "auprc_mean"]].agg(["mean", "std"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]

    rows = []
    for arm in arms:
        cells = [pretty_arm(arm)]
        for dataset in datasets:
            key = (arm, dataset)
            if key not in grouped.index:
                cells += ["--", "--"]
                continue
            record = grouped.loc[key]
            cells.append(_cell(100 * record["catch_rate_mean"],
                               100 * record["catch_rate_std"], 1))
            cells.append(_cell(record["auprc_mean_mean"], record["auprc_mean_std"], 4))
        rows.append(cells)

    return latex_table(
        ["Bookkeeping rule"] + ["Caught (\\%)", "AUPRC"] * len(datasets),
        rows,
        caption=(
            "Every rule on every stream at the default review budget of 0.5\\% "
            "of transactions per round. The same budget buys very different "
            "capacities --- 0.14 reviews per fraud on IEEE-CIS, 0.46 on Bank "
            "Account Fraud and 3.20 on ULB --- so these are points on a "
            "capacity curve rather than replications. The two censoring rules "
            "are the only ones that improve on deployed practice everywhere. "
            "Fraud caught is a percentage of all fraud in the stream and AUPRC "
            "is the mean over rounds; both are mean $\\pm$ standard deviation "
            "over seeds. The full metric set with paired tests is "
            "in Table~\\ref{tab:p1-feedback-baf-base} for Bank Account Fraud and "
            "in the released runs for the rest. BAF (variant III) is the harder "
            "released variant, replayed under an identical protocol."
        ),
        label="tab:p1-feedback-datasets",
        column_format="l" + "cc" * len(datasets),
        groups=[("", 1)] + [(dataset_label(d), 2) for d in datasets],
    )


# --------------------------------------------------------------------------- #
# Figures


def _arm_colour(arm: str) -> str:
    order = [ORACLE, NAIVE, PROPOSED, "censor", "explore_banded", "soft_censor",
             "explore_soft_censor", "reviewed_only", "ips", "ips_augmented",
             "censor_ips", "explore_uniform"]
    return PALETTE[order.index(arm) % len(PALETTE)] if arm in order else "#95a5a6"


def _arm_style(arm: str) -> dict:
    if arm == ORACLE:
        return {"linestyle": "--", "linewidth": 1.4}
    if arm == PROPOSED:
        return {"linestyle": "-", "linewidth": 2.2}
    return {"linestyle": "-", "linewidth": 1.5}


def figure_degradation(rounds: pd.DataFrame, dataset: str, output_dir: Path,
                       arms: list[str] | None = None) -> None:
    """Ranking quality and pool contamination, round by round.

    The two panels are the argument in miniature: the pool of the deployed rule
    fills up with transactions labelled legitimate that were not, and its
    ability to rank the next round's fraud falls away as it does.
    """
    subset = rounds[rounds["dataset"] == dataset]
    if subset.empty:
        return

    arms = arms or [a for a in CURVE_ARMS if a in set(subset["arm"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))

    for arm in arms:
        rows = subset[subset["arm"] == arm]
        if rows.empty:
            continue
        style = _arm_style(arm)
        colour = _arm_colour(arm)

        curve = rows.groupby("round_index")["auprc"].agg(["mean", "std"])
        axes[0].plot(curve.index, curve["mean"], color=colour,
                     label=pretty_arm(arm), **style)
        axes[0].fill_between(curve.index, curve["mean"] - curve["std"].fillna(0),
                             curve["mean"] + curve["std"].fillna(0),
                             color=colour, alpha=0.12, linewidth=0)

        contamination = rows.groupby("round_index")["pool_error_rate"].mean()
        axes[1].plot(contamination.index, 100 * contamination, color=colour, **style)

    axes[0].set_xlabel("Retraining round")
    axes[0].set_ylabel("AUPRC on the round's traffic")
    axes[0].set_title("Ranking quality")
    axes[1].set_xlabel("Retraining round")
    axes[1].set_ylabel("Pool rows with a wrong label (%)")
    axes[1].set_title("Training pool contamination")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(dataset_label(dataset), fontsize=10)
    fig.tight_layout()
    save(fig, output_dir, f"p1_feedback_degradation_{dataset}")


def figure_cumulative_caught(rounds: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    """Fraud caught as the deployment runs, which is the operational bottom line."""
    subset = rounds[rounds["dataset"] == dataset]
    if subset.empty:
        return

    arms = [a for a in CURVE_ARMS if a in set(subset["arm"])]
    fig, axis = plt.subplots(figsize=(4.6, 3.2))
    for arm in arms:
        rows = subset[subset["arm"] == arm]
        curve = rows.groupby("round_index")["cumulative_caught"].mean()
        axis.plot(curve.index, curve.to_numpy(), color=_arm_colour(arm),
                  label=pretty_arm(arm), **_arm_style(arm))

    axis.set_xlabel("Retraining round")
    axis.set_ylabel("Cumulative fraud caught")
    axis.set_title(f"{dataset_label(dataset)}: fraud caught over the deployment")
    axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, output_dir, f"p1_feedback_cumulative_{dataset}")


def figure_capacity(runs: pd.DataFrame, output_dir: Path) -> None:
    """How the damage and its remedy scale with analyst capacity."""
    subset = runs.dropna(subset=["budget_fraction"])
    if subset.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    arms = [a for a in ARM_ORDER if a in set(subset["arm"])]

    for arm in arms:
        rows = subset[subset["arm"] == arm]
        curve = rows.groupby("capacity_ratio")[["catch_rate", "auprc_mean"]].mean()
        if curve.empty:
            continue
        axes[0].plot(curve.index, 100 * curve["catch_rate"], marker="o", markersize=3,
                     color=_arm_colour(arm), label=pretty_arm(arm), **_arm_style(arm))
        axes[1].plot(curve.index, curve["auprc_mean"], marker="o", markersize=3,
                     color=_arm_colour(arm), **_arm_style(arm))

    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Analyst capacity (reviews per fraud)")
    axes[0].set_ylabel("Fraud caught (%)")
    axes[0].set_title("Detection")
    axes[1].set_ylabel("AUPRC, mean over rounds")
    axes[1].set_title("Ranking quality")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, output_dir, "p1_feedback_capacity")


def figure_gap_closed(runs: pd.DataFrame, output_dir: Path) -> None:
    """Share of the oracle-to-practice gap each rule recovers, per dataset."""
    if runs.empty:
        return
    datasets = [d for d in ["baf_base", "ieee_cis", "ulb_creditcard"] if d in set(runs["dataset"])]
    arms = [a for a in ARM_ORDER if a in set(runs["arm"]) and a not in (ORACLE, NAIVE)]
    if not datasets or not arms:
        return

    means = runs.groupby(["dataset", "arm"])["catch_rate"].mean()
    fig, axis = plt.subplots(figsize=(6.2, 3.2))
    width = 0.8 / max(len(datasets), 1)

    for offset, dataset in enumerate(datasets):
        naive = means.get((dataset, NAIVE), np.nan)
        oracle = means.get((dataset, ORACLE), np.nan)
        values = [100 * _recovery(means.get((dataset, arm), np.nan), naive, oracle)
                  for arm in arms]
        positions = np.arange(len(arms)) + offset * width
        axis.bar(positions, values, width=width, label=dataset_label(dataset),
                 color=PALETTE[offset % len(PALETTE)])

    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(arms)) + width * (len(datasets) - 1) / 2)
    axis.set_xticklabels([pretty_arm(a) for a in arms], rotation=30, ha="right", fontsize=7)
    axis.set_ylabel("Gap to the oracle closed (%)")
    axis.set_title("How much of the loss each rule recovers")
    axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, output_dir, "p1_feedback_gap_closed")


# --------------------------------------------------------------------------- #


def build(tables_dir: Path, figures_dir: Path, runs_dir: Path | None = None) -> dict[str, str]:
    """Write every feedback table and figure that the available runs support."""
    # These are Paper 1's tables, so they follow Paper 1 to BEEI. Without this
    # they would keep emitting two-column IEEE floats into a single-column
    # class, where table* still typesets but loses its placement.
    with use_layout(BEEI):
        return _build(tables_dir, figures_dir, runs_dir)


def _build(tables_dir: Path, figures_dir: Path, runs_dir: Path | None = None) -> dict[str, str]:
    apply_style()

    main = load_feedback("main", runs_dir)
    main_rounds = load_feedback_rounds("main", runs_dir)
    capacity = load_feedback("capacity", runs_dir)
    censoring = load_feedback("censoring", runs_dir)
    detector = load_feedback("detector", runs_dir)
    robustness = load_feedback("robustness", runs_dir)

    tables: dict[str, str] = {}
    if not main.empty:
        # Only Bank Account Fraud gets the full metric set. The other streams
        # used to get an identical table each, which cost three floats to make
        # a comparison the cross-stream table makes better in one.
        body = main_table(main, "baf_base", main_rounds)
        if body:
            tables["paper1_feedback_baf_base"] = body

        # Variant III is the same experiment on a harder stream, so it belongs
        # beside the others rather than in a table of its own.
        streams = main if robustness.empty else pd.concat([main, robustness], ignore_index=True)
        body = cross_dataset_table(streams)
        if body:
            tables["paper1_feedback_datasets"] = body

    default = main[main["dataset"] == "baf_base"] if not main.empty else None
    body = sweep_table(capacity, censoring, detector, default)
    if body:
        tables["paper1_feedback_sweeps"] = body

    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, body in tables.items():
        (tables_dir / f"{name}.tex").write_text(body, encoding="utf-8")

    if not main_rounds.empty:
        for dataset in sorted(main_rounds["dataset"].dropna().unique()):
            figure_degradation(main_rounds, dataset, figures_dir)
            figure_cumulative_caught(main_rounds, dataset, figures_dir)
    if not main.empty:
        figure_gap_closed(main, figures_dir)
    if not capacity.empty:
        figure_capacity(capacity, figures_dir)

    return tables
