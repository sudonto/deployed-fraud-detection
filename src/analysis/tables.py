"""LaTeX table generation, driven entirely by the raw run files."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.aggregate import (
    DATASET_LABELS,
    METRIC_LABELS,
    best_arm,
    format_mean_std,
    load_runs,
    per_fold_matrix,
    pretty_arm,
    summarise,
)
from src.analysis.literature import (
    PAPER1_LITERATURE,
    PAPER2_LITERATURE,
    LiteratureResult,
)
from src.analysis.significance import stars
from src.common.io import PROJECT_ROOT
from src.common.stats import friedman_nemenyi


_MARKUP = re.compile(r"\\[a-zA-Z]+|[{}$\\]")


@dataclass(frozen=True)
class Layout:
    """How wide the target page is, and which float machinery it tolerates.

    The character counts are approximate on purpose: they only have to decide
    between plain, shrunk, and scaled, and being off by a few characters costs
    a font step rather than a broken page.

    Four targets, because the two papers no longer go to the same place.
    IEEEtran is two columns, so a table either fits a column or is promoted to
    a full-width float. BEEI is one column of 15.5 cm, so there is no promotion
    to make and the template asks for 8 pt bodies pinned in place with ``[H]``.
    IJIES is two columns again but a little narrower, and asks for 10 pt inside
    tables; ``ijies.tex`` imposes that on the float as a whole, so what is
    needed here is only the honest character budget to decide against. The docx
    path refuses ``\\resizebox`` and ``threeparttable`` outright, since pandoc
    has no equivalent for either and drops them without saying so.
    """

    column_chars: int
    text_chars: int
    wide_environment: str
    placement: str
    resizebox: bool = True
    threeparttable: bool = True
    cmidrule: bool = True
    body_size: str | None = None


IEEETRAN = Layout(column_chars=48, text_chars=100, wide_environment="table*",
                  placement="!t")
BEEI = Layout(column_chars=110, text_chars=110, wide_environment="table",
              placement="H", body_size="\\fontsize{8pt}{10pt}\\selectfont")
# 82.2 mm and 172 mm at 10 pt Times, whose figures are exactly half an em wide,
# so a character costs 5 pt: 47 to a column, 98 to the page. Two tables of
# six-fold means and standard deviations exceed 98 whatever is done to the
# column separation, and \resizebox is what stops them running into the margin;
# those two end up nearer 8 pt than the 10 pt the template asks for, which is
# the one place Paper 2 knowingly departs from it.
IJIES = Layout(column_chars=47, text_chars=98, wide_environment="table*",
               placement="!t")
PLAIN = Layout(column_chars=10**6, text_chars=10**6, wide_environment="table",
               placement="!t", resizebox=False, threeparttable=False,
               cmidrule=False)

LAYOUT = IEEETRAN


def fit_body(rows: list[list[str] | str]) -> list[str]:
    """The sizing directives a hand-assembled tabular of ``rows`` needs.

    The same decision ``latex_table`` makes, factored out for the two tables
    that are assembled by hand because their panels label columns differently.
    Without it those tables set at natural size and silently overrun a narrow
    column.
    """
    if table_width(rows) <= LAYOUT.text_chars:
        return [LAYOUT.body_size] if LAYOUT.body_size else []
    if LAYOUT.resizebox:
        return ["\\resizebox{\\textwidth}{!}{%"]
    return ["\\footnotesize", "\\setlength{\\tabcolsep}{4pt}"]


def current_layout() -> Layout:
    """The active layout, for the tables assembled outside ``latex_table``.

    A plain ``from ... import LAYOUT`` would bind the default once at import
    time and never see ``use_layout`` change it, which is the sort of bug that
    shows up as one stray two-column float in an otherwise converted paper.
    """
    return LAYOUT


@contextmanager
def use_layout(layout: Layout):
    """Generate tables for one target, then put the default back.

    A module-level setting rather than an argument on every call because the
    choice belongs to the paper, not to the table, and threading it through
    fifteen call sites would invite one of them to disagree with the rest.
    """
    global LAYOUT
    previous, LAYOUT = LAYOUT, layout
    try:
        yield
    finally:
        LAYOUT = previous


_SPECIALS = {"&": "\\&", "%": "\\%", "#": "\\#", "_": "\\_"}


def escape(text: str) -> str:
    """Make plain prose safe to drop into a table cell.

    Only the characters that appear in study names and protocol descriptions are
    handled. Anything that has to carry real markup is passed through by the
    caller instead of going through here.
    """
    return "".join(_SPECIALS.get(character, character) for character in text)


def table_width(rows: list[list[str] | str]) -> int:
    """How wide the tabular will set, in characters, plus column padding.

    Summing the maxima column by column rather than taking the widest row,
    because LaTeX sizes each column independently: a table whose longest labels
    and longest numbers live in different rows is wider than any row in it. The
    row-wise estimate missed that and let a table overrun the page by a
    centimetre while every row it looked at fitted.

    A row given as a bare string is a rule or a spanning heading, which sits in
    one cell and does not set any column's width, so it is skipped.
    """
    widths: list[int] = []
    for row in rows:
        if isinstance(row, str):
            continue
        for column, cell in enumerate(row):
            plain = len(_MARKUP.sub("", cell))
            if column < len(widths):
                widths[column] = max(widths[column], plain)
            else:
                widths.append(plain)
    return sum(widths) + 2 * len(widths)


def latex_table(
    headers: list[str],
    rows: list[list[str] | str],
    caption: str,
    label: str,
    column_format: str | None = None,
    notes: str | None = None,
    groups: list[tuple[str, int]] | None = None,
) -> str:
    """Assemble a booktabs table. Cell contents are passed through unchanged.

    The layout is chosen by measuring the tabular against the active ``LAYOUT``:
    a table that fits one column stays in one, a wider one is promoted to the
    target's wide float, and one wider still is scaled to the text width.
    Choosing here rather than at the call sites means a table that grows a
    column when an experiment is added does not silently start running off the
    page.

    A table carrying notes is wrapped in threeparttable, which is what makes the
    note block the width of the tabular above it rather than of the column.
    Such a table is never scaled with \\resizebox: threeparttable measures its
    own width by typesetting the body, a resized box reports the width it was
    told to be rather than the width it needs, and the note block underneath is
    then set too wide and runs into the margin. Shrinking the font instead keeps
    the measurement honest.

    ``groups`` adds a spanning row above the headers, given as (label, width)
    pairs covering every column in order. It is what lets one table carry the
    same metrics for several datasets side by side instead of one table each.
    """
    column_format = column_format or "l" + "c" * (len(headers) - 1)
    width = table_width([headers, *rows])
    wide = width > LAYOUT.column_chars
    scaled = width > LAYOUT.text_chars

    environment = LAYOUT.wide_environment if wide else "table"
    noted = bool(notes) and LAYOUT.threeparttable
    if notes and not noted:
        # Nothing to hang a note block from, so it joins the caption rather
        # than becoming a stray paragraph inside the float.
        caption = f"{caption} {notes}"
    lines = [
        f"\\begin{{{environment}}}[{LAYOUT.placement}]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
    ]
    boxed = scaled and not noted and LAYOUT.resizebox
    if noted:
        lines.append("\\begin{threeparttable}")
    if boxed:
        lines.append("\\resizebox{\\textwidth}{!}{%")
    elif scaled:
        lines += ["\\footnotesize", "\\setlength{\\tabcolsep}{4pt}"]
    elif LAYOUT.body_size:
        lines.append(LAYOUT.body_size)
    elif wide:
        lines.append("\\small")
    lines += [f"\\begin{{tabular}}{{{column_format}}}", "\\toprule"]
    if groups:
        cells, rules, column = [], [], 1
        for title, span in groups:
            cells.append(title if span == 1 else f"\\multicolumn{{{span}}}{{c}}{{{title}}}")
            if span > 1 and title and LAYOUT.cmidrule:
                rules.append(f"\\cmidrule(lr){{{column}-{column + span - 1}}}")
            column += span
        lines.append(" & ".join(cells) + " \\\\")
        if rules:
            lines.append(" ".join(rules))
    lines += [" & ".join(headers) + " \\\\", "\\midrule"]
    lines.extend(
        row if isinstance(row, str) else " & ".join(row) + " \\\\" for row in rows
    )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if boxed:
        lines.append("}")
    if noted:
        lines.append(f"\\begin{{tablenotes}}\\small\\item {notes}\\end{{tablenotes}}")
        lines.append("\\end{threeparttable}")
    lines.append(f"\\end{{{environment}}}")
    return "\n".join(lines) + "\n"


def metric_headers(metrics: list[str]) -> list[str]:
    return [METRIC_LABELS.get(m, (m, 4, "max"))[0] for m in metrics]


def summary_rows(
    summary: pd.DataFrame,
    metrics: list[str],
    highlight: bool = True,
    label_column: str = "arm",
) -> list[list[str]]:
    """One row per arm, with the winning cell in each metric column bolded."""
    winners = {m: best_arm(summary, m) for m in metrics} if highlight else {}
    rows = []
    for _, record in summary.iterrows():
        cells = [pretty_arm(str(record[label_column]))]
        for metric in metrics:
            mean_key, std_key = f"{metric}_mean", f"{metric}_std"
            if mean_key not in summary.columns:
                cells.append("--")
                continue
            decimals = METRIC_LABELS.get(metric, (metric, 4, "max"))[1]
            cells.append(
                format_mean_std(
                    record.get(mean_key),
                    record.get(std_key),
                    decimals,
                    bold=winners.get(metric) == record[label_column],
                )
            )
        rows.append(cells)
    return rows


def cross_dataset_table(
    runs: pd.DataFrame,
    stage: str,
    metrics: list[str],
    arm_order: list[str],
    first_header: str,
    caption: str,
    label: str,
    notes: str | None = None,
) -> str:
    """One stage, with the datasets set side by side rather than stacked.

    Three tables of identical shape make the reader hold three captions and
    three headers in mind to perform one comparison, and each of them spans
    both columns of the page. Putting the datasets in column groups asks for
    none of that and costs a third of the space.

    Arms absent from a dataset get a dash rather than a missing row, so the
    reader can see that a configuration was not run there instead of having to
    infer it from the row count.
    """
    frame = runs[runs["stage"] == stage]
    if frame.empty:
        return ""

    order = {name: index for index, name in enumerate(DATASET_LABELS)}
    datasets = sorted(set(frame["dataset"].astype(str)),
                      key=lambda d: order.get(d, len(order)))

    summaries = {d: summarise(frame[frame["dataset"] == d], metrics) for d in datasets}
    winners = {
        (dataset, metric): best_arm(summary, metric)
        for dataset, summary in summaries.items()
        for metric in metrics
    }

    present = set()
    for summary in summaries.values():
        present |= set(summary["arm"].astype(str))
    arms = [a for a in arm_order if a in present] + sorted(present - set(arm_order))

    rows = []
    for arm in arms:
        cells = [pretty_arm(arm)]
        for dataset in datasets:
            summary = summaries[dataset]
            match = summary[summary["arm"].astype(str) == arm]
            for metric in metrics:
                mean_key = f"{metric}_mean"
                if match.empty or mean_key not in summary.columns:
                    cells.append("--")
                    continue
                record = match.iloc[0]
                cells.append(format_mean_std(
                    record.get(mean_key),
                    record.get(f"{metric}_std"),
                    METRIC_LABELS.get(metric, (metric, 4, "max"))[1],
                    bold=winners[(dataset, metric)] == arm,
                ))
        rows.append(cells)

    return latex_table(
        [first_header] + metric_headers(metrics) * len(datasets),
        rows,
        caption=caption,
        label=label,
        column_format="l" + "c" * (len(metrics) * len(datasets)),
        notes=notes,
        groups=[("", 1)] + [(DATASET_LABELS.get(d, d), len(metrics)) for d in datasets],
    )


def order_by(summary: pd.DataFrame, metric: str, preferred: list[str] | None = None) -> pd.DataFrame:
    """Sort arms by a metric, optionally forcing a fixed order first."""
    if summary.empty:
        return summary
    if preferred:
        rank = {arm: i for i, arm in enumerate(preferred)}
        summary = summary.copy()
        summary["_rank"] = summary["arm"].map(lambda a: rank.get(a, len(rank)))
        column = f"{metric}_mean"
        ascending = [True, False] if column in summary.columns else [True]
        by = ["_rank", column] if column in summary.columns else ["_rank"]
        return summary.sort_values(by, ascending=ascending).drop(columns="_rank")
    column = f"{metric}_mean"
    if column in summary.columns:
        return summary.sort_values(column, ascending=False)
    return summary


# --------------------------------------------------------------------------- #
# Literature tables
# --------------------------------------------------------------------------- #
def _spanning(entries: list[LiteratureResult], metric: str, limit: int) -> list[LiteratureResult]:
    """Thin a literature list down to entries spanning the reported range.

    The point of these tables is that published figures on one benchmark differ
    by more than the methods do, and that point is made by the extremes and a
    few points between them. Listing every row a search returned makes it at the
    cost of a third of a page, so the extremes are kept and the interior is
    sampled evenly.

    The limit is per dataset, since a table covering three benchmarks must keep
    the range of each rather than the range of whichever has the highest
    reported numbers.
    """
    kept: list[LiteratureResult] = []
    for dataset in dict.fromkeys(e.dataset for e in entries):
        block = [e for e in entries if e.dataset == dataset]
        scored = [e for e in block if e.metrics.get(metric) is not None]
        if len(block) <= limit or len(scored) < limit:
            kept += block
            continue
        scored.sort(key=lambda e: e.metrics[metric])
        positions = np.linspace(0, len(scored) - 1, limit).round().astype(int)
        keep = {id(scored[p]) for p in dict.fromkeys(positions)}
        kept += [e for e in block if id(e) in keep]
    return kept


def literature_table(
    entries: list[LiteratureResult],
    metrics: list[str],
    caption: str,
    label: str,
    limit: int | None = None,
) -> str:
    if limit:
        entries = _spanning(entries, metrics[0], limit)
    headers = ["Study", "Model", "Protocol"] + metric_headers(metrics)
    rows = []
    for entry in sorted(entries, key=lambda e: (e.dataset, -e.year, e.model)):
        protocol = escape(entry.protocol)
        if entry.leaky:
            protocol += " \\textsuperscript{\\dag}"
        cells = [f"{escape(entry.study)} ({entry.year})", escape(entry.model), protocol]
        for metric in metrics:
            value = entry.metrics.get(metric)
            decimals = METRIC_LABELS.get(metric, (metric, 4, "max"))[1]
            cells.append("--" if value is None else f"{value:.{decimals}f}")
        rows.append(cells)
    return latex_table(
        headers, rows, caption, label,
        column_format="p{2.8cm}p{2.6cm}p{3.4cm}" + "c" * len(metrics),
        notes=(
            "\\textsuperscript{\\dag} Reported under a protocol in which resampling or "
            "augmentation precedes the train/test split, so the figure is not directly "
            "comparable with the leakage-free results in this paper."
        ),
    )


# --------------------------------------------------------------------------- #
# Paper 1
# --------------------------------------------------------------------------- #
PAPER1_METRICS = ["auprc", "mcc", "f1", "precision", "recall", "roc_auc"]
BASELINE_ORDER = [
    "logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm",
    "catboost", "mlp", "cnn1d", "bilstm", "autoencoder",
]
# SMOTE-Tomek is left out: Tomek cleaning removed no links on this dataset, so
# its row repeated the SMOTE row digit for digit in all six metrics.
SAMPLER_ORDER = [
    "lgbm_none", "lgbm_smote", "lgbm_adasyn", "lgbm_borderline_smote",
    "lgbm_smote_enn", "lgbm_random_under", "lgbm_random_over", "lgbm_tasr",
]
# The full pipeline first, then what each row takes away from it. Only the arms
# that isolate a component are shown: dropping the encoder leaves TASR alone,
# which is already a row in the sampler block, and the width, cleaning and
# feature-set variants are configuration choices rather than components.
ABLATION_ORDER = [
    "full", "no_contrastive", "no_reconstruction", "no_tasr",
    "smote_instead_of_tasr",
]


def build_paper1_tables(output_dir: Path) -> dict[str, str]:
    with use_layout(BEEI):
        return _build_paper1_tables(output_dir)


def _build_paper1_tables(output_dir: Path) -> dict[str, str]:
    tables: dict[str, str] = {}
    runs = load_runs(1)
    if runs.empty:
        return tables

    baselines = runs[runs["stage"] == "baselines"]
    sampling = runs[runs["stage"] == "sampling"]
    ablation = runs[runs["stage"] == "ablation"]

    # The three tables of the negative-results section, in one float. They share
    # a dataset, a protocol and a metric set, and the section reads them in
    # sequence, so three captions and three headers were paying for nothing.
    blocks = [
        (baselines, BASELINE_ORDER, "Classifiers, no resampling"),
        (sampling, SAMPLER_ORDER, "Resampling strategies, identical LightGBM classifier"),
        (ablation, ABLATION_ORDER, "Components of the proposed representation pipeline"),
    ]
    negative: list[list[str] | str] = []
    for frame, order, title in blocks:
        frame = frame[frame["arm"].isin(order)]
        if frame.empty:
            continue
        if negative:
            negative.append("\\midrule")
        summary = order_by(summarise(frame, PAPER1_METRICS), "auprc", order)
        span = 1 + len(PAPER1_METRICS)
        negative.append(f"\\multicolumn{{{span}}}{{l}}{{\\textit{{{title}}}}} \\\\")
        negative.extend(summary_rows(summary, PAPER1_METRICS))

    if negative:
        tables["paper1_negative"] = latex_table(
            ["Configuration"] + metric_headers(PAPER1_METRICS),
            negative,
            caption=(
                "Everything the negative-results section compares, on the ULB "
                "credit card dataset under one chronological rolling-origin "
                "protocol. Mean $\\pm$ standard deviation over five temporal "
                "folds, best within each block in bold. Resampling is applied "
                "inside the training fold only. SMOTE-Tomek is omitted because "
                "Tomek cleaning removed no links here, so its results coincide "
                "with SMOTE in every metric, and the encoder-free ablation "
                "because it is the TASR row of the second block. Ablations of "
                "the encoder width, the cleaning step and the feature set are "
                "in the released runs and change nothing in the reading."
            ),
            label="tab:p1-negative",
        )

    # The protocol sweep and the published figures answer one question between
    # them --- how much of a reported number is the protocol rather than the
    # method --- and the section reads them in sequence, so they share a float.
    tables["paper1_protocol"] = _protocol_table(runs[runs["stage"] == "leakage"])

    _write_all(tables, output_dir)
    return tables


def _protocol_table(leakage: pd.DataFrame) -> str:
    """Protocol variants above, published figures below, in one float.

    Both blocks make the same point from opposite ends: the top measures what a
    protocol choice is worth on one classifier, and the bottom shows the spread
    across studies that choose differently. Read together they say the
    literature is ordered more by protocol than by method, which is why they now
    share a caption instead of paying for two.
    """
    rows: list[list[str] | str] = []
    span = 1 + len(PAPER1_METRICS)

    summary = summarise(leakage, PAPER1_METRICS) if not leakage.empty else pd.DataFrame()
    if not summary.empty:
        rows.append(
            f"\\multicolumn{{{span}}}{{l}}{{\\textit{{Protocol variants, identical "
            "LightGBM classifier}} \\\\"
        )
        labels = {
            "chronological__in_fold": "Chronological split, resampling in fold",
            "chronological__before_split": "Chronological split, resampling before split",
            "random__in_fold": "Random split, resampling in fold",
            "random__before_split": "Random split, resampling before split",
        }
        for arm, label in labels.items():
            record = summary[summary["arm"] == arm]
            if record.empty:
                continue
            record = record.iloc[0]
            cells = [label]
            for metric in PAPER1_METRICS:
                cells.append(format_mean_std(
                    record.get(f"{metric}_mean"), record.get(f"{metric}_std")))
            rows.append(cells)

    entries = _spanning(PAPER1_LITERATURE, PAPER1_METRICS[0], 4)
    if entries:
        if rows:
            rows.append("\\midrule")
        rows.append(
            f"\\multicolumn{{{span}}}{{l}}{{\\textit{{Published results on this "
            "benchmark}} \\\\"
        )
        for entry in sorted(entries, key=lambda e: -e.year):
            label = f"{escape(entry.study)} ({entry.year}), {escape(entry.model)}"
            if entry.leaky:
                label += " \\textsuperscript{\\dag}"
            cells = [label]
            for metric in PAPER1_METRICS:
                value = entry.metrics.get(metric)
                decimals = METRIC_LABELS.get(metric, (metric, 4, "max"))[1]
                cells.append("--" if value is None else f"{value:.{decimals}f}")
            rows.append(cells)

    if not rows:
        return ""

    return latex_table(
        ["Configuration"] + metric_headers(PAPER1_METRICS),
        rows,
        caption=(
            "How much of a reported number is the evaluation protocol rather than "
            "the model, on the ULB credit card dataset. The upper block varies only "
            "the split and the placement of resampling, holding the classifier and "
            "its hyperparameters fixed; the lower block lists four published studies "
            "spanning the reported range. A dagger marks a protocol in which "
            "resampling or augmentation precedes the split, so the figure is not "
            "directly comparable with the leakage-free results here."
        ),
        label="tab:p1-protocol",
    )


# --------------------------------------------------------------------------- #
# Paper 2
# --------------------------------------------------------------------------- #
PAPER2_METRICS = ["roc_auc", "ks", "f1", "g_mean", "brier"]
# Two of the five, for the tables that carry three datasets at once. Five
# metrics times three datasets is sixteen columns, which no page width forgives.
PAPER2_HEADLINE_METRICS = ["roc_auc", "ks"]
PAPER2_BASELINE_ORDER = [
    "logistic_regression", "decision_tree", "random_forest", "knn", "svm",
    "xgboost", "lightgbm", "catboost", "mlp", "tabnet_reference",
]
ARCHITECTURE_ORDER = [
    "tabnet_vanilla", "gma_single_head", "gma_fixed_temperature",
    "gma_no_interaction_gate", "gma_no_focal", "gma_full", "gma_pretrained",
]
STACKING_ORDER = [
    "stack_no_extractor", "stack_tabnet_vanilla", "stack_gma", "stack_gma_nested",
]


def build_paper2_tables(output_dir: Path) -> dict[str, str]:
    tables: dict[str, str] = {}
    runs = load_runs(2)

    if not runs.empty:
        for name, stage, order, header, caption in [
            ("paper2_baselines", "baselines", PAPER2_BASELINE_ORDER, "Model",
             "Reproduced baselines on all three datasets."),
            ("paper2_architecture", "architecture", ARCHITECTURE_ORDER, "Configuration",
             "Architecture ablation. Each row changes exactly one component of the "
             "proposed model, so a row is read against the vanilla TabNet above it."),
            ("paper2_stacking", "stacking", STACKING_ORDER, "Ensemble",
             "Stacking configurations, with the base learners held fixed and only "
             "the feature extractor varied."),
        ]:
            # The convention is stated once, in the first table, and referred to
            # afterwards. Repeating it verbatim under all three cost four lines
            # of caption apiece, and the three of them share a page.
            convention = (
                " Mean $\\pm$ standard deviation over five stratified folds; best "
                "per column in bold. ROC-AUC and the Kolmogorov--Smirnov statistic "
                "are shown because they are the two the text argues over; F1, "
                "G-mean and the Brier score are in the released results."
                if stage == "baselines" else
                " Read as Table~\\ref{tab:p2-baselines}."
            )
            built = cross_dataset_table(
                runs, stage, PAPER2_HEADLINE_METRICS, order, header,
                caption=caption + convention,
                label=f"tab:p2-{stage}",
            )
            if built:
                tables[name] = built

        friedman = _friedman_table(runs)
        if friedman:
            tables["paper2_friedman"] = friedman

    hpo = _hpo_table()
    if hpo:
        tables["paper2_hpo"] = hpo

    tables["paper2_literature"] = literature_table(
        PAPER2_LITERATURE,
        ["roc_auc", "auprc", "ks", "f1", "brier"],
        caption=(
            "Published results on the three credit scoring datasets, three studies "
            "per dataset spanning the reported range. Figures marked with a dagger "
            "come from protocols that augment or resample before evaluation and sit "
            "far above the honest ceiling for these datasets."
        ),
        label="tab:p2-literature",
        limit=3,
    )
    _write_all(tables, output_dir)
    return tables


def _friedman_table(runs: pd.DataFrame) -> str:
    """Rank the headline models across datasets and test the differences."""
    candidates = ["xgboost", "lightgbm", "catboost", "tabnet_reference",
                  "tabnet_vanilla", "gma_full", "stack_tabnet_vanilla", "stack_gma"]
    scores: dict[str, list[float]] = {}
    for arm in candidates:
        subset = runs[runs["arm"] == arm]
        if subset.empty or "roc_auc" not in subset.columns:
            continue
        per_dataset = subset.groupby("dataset")["roc_auc"].mean().sort_index()
        if len(per_dataset) >= 2:
            scores[arm] = per_dataset.tolist()

    lengths = {len(v) for v in scores.values()}
    if len(scores) < 3 or len(lengths) != 1:
        return ""

    outcome = friedman_nemenyi(scores)
    rows = [
        [pretty_arm(arm), f"{rank:.2f}"]
        for arm, rank in sorted(outcome["average_ranks"].items(), key=lambda kv: kv[1])
    ]
    return latex_table(
        ["Model", "Average rank"],
        rows,
        caption=(
            f"Friedman test across the credit scoring datasets "
            f"($\\chi^2 = {outcome['friedman_statistic']:.3f}$, "
            f"$p = {outcome['friedman_p_value']:.4f}$), over "
            f"{outcome['n_datasets']} datasets. Two models differ significantly "
            f"under the Nemenyi post-hoc test at $\\alpha = 0.05$ when their "
            f"average ranks are more than "
            f"{outcome['critical_difference']:.2f} apart. "
            f"Lower average rank is better."
        ),
        label="tab:p2-friedman",
    )


def _hpo_table() -> str:
    """Compare the optimisers on what each achieved within the same budget."""
    directory = PROJECT_ROOT / "results" / "paper2" / "tuning"
    if not directory.exists():
        return ""

    records = []
    for path in sorted(directory.glob("*__hpo__*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append({
            "dataset": payload.get("dataset"),
            "optimiser": payload.get("optimiser"),
            "best_value": payload.get("best_value"),
            "n_evaluations": payload.get("n_evaluations"),
            "elapsed_seconds": payload.get("elapsed_seconds"),
        })
    if not records:
        return ""

    frame = pd.DataFrame(records)
    datasets = sorted(frame["dataset"].dropna().unique())
    headers = ["Optimiser"] + [
        f"{DATASET_LABELS.get(d, d)}" for d in datasets
    ] + ["Mean evaluations"]

    rows = []
    for optimiser in ["random", "tpe", "pso", "ga"]:
        subset = frame[frame["optimiser"] == optimiser]
        if subset.empty:
            continue
        cells = [pretty_arm(optimiser)]
        for dataset in datasets:
            record = subset[subset["dataset"] == dataset]
            if record.empty:
                cells.append("--")
                continue
            value = record.iloc[0]["best_value"]
            column = frame[frame["dataset"] == dataset]["best_value"]
            bold = bool(value >= column.max() - 1e-9)
            cells.append(f"\\textbf{{{value:.4f}}}" if bold else f"{value:.4f}")
        cells.append(f"{subset['n_evaluations'].mean():.1f}")
        rows.append(cells)

    budget = ""
    return latex_table(
        headers, rows,
        caption=(
            "Hyperparameter optimisers under a matched wall-clock budget on an "
            "identical search space. The reported value is the best validation ROC-AUC "
            "reached within the budget." + budget
        ),
        label="tab:p2-hpo",
        notes=(
            "Random search is included as the control that distinguishes a genuinely "
            "effective optimiser from one that merely samples more configurations."
        ),
    )


def _write_all(tables: dict[str, str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in tables.items():
        if content:
            (output_dir / f"{name}.tex").write_text(content, encoding="utf-8")
