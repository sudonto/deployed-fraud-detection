"""Figures for both manuscripts, drawn from the stored runs and predictions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis.aggregate import (  # noqa: E402
    DATASET_LABELS,
    load_runs,
    pretty_arm,
    summarise,
)
from src.common.io import PROJECT_ROOT, load_scores  # noqa: E402

PALETTE = ["#1f4e79", "#c0392b", "#2e8b57", "#8e44ad", "#e67e22",
           "#16a085", "#7f8c8d", "#2c3e50", "#d35400", "#27ae60"]


def apply_style(serif: bool = False) -> None:
    """Set the drawing style; ``serif`` switches to the IJIES lettering.

    IJIES asks for 10 pt Times New Roman inside figures, matching the body, and
    its submission checklist asks the authors to confirm it. Paper 1 goes to
    BEEI, which asks for nothing of the kind, so the sans default stays put
    rather than both papers being changed to suit one of them.

    A figure only lands on 10 pt if it is drawn at the width it is placed at.
    ``width=\\columnwidth`` scales the whole graphic, lettering included, so a
    figure drawn four times too wide arrives with 2 pt labels no matter what
    ``font.size`` says here. The two multi-panel figures used to do exactly
    that; they are now drawn at text width and placed as full-width floats.
    """
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10 if serif else 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10 if serif else 9,
        "xtick.labelsize": 9 if serif else 8.5,
        "ytick.labelsize": 9 if serif else 8.5,
        "legend.fontsize": 9 if serif else 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 1.6,
    })
    if serif:
        plt.rcParams.update({
            "font.family": "serif",
            # Nimbus and Liberation are metric-compatible stand-ins, in case
            # this is ever rebuilt somewhere without the Microsoft font.
            "font.serif": ["Times New Roman", "Nimbus Roman",
                           "Liberation Serif", "DejaVu Serif"],
            # Matplotlib's own mathtext faces are sans by default, which would
            # leave every symbol in the figures disagreeing with the prose.
            "mathtext.fontset": "stix",
        })
    else:
        plt.rcParams.update({
            "font.family": "sans-serif",
            "mathtext.fontset": "dejavusans",
        })


# The IJIES print area, in inches: a column is 82.2 mm and the full text width
# 172 mm. A figure drawn at one of these two widths is placed at scale 1, so its
# declared point sizes are the sizes that reach the page.
COLUMN_IN = 82.2 / 25.4
TEXT_IN = 172.0 / 25.4


def save(fig, output_dir: Path, name: str, pad: float | None = None) -> Path:
    """Write the figure as PDF and PNG.

    ``pad`` overrides the white border that the tight bounding box leaves. A
    plot wants that border, since its axis labels would otherwise sit flush
    against the caption. A schematic drawn to fill its own canvas does not: the
    border is scaled up with everything else by ``width=\\columnwidth``, so a
    tenth of an inch of nothing costs six per cent of the type size.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extra = {} if pad is None else {"pad_inches": pad}
    path = output_dir / f"{name}.pdf"
    fig.savefig(path, **extra)
    fig.savefig(output_dir / f"{name}.png", **extra)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Paper 1
# --------------------------------------------------------------------------- #
def figure_review_policy(output_dir: Path) -> None:
    """One retraining round: where the reviews go and what gets written down.

    The only figure in either paper that draws a procedure rather than a result.
    It earns the space because the contribution is a bookkeeping rule, and the
    whole difference between the proposed policy and deployed practice is a
    single arrow into the training pool. That is far easier to see than to read,
    which is why Section IV previously needed four paragraphs to say it.

    Band heights are not to scale: at a review budget of half a percent the
    bottom band holds more than 99% of the round. Drawing it to scale would
    reduce the three bands that matter to invisible slivers.
    """
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig, axis = plt.subplots(figsize=(3.4, 3.4))
    axis.set_xlim(-1.5, 10.6)
    axis.set_ylim(0.0, 8.4)
    axis.axis("off")
    axis.grid(False)

    ink = "#2c3e50"
    axis.text(1.7, 7.95, "round $t$: score $\\mathcal{D}_t$ with $f_{t-1}$,\nsort by score $s$",
              ha="center", va="center", fontsize=7.2, color=ink)

    bar_left, bar_right = 0.2, 3.2

    # Top to bottom in score order. The two middle bands are the contribution:
    # one spends reviews below the alert cutoff, the other refuses to label what
    # the reviews could not reach.
    bands = [
        (5.60, 6.60, PALETTE[0], "top-$k$\nreviewed", "white"),
        (4.75, 5.60, PALETTE[2], "exploration\nband", "white"),
        (3.75, 4.75, PALETTE[1], "censored", "white"),
        (0.95, 3.75, "#d5d8dc", "unreviewed\nremainder", ink),
    ]
    for y0, y1, colour, label, text_colour in bands:
        axis.add_patch(Rectangle(
            (bar_left, y0), bar_right - bar_left, y1 - y0,
            facecolor=colour, edgecolor="white", linewidth=0.8,
        ))
        axis.text((bar_left + bar_right) / 2, (y0 + y1) / 2, label,
                  ha="center", va="center", fontsize=6.4, color=text_colour,
                  linespacing=1.15)

    axis.text(bar_left - 0.15, 6.60, "highest $s$", ha="right", va="center",
              fontsize=6.0, color=ink)
    axis.text(bar_left - 0.15, 0.95, "lowest $s$", ha="right", va="center",
              fontsize=6.0, color=ink)

    pool = (5.90, 3.20, 3.90, 1.45)
    fit = (5.90, 1.45, 3.90, 1.15)
    for x, y, width, height, label in [
        (*pool, "training pool $\\mathcal{P}_t$"),
        (*fit, "refit $f_t$"),
    ]:
        axis.add_patch(Rectangle(
            (x, y), width, height,
            facecolor="white", edgecolor=ink, linewidth=1.0,
        ))
        axis.text(x + width / 2, y + height / 2, label,
                  ha="center", va="center", fontsize=7.0, color=ink)

    def arrow(y_from: float, y_to: float, text: str, at: float = 0.5,
              dashed: bool = False) -> None:
        """One band's contribution to the pool, labelled with what gets written.

        Straight rather than curved, and the label carries an opaque box that
        interrupts its own arrow: four arrows converge into a box barely wider
        than they are tall, so a label set beside its arrow lands on the next
        one. ``at`` slides a label along its arrow to keep it off the others.
        """
        colour = "#95a5a6" if dashed else ink
        axis.add_patch(FancyArrowPatch(
            (bar_right + 0.05, y_from), (pool[0] - 0.05, y_to),
            arrowstyle="-|>", mutation_scale=8, linewidth=0.9, color=colour,
            linestyle=(0, (2.5, 1.5)) if dashed else "solid",
        ))
        axis.text(bar_right + at * (pool[0] - bar_right),
                  y_from + at * (y_to - y_from), text,
                  ha="center", va="center", fontsize=6.0, color=colour,
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})

    arrow(6.10, 4.40, "$y$ known", at=0.30)
    arrow(5.17, 4.05, "$y$ known", at=0.52)
    arrow(4.25, 3.70, "$\\tilde{y}=0$", at=0.72, dashed=True)
    arrow(2.35, 3.38, "$\\tilde{y}=0$", at=0.45)

    axis.add_patch(FancyArrowPatch(
        (pool[0] + pool[2] / 2, pool[1]), (fit[0] + fit[2] / 2, fit[1] + fit[3]),
        arrowstyle="-|>", mutation_scale=8, linewidth=0.9, color=ink,
    ))

    # The loop back to the next round, routed up the empty right-hand side and
    # across above the bar. Taking it round the left instead would run it
    # through the caption strip under the bar.
    loop_x, loop_y = 10.30, 7.25
    axis.plot([fit[0] + fit[2], loop_x], [fit[1] + fit[3] / 2] * 2, color=ink, linewidth=0.9)
    axis.plot([loop_x, loop_x], [fit[1] + fit[3] / 2, loop_y], color=ink, linewidth=0.9)
    axis.plot([loop_x, 2.60], [loop_y, loop_y], color=ink, linewidth=0.9)
    axis.add_patch(FancyArrowPatch(
        (2.60, loop_y), (2.60, 6.75),
        arrowstyle="-|>", mutation_scale=8, linewidth=0.9, color=ink,
    ))
    axis.text(6.40, loop_y + 0.10, "round $t+1$", ha="center", va="bottom",
              fontsize=6.0, color=ink)

    axis.text(4.20, 0.40,
              "dashed: recorded by deployed practice,\nomitted by the proposed rule",
              ha="center", va="center", fontsize=6.0, color="#95a5a6", linespacing=1.15)

    save(fig, output_dir, "p1_review_policy")


def figure_precision_recall(output_dir: Path, arms: list[tuple[str, str]], fold: int = 4) -> None:
    """Precision-recall curves on the most recent temporal fold.

    Precision-recall rather than ROC: with 0.17% positives, the ROC curve is
    almost entirely determined by the negatives and hides differences that
    matter operationally.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, axis = plt.subplots(figsize=(5.0, 3.6))
    drawn = 0
    for index, (stage, arm) in enumerate(arms):
        loaded = load_scores(1, stage, arm, fold, 0)
        if loaded is None:
            continue
        y_true, scores = loaded
        precision, recall, _ = precision_recall_curve(y_true, scores)
        auprc = average_precision_score(y_true, scores)
        axis.step(
            recall, precision, where="post",
            color=PALETTE[index % len(PALETTE)],
            label=f"{pretty_arm(arm)} (AUPRC {auprc:.3f})",
        )
        drawn += 1

    if drawn == 0:
        plt.close(fig)
        return

    prevalence = float(np.mean(y_true))
    axis.axhline(prevalence, color="grey", linestyle=":", linewidth=1.0,
                 label=f"Chance ({prevalence:.4f})")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(f"Precision-recall on the final temporal fold (fold {fold})")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.legend(loc="upper right", frameon=False)
    save(fig, output_dir, "p1_precision_recall")


def figure_fold_variance(output_dir: Path, runs: pd.DataFrame, arms: list[str]) -> None:
    """AUPRC per temporal fold: how much the answer depends on when you test."""
    subset = runs[runs["arm"].isin(arms) & runs["auprc"].notna()]
    if subset.empty:
        return

    fig, axis = plt.subplots(figsize=(5.4, 3.4))
    for index, arm in enumerate(arms):
        rows = subset[subset["arm"] == arm].sort_values("fold")
        if rows.empty:
            continue
        axis.plot(
            rows["fold"], rows["auprc"], marker="o", markersize=4,
            color=PALETTE[index % len(PALETTE)], label=pretty_arm(arm),
        )
    axis.set_xlabel("Temporal fold (later fold = later transactions)")
    axis.set_ylabel("AUPRC")
    axis.set_title("Performance drifts with the test period")
    axis.set_xticks(sorted(subset["fold"].unique()))
    axis.legend(frameon=False, ncol=2)
    save(fig, output_dir, "p1_fold_variance")


def figure_leakage(output_dir: Path, runs: pd.DataFrame) -> None:
    """The size of the protocol effect, next to the size of the model effect."""
    leakage = runs[runs["stage"] == "leakage"]
    if leakage.empty:
        return

    order = ["chronological__in_fold", "chronological__before_split",
             "random__in_fold", "random__before_split"]
    labels = ["Chronological\nresample in fold", "Chronological\nresample before split",
              "Random split\nresample in fold", "Random split\nresample before split"]
    summary = summarise(leakage, ["auprc", "mcc", "f1"])

    values, errors, kept_labels = [], [], []
    for arm, label in zip(order, labels):
        record = summary[summary["arm"] == arm]
        if record.empty:
            continue
        values.append(float(record.iloc[0]["auprc_mean"]))
        deviation = record.iloc[0].get("auprc_std")
        errors.append(0.0 if pd.isna(deviation) else float(deviation))
        kept_labels.append(label)

    if not values:
        return

    fig, axis = plt.subplots(figsize=(5.6, 3.4))
    colours = ["#1f4e79", "#c0392b", "#c0392b", "#c0392b"][: len(values)]
    bars = axis.bar(range(len(values)), values, yerr=errors, capsize=3,
                    color=colours, alpha=0.9)
    for rectangle, value in zip(bars, values):
        axis.text(rectangle.get_x() + rectangle.get_width() / 2, value + 0.015,
                  f"{value:.3f}", ha="center", fontsize=8)

    baseline = values[0]
    axis.axhline(baseline, color="#1f4e79", linestyle="--", linewidth=1.0)
    axis.set_xticks(range(len(kept_labels)))
    axis.set_xticklabels(kept_labels, fontsize=7.5)
    axis.set_ylabel("AUPRC")
    axis.set_ylim(0, min(1.05, max(values) * 1.2))
    axis.set_title("Identical model, four evaluation protocols")
    save(fig, output_dir, "p1_leakage")


def figure_sampling(output_dir: Path, runs: pd.DataFrame) -> None:
    sampling = runs[runs["stage"] == "sampling"]
    if sampling.empty:
        return
    summary = summarise(sampling, ["auprc", "mcc"]).sort_values("auprc_mean")

    fig, axis = plt.subplots(figsize=(5.4, 3.6))
    labels = [pretty_arm(a) for a in summary["arm"]]
    colours = ["#c0392b" if "tasr" in a else "#1f4e79" for a in summary["arm"]]
    axis.barh(range(len(summary)), summary["auprc_mean"],
              xerr=summary["auprc_std"].fillna(0), capsize=2, color=colours, alpha=0.9)
    axis.set_yticks(range(len(summary)))
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("AUPRC")
    axis.set_title("Resampling strategies, identical classifier")
    save(fig, output_dir, "p1_sampling")


def figure_ablation(output_dir: Path, runs: pd.DataFrame) -> None:
    """Change in AUPRC when each component is removed from the full model."""
    ablation = runs[runs["stage"] == "ablation"]
    if ablation.empty:
        return
    summary = summarise(ablation, ["auprc"])
    full = summary[summary["arm"] == "full"]
    if full.empty:
        return
    reference = float(full.iloc[0]["auprc_mean"])

    others = summary[summary["arm"] != "full"].copy()
    others["delta"] = others["auprc_mean"] - reference
    others = others.sort_values("delta")

    fig, axis = plt.subplots(figsize=(5.4, 3.6))
    colours = ["#c0392b" if d < 0 else "#2e8b57" for d in others["delta"]]
    axis.barh(range(len(others)), others["delta"], color=colours, alpha=0.9)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(range(len(others)))
    axis.set_yticklabels([pretty_arm(a) for a in others["arm"]], fontsize=8)
    axis.set_xlabel(f"Change in AUPRC relative to the full model ({reference:.4f})")
    axis.set_title("Removing one component at a time")
    save(fig, output_dir, "p1_ablation")


# --------------------------------------------------------------------------- #
# Paper 2
# --------------------------------------------------------------------------- #
def figure_gma_tabnet(output_dir: Path) -> None:
    """One decision step, with the three changes to TabNet marked.

    The contribution of this paper is architectural, and until now the only
    figures were results. A reader had to assemble the design from five
    equations spread over two pages to see what actually differs from standard
    TabNet, which is a lot of work for three localised changes.

    The mask strips beside each head carry the argument that the prose makes at
    length: a low temperature makes a head almost one-hot, a high one spreads
    it, and the gate-weighted mixture is denser than any single head while
    still summing to one. Drawing the strips is what turns the simplex
    proposition from an assertion into something visible.

    Nothing here is captioned twice. Earlier versions annotated the strips, the
    mixture and the dashed box in grey, which repeated three sentences of the
    caption inside a column-width drawing and left the right-hand margin too
    crowded to read. The words belong in the caption; the figure keeps only
    labels that name a part or state a formula.
    """
    from matplotlib.colors import to_rgb
    from matplotlib.patches import FancyArrowPatch, Rectangle

    # The axes fills the canvas and the limits hug the drawing. A tight bounding
    # box takes the union of the axes rectangle and whatever sticks out of it,
    # so an axes inset by the default subplot margins ships that inset inside the
    # PDF, \includegraphics scales it up with the drawing, and the schematic ends
    # up filling seven eighths of the column it was told to fill.
    # Drawn one column wide plus the two hundredths of an inch of padding that
    # save() leaves, so the graphic is placed at scale 1 and the sizes below are
    # the sizes that reach the page: 10 pt for anything that names a part, one
    # step down for the two annotations that qualify one.
    fig = plt.figure(figsize=(COLUMN_IN - 0.04, 3.52))
    axis = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_xlim(0.12, 9.58)
    axis.set_ylim(0.40, 10.02)
    axis.set_axis_off()
    axis.grid(False)

    ink = "#2c3e50"
    faint = "#95a5a6"
    accent = PALETTE[0]
    margin = 1.10  # Left edge of every box, leaving the prior rail a channel.
    spine = 4.35   # The lower column of boxes is centred on this x.
    LABEL, ASIDE = 10.0, 8.5

    def box(x0, y0, x1, y1, label, *, text=ink, dashed=False, size=LABEL):
        axis.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor="white", edgecolor=faint if dashed else ink, linewidth=0.9,
            linestyle=(0, (2.5, 1.5)) if dashed else "solid",
        ))
        if label:
            axis.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center",
                      va="center", fontsize=size, color=text, linespacing=1.2)

    def strip(x0, y_centre, weights):
        """A mask over six features, shaded by how much each one is selected.

        The outline matters: a head with a low temperature leaves most cells at
        weight zero, and without a border those cells are white on white and
        the strip stops reading as a strip.
        """
        cell, height = 0.28, 0.28
        base = np.array(to_rgb(accent))
        for index, weight in enumerate(weights):
            shade = tuple(1.0 - (1.0 - base) * weight)
            axis.add_patch(Rectangle(
                (x0 + index * cell, y_centre - height / 2), cell, height,
                facecolor=shade, edgecolor="white", linewidth=0.5,
            ))
        axis.add_patch(Rectangle(
            (x0, y_centre - height / 2), cell * len(weights), height,
            facecolor="none", edgecolor=faint, linewidth=0.5,
        ))

    def arrow(start, end):
        axis.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=7, linewidth=0.85,
            color=ink,
        ))

    axis.text(4.90, 9.78, "$a_{t-1}$ from decision step $t-1$",
              ha="center", va="center", fontsize=ASIDE, color=ink)

    # Heads. Temperature rises down the stack, so the strips go from almost
    # one-hot at the top to nearly flat at the bottom.
    heads = [
        (8.80, "head $1$   $\\tau_1$", [1.00, 0.06, 0.00, 0.00, 0.00, 0.00]),
        (8.05, "head $2$   $\\tau_2$", [0.48, 0.40, 0.16, 0.04, 0.00, 0.00]),
        (7.30, "head $H$   $\\tau_H$", [0.30, 0.25, 0.20, 0.16, 0.12, 0.09]),
    ]
    # Head box, mask strip and gate share one row 8.48 units wide. At 10 pt a
    # head label needs 2.6 of those and the gate's formula 2.4, which is what
    # sets the two box widths below; the row was a third narrower when the
    # lettering was 7 pt.
    head_right, strip_left, gate_left = 4.10, 4.30, 6.30
    head_middle = (margin + head_right) / 2
    for y_centre, label, weights in heads:
        box(margin, y_centre - 0.32, head_right, y_centre + 0.32, label)
        strip(strip_left, y_centre, weights)

    # The gate carries its formula inside the box, so no annotation has to be
    # parked in the margin beside it.
    gate_middle = (gate_left + 9.55) / 2
    box(gate_left, 8.10, 9.55, 9.20, "")
    axis.text(gate_middle, 8.90, "gate $\\pi_t$", ha="center", va="center",
              fontsize=LABEL, color=ink)
    axis.text(gate_middle, 8.42, "softmax$(W^{g} a_{t-1})$", ha="center",
              va="center", fontsize=ASIDE, color=faint)

    arrow((4.45, 9.45), (head_middle, 9.14))
    arrow((5.45, 9.45), (gate_middle - 0.20, 9.23))

    box(margin, 5.75, 7.60, 6.55, "$M_t=\\sum_h \\pi_{t,h} M_t^{(h)}$")
    strip(7.80, 6.15, [0.62, 0.24, 0.12, 0.07, 0.04, 0.03])

    arrow((head_middle, 6.98), (head_middle, 6.57))
    arrow((7.10, 8.10), (7.10, 6.57))

    box(2.90, 4.50, 5.80, 5.25, "$M_t \\odot \\mathbf{x}$")
    arrow((spine, 5.75), (spine, 5.27))

    box(1.75, 2.95, 6.95, 3.95,
        "interaction gate\nscale $s$ initialised at $0$",
        text=faint, dashed=True, size=ASIDE)
    arrow((spine, 4.50), (spine, 3.97))

    box(2.00, 1.45, 6.70, 2.25, "feature transformer")
    arrow((spine, 2.95), (spine, 2.27))
    arrow((spine, 1.45), (spine, 0.95))
    axis.text(spine, 0.78, "to decision output and $a_t$", ha="center",
              va="top", fontsize=ASIDE, color=ink)

    # Prior update, routed up the left margin so it does not cross the column
    # of boxes down the middle. The label sits below the return arrow rather
    # than beside it, which is the only part of the channel wide enough for it.
    rail = 0.20
    axis.plot([margin, rail], [6.15, 6.15], color=ink, linewidth=0.85)
    axis.plot([rail, rail], [6.15, 8.05], color=ink, linewidth=0.85)
    arrow((rail, 8.05), (margin - 0.02, 8.05))
    axis.text(rail + 0.16, 7.05, "prior update", rotation=90, ha="left",
              va="center", fontsize=ASIDE, color=ink)

    save(fig, output_dir, "p2_gma_tabnet", pad=0.02)


def figure_hpo_convergence(output_dir: Path) -> None:
    """Best validation score so far against wall-clock time, per optimiser."""
    directory = PROJECT_ROOT / "results" / "paper2" / "tuning"
    if not directory.exists():
        return

    by_dataset: dict[str, dict[str, list]] = {}
    for path in sorted(directory.glob("*__hpo__*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = payload.get("dataset", "unknown")
        by_dataset.setdefault(dataset, {})[payload["optimiser"]] = payload.get("history", [])

    if not by_dataset:
        return

    # Drawn at the full text width, because that is where it is placed. At the
    # old 4.4 inches per panel the graphic was thirteen inches wide, was pressed
    # into a 3.2 inch column, and arrived with two-point axis labels.
    datasets = sorted(by_dataset)
    fig, axes = plt.subplots(1, len(datasets), figsize=(TEXT_IN, 2.2),
                             squeeze=False, sharey=True)
    handles: list = []
    for column, dataset in enumerate(datasets):
        axis = axes[0][column]
        for index, (optimiser, history) in enumerate(sorted(by_dataset[dataset].items())):
            if not history:
                continue
            elapsed = [point["elapsed"] for point in history]
            best = [point["best"] for point in history]
            line, = axis.step(elapsed, best, where="post",
                              color=PALETTE[index % len(PALETTE)],
                              label=pretty_arm(optimiser))
            if column == 0:
                handles.append(line)
        axis.set_xlabel("Wall-clock seconds")
        if column == 0:
            axis.set_ylabel("Best validation ROC-AUC")
        axis.set_title(DATASET_LABELS.get(dataset, dataset))
    # One legend for the row rather than one per panel: four optimiser names do
    # not fit inside a panel a third of the text width without shrinking below
    # the size the journal asks for.
    if handles:
        fig.legend(handles=handles, frameon=False, ncol=len(handles),
                   loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, output_dir, "p2_hpo_convergence")


def figure_temperatures(output_dir: Path) -> None:
    """What sparsity each attention head settled on, per decision step."""
    runs_dir = PROJECT_ROOT / "results" / "paper2" / "raw_runs"
    if not runs_dir.exists():
        return

    collected: dict[str, np.ndarray] = {}
    for path in sorted(runs_dir.glob("*__architecture__gma_full__fold0__*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        temperatures = payload.get("learned_temperatures") or {}
        if not temperatures:
            continue
        matrix = np.array([temperatures[key] for key in sorted(temperatures)])
        collected[payload["dataset"]] = matrix

    if not collected:
        return

    # Text width again, for the same reason as the convergence figure. Laid out
    # under constraint so the panels grow into the canvas: left to the default
    # margins the row of heat maps ended half an inch short of the width it was
    # given, the tight bounding box trimmed to the drawing, and \includegraphics
    # then scaled the lettering back up past the size it was set at.
    #
    # Panels are widened in proportion to their head count. Tuning picked a
    # different number of heads per dataset, so equal panels gave the eight-head
    # dataset cells a quarter narrower than the six-head one, and a cell entry
    # small enough to fit the narrowest was two points under what IJIES asks
    # for. Equal cells let one size serve all three.
    ordered = sorted(collected.items())
    heads = [matrix.shape[1] for _, matrix in ordered]
    fig, axes = plt.subplots(
        1, len(ordered), figsize=(TEXT_IN, 2.05), squeeze=False,
        layout="constrained", width_ratios=heads,
    )
    # A cell entry is three digits and a point, 1.75 em of Times, and wants half
    # an em of air around it.
    cell_pt = (TEXT_IN - 0.9) / max(sum(heads), 1) * 72.0
    annotation = min(10.0, cell_pt / 2.25)
    image = None
    for column, (dataset, matrix) in enumerate(ordered):
        axis = axes[0][column]
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=0.5, vmax=1.5, aspect="auto")
        axis.set_xlabel("Attention head")
        if column == 0:
            axis.set_ylabel("Decision step")
        axis.set_xticks(range(matrix.shape[1]))
        axis.set_yticks(range(matrix.shape[0]))
        axis.set_title(DATASET_LABELS.get(dataset, dataset))
        axis.grid(False)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                          fontsize=annotation)
    # One colour bar for the row: three of them ate a third of the width and
    # repeated the same scale three times.
    if image is not None:
        fig.colorbar(image, ax=axes[0].tolist(), fraction=0.02, pad=0.015,
                     label="Learned temperature")
    save(fig, output_dir, "p2_temperatures")


def figure_critical_difference(output_dir: Path, scores: dict[str, list[float]]) -> None:
    """Nemenyi critical-difference diagram over the credit scoring datasets."""
    from src.common.stats import friedman_nemenyi

    if len(scores) < 3:
        return
    outcome = friedman_nemenyi(scores)
    ranks = outcome["average_ranks"]
    critical_distance = outcome.get("critical_difference")

    ordered = sorted(ranks.items(), key=lambda kv: kv[1])
    names = [pretty_arm(name) for name, _ in ordered]
    values = [value for _, value in ordered]

    fig, axis = plt.subplots(figsize=(6.0, 2.4 + 0.22 * len(names)))
    axis.scatter(values, range(len(values)), color="#1f4e79", zorder=3)
    for position, (name, value) in enumerate(zip(names, values)):
        axis.text(value + 0.06, position, f"{name} ({value:.2f})", va="center", fontsize=8)
    if critical_distance:
        axis.errorbar(
            values[0], -0.8, xerr=critical_distance / 2, fmt="none",
            ecolor="#c0392b", capsize=4,
        )
        axis.text(values[0], -1.35, f"critical difference = {critical_distance:.2f}",
                  ha="center", fontsize=8, color="#c0392b")
    axis.set_yticks([])
    axis.set_xlabel("Average rank (lower is better)")
    axis.set_xlim(min(values) - 0.4, max(values) + 2.2)
    axis.set_ylim(-1.8, len(values))
    axis.set_title(
        f"Friedman test: chi-square {outcome['friedman_statistic']:.2f}, "
        f"p = {outcome['friedman_p_value']:.4f}"
    )
    save(fig, output_dir, "p2_critical_difference")


def figure_dataset_comparison(output_dir: Path, runs: pd.DataFrame) -> None:
    """Headline models side by side on every credit dataset."""
    interesting = ["logistic_regression", "random_forest", "xgboost", "lightgbm",
                   "catboost", "tabnet_vanilla", "gma_full", "stack_gma"]
    subset = runs[runs["arm"].isin(interesting) & runs["roc_auc"].notna()]
    if subset.empty:
        return

    summary = subset.groupby(["dataset", "arm"])["roc_auc"].agg(["mean", "std"]).reset_index()
    datasets = sorted(summary["dataset"].unique())
    arms = [a for a in interesting if a in set(summary["arm"])]

    fig, axis = plt.subplots(figsize=(1.9 * len(datasets) + 3.0, 3.6))
    width = 0.8 / max(len(arms), 1)
    for index, arm in enumerate(arms):
        rows = summary[summary["arm"] == arm].set_index("dataset").reindex(datasets)
        positions = np.arange(len(datasets)) + index * width - 0.4 + width / 2
        axis.bar(positions, rows["mean"].fillna(0), width=width,
                 yerr=rows["std"].fillna(0), capsize=2,
                 color=PALETTE[index % len(PALETTE)], label=pretty_arm(arm))
    axis.set_xticks(np.arange(len(datasets)))
    axis.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    axis.set_ylabel("ROC-AUC")
    axis.set_ylim(0.5, 1.0)
    axis.legend(frameon=False, ncol=3, fontsize=7.5)
    axis.set_title("Credit scoring models across three datasets")
    save(fig, output_dir, "p2_dataset_comparison")


# --------------------------------------------------------------------------- #
def build_paper1_figures(output_dir: Path) -> None:
    apply_style()
    # Drawn before the guard below: it is a schematic of the protocol and owes
    # nothing to the runs, so a partial sweep should not withhold it.
    figure_review_policy(output_dir)

    runs = load_runs(1)
    if runs.empty:
        return

    figure_leakage(output_dir, runs)
    figure_sampling(output_dir, runs)
    figure_ablation(output_dir, runs)
    figure_fold_variance(
        output_dir, runs,
        ["random_forest", "lightgbm", "xgboost", "mlp", "proposed_lightgbm"],
    )
    figure_precision_recall(
        output_dir,
        [
            ("proposed", "proposed_lightgbm"),
            ("baselines", "lightgbm"),
            ("baselines", "random_forest"),
            ("baselines", "mlp"),
            ("baselines", "logistic_regression"),
        ],
        fold=int(runs["fold"].max()) if runs["fold"].notna().any() else 4,
    )


def build_paper2_figures(output_dir: Path) -> None:
    apply_style(serif=True)
    # Drawn before the guard below: it is a schematic of the architecture and
    # owes nothing to the runs, so a partial sweep should not withhold it.
    figure_gma_tabnet(output_dir)

    figure_hpo_convergence(output_dir)
    figure_temperatures(output_dir)

    runs = load_runs(2)
    if runs.empty:
        return
    figure_dataset_comparison(output_dir, runs)

    candidates = ["xgboost", "lightgbm", "catboost", "tabnet_vanilla",
                  "gma_full", "stack_tabnet_vanilla", "stack_gma"]
    scores: dict[str, list[float]] = {}
    for arm in candidates:
        subset = runs[runs["arm"] == arm]
        if subset.empty or "roc_auc" not in subset.columns:
            continue
        per_dataset = subset.groupby("dataset")["roc_auc"].mean().sort_index()
        if len(per_dataset) >= 2:
            scores[arm] = per_dataset.tolist()
    lengths = {len(v) for v in scores.values()}
    if len(scores) >= 3 and len(lengths) == 1:
        figure_critical_difference(output_dir, scores)
