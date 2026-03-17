from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from .collision_figure_style import (
        BODY_A_COLOR,
        BODY_B_COLOR,
        GUIDE_COLOR,
        MANIFOLD_COLOR,
        NEUTRAL_EDGE,
        NEUTRAL_FILL,
        NORMAL_COLOR,
        TEXT_COLOR,
        add_label,
        add_titles,
        make_figure,
        save_figure,
        style_axis,
    )
except ImportError:
    from collision_figure_style import (
        BODY_A_COLOR,
        BODY_B_COLOR,
        GUIDE_COLOR,
        MANIFOLD_COLOR,
        NEUTRAL_EDGE,
        NEUTRAL_FILL,
        NORMAL_COLOR,
        TEXT_COLOR,
        add_label,
        add_titles,
        make_figure,
        save_figure,
        style_axis,
    )


def draw_interval(ax, xmin, xmax, y, label, facecolor, edgecolor, *, alpha=0.3, height=0.38):
    ax.add_patch(
        Rectangle(
            (xmin, y - 0.5 * height),
            xmax - xmin,
            height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=2.0,
            alpha=alpha,
        )
    )
    ax.text(
        0.5 * (xmin + xmax),
        y,
        label,
        fontsize=15,
        fontweight="bold",
        ha="center",
        va="center",
        color=TEXT_COLOR,
    )
    ax.plot([xmin, xmin], [-0.08, y - 0.5 * height], linestyle="--", color=GUIDE_COLOR, linewidth=1.1)
    ax.plot([xmax, xmax], [-0.08, y - 0.5 * height], linestyle="--", color=GUIDE_COLOR, linewidth=1.1)


def main():
    parser = argparse.ArgumentParser(description="Create a thesis-style AABB sweep-and-prune figure.")
    parser.add_argument("--save", help="Optional save path. Defaults to output/figures/aabb_sweep_phase.png")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively after saving.")
    args = parser.parse_args()

    fig = make_figure(figsize=(13, 7.2))
    add_titles(
        fig,
        "Broad-Phase AABB Sweep and Prune",
        "Sorted x-endpoints define the active list; only surviving pairs advance to the y/z overlap test.",
    )

    ax = fig.add_subplot(1, 1, 1)
    style_axis(ax)
    ax.set_xlim(0.2, 9.8)
    ax.set_ylim(-0.8, 3.9)

    ax.plot([0.6, 9.2], [0.0, 0.0], color=GUIDE_COLOR, linewidth=1.4)
    ax.annotate(
        "",
        xy=(9.35, 0.0),
        xytext=(9.0, 0.0),
        arrowprops=dict(arrowstyle="->", linewidth=1.8, color=GUIDE_COLOR),
    )
    add_label(ax, 9.42, 0.0, "x", color=GUIDE_COLOR, fontsize=13, ha="left")

    draw_interval(ax, 1.1, 3.8, 1.0, "A", BODY_A_COLOR, BODY_A_COLOR, alpha=0.28)
    draw_interval(ax, 2.6, 5.2, 1.9, "B", BODY_B_COLOR, BODY_B_COLOR, alpha=0.28)
    draw_interval(ax, 5.8, 7.4, 2.6, "C", NEUTRAL_FILL, NEUTRAL_EDGE, alpha=0.9)
    draw_interval(ax, 6.5, 8.6, 0.55, "D", NEUTRAL_FILL, NEUTRAL_EDGE, alpha=0.9)

    overlap_x0, overlap_x1 = 2.6, 3.8
    ax.add_patch(
        Rectangle(
            (overlap_x0, 0.72),
            overlap_x1 - overlap_x0,
            1.46,
            facecolor=MANIFOLD_COLOR,
            edgecolor="black",
            linewidth=2.4,
            alpha=0.42,
        )
    )

    sweep_x = 3.05
    ax.plot([sweep_x, sweep_x], [-0.12, 3.2], color=NORMAL_COLOR, linewidth=2.6)
    ax.annotate(
        "",
        xy=(sweep_x, 3.45),
        xytext=(sweep_x, 3.18),
        arrowprops=dict(arrowstyle="->", linewidth=2.0, color=NORMAL_COLOR),
    )

    add_label(ax, 2.45, 1.45, "candidate pair", color="black", fontsize=14)
    add_label(ax, 2.05, 2.45, "Body A", color=BODY_A_COLOR, fontsize=15)
    add_label(ax, 4.0, 2.78, "Body B", color=BODY_B_COLOR, fontsize=15)
    add_label(ax, 3.42, 3.45, "sweep line", color=NORMAL_COLOR, fontsize=14, ha="left")

    ax.annotate(
        "active list = {Body A, Body B}",
        xy=(sweep_x, 2.85),
        xytext=(0.95, 3.25),
        fontsize=13,
        color=TEXT_COLOR,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-|>", color=TEXT_COLOR, linewidth=1.5),
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.96),
    )

    ax.annotate(
        "emit a candidate pair once the x-intervals overlap;\nkeep it only if the y/z AABB test also survives",
        xy=(3.2, 1.48),
        xytext=(5.65, 3.1),
        fontsize=12.5,
        color=TEXT_COLOR,
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="-|>", color=TEXT_COLOR, linewidth=1.5),
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="none", alpha=0.96),
    )

    inset = ax.inset_axes([0.63, 0.07, 0.3, 0.28])
    style_axis(inset, equal=True)
    inset.set_xlim(0.0, 4.6)
    inset.set_ylim(0.0, 3.2)
    inset.add_patch(Rectangle((0.45, 0.6), 1.7, 1.05, facecolor=BODY_A_COLOR, edgecolor=BODY_A_COLOR, linewidth=2, alpha=0.28))
    inset.add_patch(Rectangle((1.45, 1.0), 1.75, 1.0, facecolor=BODY_B_COLOR, edgecolor=BODY_B_COLOR, linewidth=2, alpha=0.28))
    inset.add_patch(Rectangle((3.35, 0.18), 0.75, 0.6, facecolor=NEUTRAL_FILL, edgecolor=NEUTRAL_EDGE, linewidth=2, alpha=0.95))
    add_label(inset, 1.3, 2.55, "y/z overlap check", fontsize=12)
    add_label(inset, 1.0, 0.35, "A", color=BODY_A_COLOR, fontsize=12)
    add_label(inset, 2.55, 2.42, "B", color=BODY_B_COLOR, fontsize=12)
    add_label(inset, 3.72, 1.02, "D pruned", color=NEUTRAL_EDGE, fontsize=11)

    fig.text(
        0.5,
        0.05,
        "Implementation alignment: broad_phase() sorts AABB endpoints, maintains the active list, and emits candidate body pairs.",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#3c4043",
    )

    save_figure(fig, "aabb_sweep_phase.png", args.save, show=args.show)


if __name__ == "__main__":
    main()
