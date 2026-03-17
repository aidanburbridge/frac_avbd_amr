from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

BODY_A_COLOR = "#c43c39"
BODY_B_COLOR = "#2f6db3"
NORMAL_COLOR = "black"
TANGENT_COLOR = "#2f9e44"
MANIFOLD_COLOR = "#ffd84d"
NEUTRAL_FILL = "#d8dde6"
NEUTRAL_EDGE = "#6b7280"
GUIDE_COLOR = "#8d949c"
TEXT_COLOR = "#1f2328"


def make_figure(figsize=(12, 7)):
    return plt.figure(figsize=figsize, facecolor="white", constrained_layout=True)


def add_titles(fig, title: str, subtitle: str | None = None):
    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.985, color=TEXT_COLOR)
    if subtitle:
        fig.text(0.5, 0.945, subtitle, ha="center", va="top", fontsize=13, color="#3c4043")


def style_axis(ax, *, equal: bool = False):
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if equal:
        ax.set_aspect("equal", adjustable="box")


def add_label(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    color: str = TEXT_COLOR,
    fontsize: int = 14,
    fontweight: str = "bold",
    ha: str = "center",
    va: str = "center",
):
    ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=fontsize,
        fontweight=fontweight,
        ha=ha,
        va=va,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.96),
    )


def save_figure(fig, default_name: str, save_path: str | None = None, *, dpi: int = 300, show: bool = False):
    path = Path(save_path) if save_path else Path("output") / "figures" / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved figure to {path.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(fig)
