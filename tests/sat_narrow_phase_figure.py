from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

try:
    from .collision_figure_style import (
        BODY_A_COLOR,
        BODY_B_COLOR,
        GUIDE_COLOR,
        MANIFOLD_COLOR,
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
        NORMAL_COLOR,
        TEXT_COLOR,
        add_label,
        add_titles,
        make_figure,
        save_figure,
        style_axis,
    )


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)


def obb_corners(center, size, angle_deg):
    cx, cy = center
    w, h = size
    theta = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)
    local = np.array(
        [
            [-0.5 * w, -0.5 * h],
            [0.5 * w, -0.5 * h],
            [0.5 * w, 0.5 * h],
            [-0.5 * w, 0.5 * h],
        ],
        dtype=float,
    )
    return local @ rot.T + np.array([cx, cy], dtype=float)


def face_axes(corners):
    edges = [corners[1] - corners[0], corners[3] - corners[0]]
    axes = []
    for edge in edges:
        axis = unit(np.array([-edge[1], edge[0]], dtype=float))
        if axis[0] < 0.0 or (abs(axis[0]) < 1e-12 and axis[1] < 0.0):
            axis = -axis
        axes.append(axis)
    return axes


def dedupe_axes(axes, tol=0.999):
    unique = []
    for axis in axes:
        if not any(abs(np.dot(axis, other)) > tol for other in unique):
            unique.append(axis)
    return unique


def projection_interval(points, axis):
    values = points @ axis
    return float(np.min(values)), float(np.max(values))


def sat_min_overlap_axis(points_a, points_b):
    axes = dedupe_axes(face_axes(points_a) + face_axes(points_b))
    best_axis = None
    best_overlap = None
    for axis in axes:
        min_a, max_a = projection_interval(points_a, axis)
        min_b, max_b = projection_interval(points_b, axis)
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap <= 0.0:
            continue
        if best_overlap is None or overlap < best_overlap:
            best_overlap = overlap
            best_axis = axis
    if best_axis is None:
        raise RuntimeError("The chosen OBBs do not overlap on any SAT axis.")
    center_dir = np.mean(points_b, axis=0) - np.mean(points_a, axis=0)
    if np.dot(center_dir, best_axis) < 0.0:
        best_axis = -best_axis
    return best_axis, best_overlap, axes


def draw_projection_inset(ax, points_a, points_b, axis):
    style_axis(ax, equal=False)
    min_a, max_a = projection_interval(points_a, axis)
    min_b, max_b = projection_interval(points_b, axis)
    overlap_lo = max(min_a, min_b)
    overlap_hi = min(max_a, max_b)
    span_min = min(min_a, min_b) - 0.45
    span_max = max(max_a, max_b) + 0.45

    ax.set_xlim(span_min, span_max)
    ax.set_ylim(-0.9, 1.3)
    ax.plot([span_min + 0.1, span_max - 0.1], [0.0, 0.0], color=GUIDE_COLOR, linewidth=1.2)
    ax.plot([min_a, max_a], [0.45, 0.45], color=BODY_A_COLOR, linewidth=8, solid_capstyle="round")
    ax.plot([min_b, max_b], [-0.45, -0.45], color=BODY_B_COLOR, linewidth=8, solid_capstyle="round")
    ax.plot([overlap_lo, overlap_hi], [0.0, 0.0], color=MANIFOLD_COLOR, linewidth=11, solid_capstyle="butt")
    ax.plot([overlap_lo, overlap_hi], [0.0, 0.0], color="black", linewidth=1.7)

    add_label(ax, 0.5 * (min_a + max_a), 0.82, "projection of Body A", color=BODY_A_COLOR, fontsize=12)
    add_label(ax, 0.5 * (min_b + max_b), -0.82, "projection of Body B", color=BODY_B_COLOR, fontsize=12)
    add_label(ax, 0.5 * (overlap_lo + overlap_hi), 0.2, "minimum overlap", fontsize=12)


def main():
    parser = argparse.ArgumentParser(description="Create a thesis-style SAT narrow-phase figure.")
    parser.add_argument("--save", help="Optional save path. Defaults to output/figures/sat_narrow_phase.png")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively after saving.")
    args = parser.parse_args()

    body_a = obb_corners(center=(2.6, 2.5), size=(2.5, 1.5), angle_deg=14.0)
    body_b = obb_corners(center=(4.1, 2.65), size=(2.15, 1.45), angle_deg=-24.0)
    best_axis, best_overlap, axes = sat_min_overlap_axis(body_a, body_b)
    patch_center = 0.5 * (np.mean(body_a, axis=0) + np.mean(body_b, axis=0))

    fig = make_figure(figsize=(13, 8.2))
    add_titles(
        fig,
        "SAT Narrow Phase",
        "Project both bodies onto the candidate SAT axes; the smallest positive overlap becomes the contact direction.",
    )

    gs = fig.add_gridspec(2, 1, height_ratios=[4.6, 1.7], hspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    style_axis(ax, equal=True)
    ax.set_xlim(0.2, 6.7)
    ax.set_ylim(0.4, 5.3)

    ax.add_patch(Polygon(body_a, closed=True, facecolor=BODY_A_COLOR, edgecolor=BODY_A_COLOR, linewidth=2.2, alpha=0.28))
    ax.add_patch(Polygon(body_b, closed=True, facecolor=BODY_B_COLOR, edgecolor=BODY_B_COLOR, linewidth=2.2, alpha=0.28))

    axis_length = 3.3
    for axis in axes:
        p0 = patch_center - axis_length * axis
        p1 = patch_center + axis_length * axis
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linestyle="--", color=GUIDE_COLOR, linewidth=1.6, zorder=0)

    best_p0 = patch_center - axis_length * best_axis
    best_p1 = patch_center + axis_length * best_axis
    ax.annotate(
        "",
        xy=best_p1,
        xytext=best_p0,
        arrowprops=dict(arrowstyle="<->", linewidth=2.6, color=NORMAL_COLOR),
    )

    add_label(ax, np.mean(body_a[:, 0]) - 0.2, np.mean(body_a[:, 1]) + 1.35, "Body A", color=BODY_A_COLOR, fontsize=15)
    add_label(ax, np.mean(body_b[:, 0]) + 0.15, np.mean(body_b[:, 1]) + 1.25, "Body B", color=BODY_B_COLOR, fontsize=15)
    add_label(ax, patch_center[0] - 1.7, patch_center[1] + 1.55, "candidate axes", color=GUIDE_COLOR, fontsize=13)

    axis_label_pos = patch_center + 1.12 * best_axis + np.array([0.16, 0.12])
    add_label(ax, axis_label_pos[0], axis_label_pos[1], "minimum-overlap axis n", color=NORMAL_COLOR, fontsize=13)

    ax.annotate(
        f"overlap = {best_overlap:.2f}",
        xy=patch_center + 0.35 * best_axis,
        xytext=(5.0, 1.15),
        fontsize=12.5,
        color=TEXT_COLOR,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-|>", color=TEXT_COLOR, linewidth=1.4),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.96),
    )

    proj_ax = fig.add_subplot(gs[1, 0])
    draw_projection_inset(proj_ax, body_a, body_b, best_axis)
    add_label(proj_ax, 0.5 * (proj_ax.get_xlim()[0] + proj_ax.get_xlim()[1]), 1.05, "projection on the chosen SAT axis", fontsize=13)

    fig.text(
        0.5,
        0.045,
        "Implementation alignment: _sat_and_overlap() tests face axes and keeps the axis with the minimum positive projection overlap.",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#3c4043",
    )

    save_figure(fig, "sat_narrow_phase.png", args.save, show=args.show)


if __name__ == "__main__":
    main()
