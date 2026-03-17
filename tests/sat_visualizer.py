from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyvista as pv

from py_solver import collisions
from tests.collision_pyvista_common import (
    BODY_A_COLOR,
    BODY_B_COLOR,
    MANIFOLD_COLOR,
    NORMAL_COLOR,
    TANGENT_COLOR,
    add_body,
    add_labeled_vector,
    add_labels,
    add_tube_segment,
    build_demo_bodies,
    choose_perpendicular,
    scene_scale,
    unit,
)


def project_interval(body, axis: np.ndarray, origin: np.ndarray) -> tuple[float, float]:
    corners = body.get_corners()
    vals = (corners - origin) @ unit(axis)
    return float(np.min(vals)), float(np.max(vals))


def aligned_face_axis(body, axis: np.ndarray) -> np.ndarray:
    axes = body.get_axes()
    idx = int(np.argmax(np.abs(axes @ unit(axis))))
    face_axis = axes[idx].astype(float)
    if np.dot(face_axis, axis) < 0.0:
        face_axis = -face_axis
    return unit(face_axis)


def main():
    boxA, boxB = build_demo_bodies()
    sat_axis, overlap, tag = collisions._sat_and_overlap(boxA, boxB)
    if sat_axis is None:
        raise RuntimeError("The demo bodies do not overlap, so SAT produced no contact axis.")

    sat_axis = unit(sat_axis)
    axis_A = aligned_face_axis(boxA, sat_axis)
    axis_B = aligned_face_axis(boxB, sat_axis)

    plotter = pv.Plotter(window_size=(1600, 1200))
    plotter.set_background("white")

    add_body(plotter, boxA, BODY_A_COLOR)
    add_body(plotter, boxB, BODY_B_COLOR)

    body_pts = np.vstack([boxA.get_corners(), boxB.get_corners()])
    scale = scene_scale(body_pts)
    centerA = boxA.position[:3]
    centerB = boxB.position[:3]
    patch_center = 0.5 * (centerA + centerB)

    perp = choose_perpendicular(sat_axis)
    lift = choose_perpendicular(sat_axis, preferred=perp)

    label_offset = 0.07 * scale
    body_label_offset = 0.23 * scale
    vector_length = 0.42 * scale
    interval_radius = 0.018 * scale

    add_labels(
        plotter,
        [centerA - lift * body_label_offset],
        ["Body A"],
        text_color=BODY_A_COLOR,
    )
    add_labels(
        plotter,
        [centerB + lift * body_label_offset],
        ["Body B"],
        text_color=BODY_B_COLOR,
    )

    add_labeled_vector(
        plotter,
        patch_center,
        sat_axis,
        "minimum-overlap axis n",
        NORMAL_COLOR,
        length=vector_length,
        label_offset=label_offset,
        font_size=18,
    )
    add_labeled_vector(
        plotter,
        centerA,
        axis_A,
        "A face axis",
        TANGENT_COLOR,
        length=0.78 * vector_length,
        label_offset=label_offset,
        font_size=18,
    )
    add_labeled_vector(
        plotter,
        centerB,
        axis_B,
        "B face axis",
        TANGENT_COLOR,
        length=0.78 * vector_length,
        label_offset=label_offset,
        font_size=18,
    )

    proj_origin = patch_center - 0.55 * scale * lift
    minA, maxA = project_interval(boxA, sat_axis, proj_origin)
    minB, maxB = project_interval(boxB, sat_axis, proj_origin)
    overlap_lo = max(minA, minB)
    overlap_hi = min(maxA, maxB)

    a_offset = 0.18 * scale * perp
    b_offset = -0.18 * scale * perp
    add_tube_segment(
        plotter,
        proj_origin + sat_axis * minA + a_offset,
        proj_origin + sat_axis * maxA + a_offset,
        BODY_A_COLOR,
        radius=interval_radius,
    )
    add_tube_segment(
        plotter,
        proj_origin + sat_axis * minB + b_offset,
        proj_origin + sat_axis * maxB + b_offset,
        BODY_B_COLOR,
        radius=interval_radius,
    )
    if overlap_hi > overlap_lo:
        add_tube_segment(
            plotter,
            proj_origin + sat_axis * overlap_lo,
            proj_origin + sat_axis * overlap_hi,
            MANIFOLD_COLOR,
            radius=1.12 * interval_radius,
        )

    add_labels(
        plotter,
        [proj_origin + sat_axis * (0.5 * (minA + maxA)) + a_offset + 0.10 * scale * lift],
        ["projection of Body A"],
        text_color=BODY_A_COLOR,
    )
    add_labels(
        plotter,
        [proj_origin + sat_axis * (0.5 * (minB + maxB)) + b_offset - 0.10 * scale * lift],
        ["projection of Body B"],
        text_color=BODY_B_COLOR,
    )
    if overlap_hi > overlap_lo:
        add_labels(
            plotter,
            [proj_origin + sat_axis * (0.5 * (overlap_lo + overlap_hi)) + 0.14 * scale * lift],
            ["minimum overlap"],
            text_color="black",
        )

    add_labels(
        plotter,
        [patch_center - 0.22 * scale * lift - 0.24 * scale * perp],
        [f"SAT tag: {tag}"],
        text_color="black",
        font_size=16,
    )

    camera_side = choose_perpendicular(sat_axis, preferred=np.array([0.0, 0.0, 1.0], dtype=float))
    camera_up = unit(np.cross(camera_side, sat_axis))
    camera_dir = unit(1.15 * camera_side + 0.45 * camera_up)
    plotter.camera_position = (
        patch_center + 3.3 * scale * camera_dir,
        patch_center,
        camera_up,
    )
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.5)

    print("SAT axis:", sat_axis)
    print("Overlap:", overlap)
    print("Tag:", tag)

    plotter.show(title="SAT narrow-phase visualization")


if __name__ == "__main__":
    main()
