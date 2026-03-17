from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyvista as pv

from geometry.primitives import box_3D
from py_solver import collisions
from tests.collision_pyvista_common import (
    BODY_A_COLOR,
    BODY_B_COLOR,
    MANIFOLD_COLOR,
    MANIFOLD_EDGE_COLOR,
    add_aabb,
    add_body,
    add_labels,
    build_demo_bodies,
    cube_from_bounds,
    scene_scale,
    unit,
)

NEUTRAL_BODY_COLOR = "#c7cdd6"
NEUTRAL_AABB_COLOR = "#8d949c"
BODY_A_LABEL_DOWN = 0.0
BODY_B_LABEL_DOWN = 1.0
BODY_A_LABEL_X = -0.2
BODY_B_LABEL_X = 0.1
AABB_A_LABEL_X = -0.2
AABB_B_LABEL_X = 0.1


def overlap_bounds(aabbA, aabbB):
    mins = np.array(
        [
            max(aabbA.min_x, aabbB.min_x),
            max(aabbA.min_y, aabbB.min_y),
            max(aabbA.min_z, aabbB.min_z),
        ],
        dtype=float,
    )
    maxs = np.array(
        [
            min(aabbA.max_x, aabbB.max_x),
            min(aabbA.max_y, aabbB.max_y),
            min(aabbA.max_z, aabbB.max_z),
        ],
        dtype=float,
    )
    return mins, maxs


def aabb_center(aabb) -> np.ndarray:
    return np.array(
        [
            0.5 * (aabb.min_x + aabb.max_x),
            0.5 * (aabb.min_y + aabb.max_y),
            0.5 * (aabb.min_z + aabb.max_z),
        ],
        dtype=float,
    )


def make_context_body(
    center,
    quat=(0.0, 0.0, 0.0, 0.0),
    size=(1.25, 1.25, 1.25),
    *,
    body_id: int,
):
    body = box_3D(
        center,
        quat,
        (0, 0, 0),
        (0, 0, 0),
        1000,
        10,
        size,
        static=False,
    )
    body.body_id = body_id
    return body


def main():
    boxA, boxB = build_demo_bodies()
    context_bodies = [
        make_context_body(center=(-4.1, -2.3, 0.0), quat=(0.18, 0.05, 0.12, 0.2), size=(1.15, 1.15, 1.15), body_id=3),
        make_context_body(center=(4.8, -2.8, 0.2), quat=(0.12, -0.08, 0.16, 0.22), size=(1.3, 1.0, 1.1), body_id=4),
        make_context_body(center=(5.6, 3.9, 4.2), quat=(0.14, 0.11, 0.05, -0.16), size=(1.0, 1.35, 1.0), body_id=5),
    ]
    bodies = [boxA, boxB, *context_bodies]
    broad_pairs = collisions.broad_phase(bodies, set())
    ab_pair = next(
        (
            pair for pair in broad_pairs
            if {pair[0].body_id, pair[1].body_id} == {boxA.body_id, boxB.body_id}
        ),
        None,
    )
    if ab_pair is None:
        raise RuntimeError("The demo bodies do not produce a broad-phase candidate pair.")

    aabbA = boxA.get_aabb()
    aabbB = boxB.get_aabb()
    overlap_mins, overlap_maxs = overlap_bounds(aabbA, aabbB)
    has_overlap = bool(np.all(overlap_maxs > overlap_mins))

    plotter = pv.Plotter(window_size=(1600, 1200))
    plotter.set_background("white")
    plotter.add_axes()
    plotter.show_grid(color="lightgray")

    add_body(plotter, boxA, BODY_A_COLOR)
    add_body(plotter, boxB, BODY_B_COLOR)
    add_aabb(plotter, aabbA, BODY_A_COLOR)
    add_aabb(plotter, aabbB, BODY_B_COLOR)
    for idx, body in enumerate(context_bodies, start=3):
        add_body(plotter, body, NEUTRAL_BODY_COLOR, opacity=0.18, show_edges=False)
        add_aabb(plotter, body.get_aabb(), NEUTRAL_AABB_COLOR, fill_opacity=0.04, line_width=2.5)

    body_pts = np.vstack([body.get_corners() for body in bodies])
    scene_pts = body_pts.copy()
    if has_overlap:
        overlap_mesh = cube_from_bounds(overlap_mins, overlap_maxs)
        plotter.add_mesh(
            overlap_mesh,
            color=MANIFOLD_COLOR,
            opacity=0.22,
            show_edges=True,
            edge_color=MANIFOLD_EDGE_COLOR,
            line_width=5,
            smooth_shading=True,
        )
        scene_pts = np.vstack([scene_pts, overlap_mins, overlap_maxs])

    scale = scene_scale(scene_pts)

    mins = np.min(scene_pts, axis=0)
    maxs = np.max(scene_pts, axis=0)
    center = 0.5 * (mins + maxs)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    camera_dir = unit(np.array([1.55, -1.05, 0.8], dtype=float))
    body_a_center = np.array(boxA.position[:3], dtype=float)
    body_b_center = np.array(boxB.position[:3], dtype=float)
    aabb_a_center = aabb_center(aabbA)
    aabb_b_center = aabb_center(aabbB)

    body_a_label_pos = body_a_center + BODY_A_LABEL_X * scale * x_axis - BODY_A_LABEL_DOWN * z_axis
    body_b_label_pos = body_b_center + BODY_B_LABEL_X * scale * x_axis - BODY_B_LABEL_DOWN * z_axis
    aabb_a_label_pos = aabb_a_center + (
        AABB_A_LABEL_X * scale * x_axis
        + 0.10 * scale * y_axis
        + 0.06 * scale * z_axis
    )
    aabb_b_label_pos = aabb_b_center + (
        AABB_B_LABEL_X * scale * x_axis
        + 0.04 * scale * y_axis
        + 0.03 * scale * z_axis
    )

    add_labels(
        plotter,
        [body_a_label_pos],
        ["Body A"],
        text_color=BODY_A_COLOR,
    )
    add_labels(
        plotter,
        [body_b_label_pos],
        ["Body B"],
        text_color=BODY_B_COLOR,
    )

    add_labels(
        plotter,
        [aabb_a_label_pos],
        ["AABB A"],
        text_color=BODY_A_COLOR,
        font_size=17,
    )
    add_labels(
        plotter,
        [aabb_b_label_pos],
        ["AABB B"],
        text_color=BODY_B_COLOR,
        font_size=17,
    )

    if has_overlap:
        overlap_center = 0.5 * (overlap_mins + overlap_maxs)
        candidate_label_pos = overlap_center + (
            + 0.08 * scale * x_axis
            + 0.16 * scale * y_axis
            + 0.08 * scale * z_axis
        )
        # add_labels(
        #     plotter,
        #     [candidate_label_pos],
        #     ["Broad-phase candidate pair"],
        #     text_color="black",
        #     font_size=17,
        # )
    else:
        overlap_center = center

    if has_overlap:
        x_overlap_label_pos = overlap_center + (
            + 0.14 * scale * x_axis
            - 0.02 * scale * y_axis
            - 0.06 * scale * z_axis
        )
        # add_labels(
        #     plotter,
        #     [x_overlap_label_pos],
        #     ["x-overlap survives"],
        #     text_color="black",
        #     font_size=16,
        # )

    plotter.camera_position = (
        overlap_center + 3.2 * scale * camera_dir,
        overlap_center,
        z_axis,
    )
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.45)

    print("Broad-phase pairs:", [(pair[0].body_id, pair[1].body_id) for pair in broad_pairs])
    print("AABB A:", aabbA)
    print("AABB B:", aabbB)
    if has_overlap:
        print("AABB overlap mins:", overlap_mins, "maxs:", overlap_maxs)

    plotter.show(title="AABB broad-phase visualization")


if __name__ == "__main__":
    main()
