from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyvista as pv

from geometry.primitives import AABB3, box_3D
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
BODY_C_COLOR = "#2f9e44"
BODY_A_LABEL_DOWN = -3
BODY_B_LABEL_DOWN = 1.0
BODY_C_LABEL_DOWN = -1.2
BODY_A_LABEL_X = -0.25
BODY_B_LABEL_X = 0.5
BODY_C_LABEL_X = -0.4
GRID_PADDING_X = 0.3
GRID_PADDING_Y = 0.30
GRID_PADDING_Z = 0.25
GRID_PADDING_Y_POSITIVE_SHIFT = 1

# Toggle all scene labels without changing the rest of the visualization.
SHOW_LABELS = False


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


def projection_quad(aabb, plane: str, plane_value: float) -> np.ndarray:
    if plane == "xy":
        return np.array(
            [
                [aabb.min_x, aabb.min_y, plane_value],
                [aabb.max_x, aabb.min_y, plane_value],
                [aabb.max_x, aabb.max_y, plane_value],
                [aabb.min_x, aabb.max_y, plane_value],
            ],
            dtype=float,
        )
    if plane == "xz":
        return np.array(
            [
                [aabb.min_x, plane_value, aabb.min_z],
                [aabb.max_x, plane_value, aabb.min_z],
                [aabb.max_x, plane_value, aabb.max_z],
                [aabb.min_x, plane_value, aabb.max_z],
            ],
            dtype=float,
        )
    if plane == "yz":
        return np.array(
            [
                [plane_value, aabb.min_y, aabb.min_z],
                [plane_value, aabb.max_y, aabb.min_z],
                [plane_value, aabb.max_y, aabb.max_z],
                [plane_value, aabb.min_y, aabb.max_z],
            ],
            dtype=float,
        )
    raise ValueError(f"Unsupported projection plane: {plane}")


def add_projection_patch(
    plotter: pv.Plotter,
    aabb,
    plane: str,
    plane_value: float,
    color: str,
    *,
    opacity: float,
    edge_color: str | None = None,
    line_width: float = 2.0,
):
    verts = projection_quad(aabb, plane, plane_value)
    faces = np.hstack([[4, 0, 1, 2, 3]])
    patch = pv.PolyData(verts, faces)
    plotter.add_mesh(
        patch,
        color=color,
        opacity=opacity,
        show_edges=True,
        edge_color=(edge_color or color),
        line_width=line_width,
        smooth_shading=True,
    )


def add_invisible_padding_cubes(
    plotter: pv.Plotter,
    mins: np.ndarray,
    maxs: np.ndarray,
    *,
    marker_size: float,
):
    for center in (mins, maxs):
        marker = pv.Cube(
            center=tuple(np.asarray(center, dtype=float)),
            x_length=marker_size,
            y_length=marker_size,
            z_length=marker_size,
        )
        plotter.add_mesh(marker, opacity=0.0, show_edges=False, lighting=False)


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
    slight_yaw = np.deg2rad(18.0)
    boxB.position[3:] = np.array(
        [np.cos(0.5 * slight_yaw), 0.0, 0.0, np.sin(0.5 * slight_yaw)],
        dtype=float,
    )
    bodyC = make_context_body(
        center=(-4, 1.0, 1.3),
        quat=(0.08, 0.0, 0.0, 0.12),
        size=(1.55, 1.35, 1.15),
        body_id=3,
    )
    bodies = [boxA, boxB, bodyC]
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
    aabbC = bodyC.get_aabb()
    overlap_mins, overlap_maxs = overlap_bounds(aabbA, aabbB)
    has_overlap = bool(np.all(overlap_maxs > overlap_mins))

    plotter = pv.Plotter(window_size=(1600, 1200))
    plotter.set_background("white")
    plotter.add_axes()

    add_body(plotter, boxA, BODY_A_COLOR)
    add_body(plotter, boxB, BODY_B_COLOR)
    add_body(plotter, bodyC, BODY_C_COLOR, opacity=0.20, show_edges=False)
    add_aabb(plotter, aabbA, BODY_A_COLOR)
    add_aabb(plotter, aabbB, BODY_B_COLOR)
    add_aabb(plotter, aabbC, BODY_C_COLOR, fill_opacity=0.06, line_width=3.0)

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
    projection_eps = 0.012 * scale

    mins = np.min(scene_pts, axis=0)
    maxs = np.max(scene_pts, axis=0)
    center = 0.5 * (mins + maxs)
    span = np.maximum(maxs - mins, 1e-9)
    pad = np.array(
        [
            GRID_PADDING_X * span[0],
            GRID_PADDING_Y * span[1],
            GRID_PADDING_Z * span[2],
        ],
        dtype=float,
    )
    padded_mins = mins - pad
    padded_maxs = maxs + pad
    padded_maxs[1] += GRID_PADDING_Y_POSITIVE_SHIFT
    padded_center = 0.5 * (padded_mins + padded_maxs)
    add_invisible_padding_cubes(
        plotter,
        padded_mins,
        padded_maxs,
        marker_size=0.02 * scale,
    )
    plotter.show_grid(color="lightgray")
    x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    camera_dir = unit(np.array([1.55, -1.05, 0.8], dtype=float))
    body_a_center = np.array(boxA.position[:3], dtype=float)
    body_b_center = np.array(boxB.position[:3], dtype=float)
    body_c_center = np.array(bodyC.position[:3], dtype=float)
    aabb_a_center = aabb_center(aabbA)
    aabb_b_center = aabb_center(aabbB)
    aabb_c_center = aabb_center(aabbC)
    xy_plane_z = padded_mins[2] + projection_eps
    xz_plane_y = padded_maxs[1] - projection_eps
    yz_plane_x = padded_mins[0] + projection_eps

    add_projection_patch(plotter, aabbA, "xy", xy_plane_z, BODY_A_COLOR, opacity=0.12)
    add_projection_patch(plotter, aabbB, "xy", xy_plane_z, BODY_B_COLOR, opacity=0.12)
    add_projection_patch(plotter, aabbC, "xy", xy_plane_z, BODY_C_COLOR, opacity=0.10)
    add_projection_patch(plotter, aabbA, "xz", xz_plane_y, BODY_A_COLOR, opacity=0.10)
    add_projection_patch(plotter, aabbB, "xz", xz_plane_y, BODY_B_COLOR, opacity=0.10)
    add_projection_patch(plotter, aabbC, "xz", xz_plane_y, BODY_C_COLOR, opacity=0.08)
    add_projection_patch(plotter, aabbA, "yz", yz_plane_x, BODY_A_COLOR, opacity=0.10)
    add_projection_patch(plotter, aabbB, "yz", yz_plane_x, BODY_B_COLOR, opacity=0.10)
    add_projection_patch(plotter, aabbC, "yz", yz_plane_x, BODY_C_COLOR, opacity=0.08)

    body_a_label_pos = body_a_center + BODY_A_LABEL_X * scale * x_axis - BODY_A_LABEL_DOWN * z_axis
    body_b_label_pos = body_b_center + BODY_B_LABEL_X * scale * x_axis - BODY_B_LABEL_DOWN * z_axis
    body_c_label_pos = body_c_center + BODY_C_LABEL_X * scale * x_axis - BODY_C_LABEL_DOWN * z_axis
    label_a_pos = 0.5 * (body_a_label_pos + aabb_a_center) + 0.06 * scale * y_axis + 0.04 * scale * z_axis
    label_b_pos = 0.5 * (body_b_label_pos + aabb_b_center) + 0.02 * scale * y_axis + 0.03 * scale * z_axis
    label_c_pos = 0.5 * (body_c_label_pos + aabb_c_center) + 0.05 * scale * y_axis + 0.04 * scale * z_axis

    add_labels(
        plotter,
        [label_a_pos],
        ["Body/AABB A"],
        text_color=BODY_A_COLOR,
        enabled=SHOW_LABELS,
    )
    add_labels(
        plotter,
        [label_b_pos],
        ["Body/AABB B"],
        text_color=BODY_B_COLOR,
        enabled=SHOW_LABELS,
    )
    add_labels(
        plotter,
        [label_c_pos],
        ["Body/AABB C"],
        text_color=BODY_C_COLOR,
        enabled=SHOW_LABELS,
    )

    if has_overlap:
        overlap_center = 0.5 * (overlap_mins + overlap_maxs)
        overlap_aabb = AABB3(
            float(overlap_mins[0]),
            float(overlap_maxs[0]),
            float(overlap_mins[1]),
            float(overlap_maxs[1]),
            float(overlap_mins[2]),
            float(overlap_maxs[2]),
        )
        add_projection_patch(
            plotter,
            overlap_aabb,
            "xy",
            xy_plane_z + 0.35 * projection_eps,
            MANIFOLD_COLOR,
            opacity=0.18,
            edge_color=MANIFOLD_EDGE_COLOR,
            line_width=2.8,
        )
        add_projection_patch(
            plotter,
            overlap_aabb,
            "xz",
            xz_plane_y - 0.35 * projection_eps,
            MANIFOLD_COLOR,
            opacity=0.16,
            edge_color=MANIFOLD_EDGE_COLOR,
            line_width=2.8,
        )
        add_projection_patch(
            plotter,
            overlap_aabb,
            "yz",
            yz_plane_x + 0.35 * projection_eps,
            MANIFOLD_COLOR,
            opacity=0.16,
            edge_color=MANIFOLD_EDGE_COLOR,
            line_width=2.8,
        )
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
        padded_center + 4.4 * scale * camera_dir,
        padded_center,
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
