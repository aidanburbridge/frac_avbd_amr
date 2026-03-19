from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyvista as pv

from tests.collision_pyvista_common import (
    BODY_A_COLOR,
    BODY_B_COLOR,
    NORMAL_COLOR,
    TANGENT_COLOR,
    add_body,
    add_labels,
    add_tube_segment,
    build_demo_bodies,
    scene_scale,
    unit,
)

GRID_PADDING_X = 0.55
GRID_PADDING_Y = 0.40
GRID_PADDING_Z = 0.32
GRID_PADDING_Y_POSITIVE_SHIFT = 0.55

BODY_A_LABEL_OFFSET = (-0.2, 0.0, 0.2)
BODY_B_LABEL_OFFSET = (0.2, 0.0, 0.1)
AXIS_TO_TEST_LABEL_OFFSET = (-0.3, 0.0, 0.05)
OVERLAP_LABEL_OFFSET = (-0.1, 0.0, -0.16)
PROJECTION_BODY_A_LABEL_OFFSET = (-0.3, 0.0, -0.07)
PROJECTION_BODY_B_LABEL_OFFSET = (0.0, 0.0, -0.10)

AXIS_RADIUS_SCALE = 0.0055
PROJECTION_RADIUS_SCALE = 0.013
OVERLAP_RADIUS_MULTIPLIER = 1.08
CYLINDER_GAP_SCALE = 0.004

# Toggle all scene labels without changing the rest of the visualization.
SHOW_LABELS = False


def xyz_offset(offset_xyz, scale: float, x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    dx, dy, dz = np.asarray(offset_xyz, dtype=float)
    return scale * (dx * x_axis + dy * y_axis + dz * z_axis)


def project_interval(body, axis: np.ndarray, origin: np.ndarray) -> tuple[float, float]:
    vals = (np.asarray(body.get_corners(), dtype=float) - np.asarray(origin, dtype=float)) @ unit(axis)
    return float(np.min(vals)), float(np.max(vals))


def main():
    boxA, boxB = build_demo_bodies()
    boxA.size = (2.0, 2.0, 2.0)
    boxB.size = (2.0, 2.0, 2.0)
    boxA.position[:3] = np.array([-1.0, 0.0, 3.1], dtype=float)
    boxB.position[:3] = np.array([1.25, 0.2, 4.1], dtype=float)
    boxA.position[3:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    slight_yaw = np.deg2rad(42.0)
    boxB.position[3:] = np.array(
        [np.cos(0.5 * slight_yaw), 0.0, np.sin(0.5 * slight_yaw), 0.0],
        dtype=float,
    )

    plotter = pv.Plotter(window_size=(1600, 1200))
    plotter.set_background("white")
    plotter.add_axes()

    add_body(plotter, boxA, BODY_A_COLOR, opacity=0.3, show_edges=True)
    add_body(plotter, boxB, BODY_B_COLOR, opacity=0.3, show_edges=True)

    body_pts = np.vstack([boxA.get_corners(), boxB.get_corners()])
    scene_pts = body_pts.copy()
    scale = scene_scale(scene_pts)

    mins = np.min(scene_pts, axis=0)
    maxs = np.max(scene_pts, axis=0)
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

    x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    test_axis = x_axis
    camera_dir = unit(np.array([0.16, -1.0, 0.06], dtype=float))
    body_a_center = np.asarray(boxA.position[:3], dtype=float)
    body_b_center = np.asarray(boxB.position[:3], dtype=float)

    add_labels(
        plotter,
        [body_a_center + xyz_offset(BODY_A_LABEL_OFFSET, scale, x_axis, y_axis, z_axis)],
        ["Body A"],
        text_color=BODY_A_COLOR,
        font_size=24,
        enabled=SHOW_LABELS,
    )
    add_labels(
        plotter,
        [body_b_center + xyz_offset(BODY_B_LABEL_OFFSET, scale, x_axis, y_axis, z_axis)],
        ["Body B"],
        text_color=BODY_B_COLOR,
        font_size=24,
        enabled=SHOW_LABELS,
    )

    axis_mid = np.array(
        [
            padded_center[0],
            padded_center[1],
            padded_mins[2] + 0.12 * (padded_maxs[2] - padded_mins[2]),
        ],
        dtype=float,
    )
    axis_half = 0.50 * (padded_maxs[0] - padded_mins[0])
    axis_radius = AXIS_RADIUS_SCALE * scale
    add_tube_segment(
        plotter,
        axis_mid - axis_half * test_axis,
        axis_mid + axis_half * test_axis,
        NORMAL_COLOR,
        radius=axis_radius,
    )
    add_labels(
        plotter,
        [
            axis_mid
            - 0.60 * axis_half * test_axis
            + xyz_offset(AXIS_TO_TEST_LABEL_OFFSET, scale, x_axis, y_axis, z_axis)
        ],
        ["Axis to test"],
        text_color="black",
        font_size=16,
        enabled=SHOW_LABELS,
    )

    interval_origin = axis_mid
    minA, maxA = project_interval(boxA, test_axis, interval_origin)
    minB, maxB = project_interval(boxB, test_axis, interval_origin)
    overlap_lo = max(minA, minB)
    overlap_hi = min(maxA, maxB)
    projection_radius = PROJECTION_RADIUS_SCALE * scale
    overlap_radius = OVERLAP_RADIUS_MULTIPLIER * projection_radius
    cylinder_gap = CYLINDER_GAP_SCALE * scale
    red_offset = (axis_radius + projection_radius + cylinder_gap) * z_axis
    blue_offset = -(axis_radius + projection_radius + cylinder_gap) * z_axis
    green_offset = -(
        axis_radius
        + 2.0 * projection_radius
        + overlap_radius
        + 2.0 * cylinder_gap
    ) * z_axis

    add_tube_segment(
        plotter,
        interval_origin + test_axis * minA + red_offset,
        interval_origin + test_axis * maxA + red_offset,
        BODY_A_COLOR,
        radius=projection_radius,
    )
    add_tube_segment(
        plotter,
        interval_origin + test_axis * minB + blue_offset,
        interval_origin + test_axis * maxB + blue_offset,
        BODY_B_COLOR,
        radius=projection_radius,
    )
    if overlap_hi > overlap_lo:
        add_tube_segment(
            plotter,
            interval_origin + test_axis * overlap_lo + green_offset,
            interval_origin + test_axis * overlap_hi + green_offset,
            TANGENT_COLOR,
            radius=overlap_radius,
        )
        translation = overlap_hi - overlap_lo
        add_labels(
            plotter,
            [
                interval_origin
                + test_axis * (0.5 * (overlap_lo + overlap_hi))
                + xyz_offset(OVERLAP_LABEL_OFFSET, scale, x_axis, y_axis, z_axis)
            ],
            [f"Overlap = {translation:.2f}"],
            text_color=TANGENT_COLOR,
            font_size=16,
            enabled=SHOW_LABELS,
        )

    add_labels(
        plotter,
        [
            interval_origin
            + test_axis * (0.5 * (minA + maxA))
            + xyz_offset(PROJECTION_BODY_A_LABEL_OFFSET, scale, x_axis, y_axis, z_axis)
        ],
        ["Projection of Body A"],
        text_color=BODY_A_COLOR,
        font_size=16,
        enabled=SHOW_LABELS,
    )
    add_labels(
        plotter,
        [
            interval_origin
            + test_axis * (0.5 * (minB + maxB))
            + xyz_offset(PROJECTION_BODY_B_LABEL_OFFSET, scale, x_axis, y_axis, z_axis)
        ],
        ["Projection of Body B"],
        text_color=BODY_B_COLOR,
        font_size=16,
        enabled=SHOW_LABELS,
    )

    plotter.camera_position = (
        padded_center + 5.4 * scale * camera_dir,
        padded_center,
        z_axis,
    )
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.58)

    print("Test axis:", test_axis)
    print("Projected overlap:", max(0.0, overlap_hi - overlap_lo))

    plotter.show(title="SAT narrow-phase visualization")


if __name__ == "__main__":
    main()
