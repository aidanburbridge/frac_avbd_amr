from __future__ import annotations

import numpy as np
import pyvista as pv

from geometry.primitives import AABB3, box_3D

BODY_A_COLOR = "#c43c39"
BODY_B_COLOR = "#2f6db3"
NORMAL_COLOR = "black"
TANGENT_COLOR = "#2f9e44"
MANIFOLD_COLOR = "#ffd84d"
MANIFOLD_EDGE_COLOR = "black"


def build_demo_bodies() -> tuple[box_3D, box_3D]:
    boxA = box_3D(
        (0.0, 0.5, 2.9),
        (0.2, 0.1, 0.4, 0.5),
        (0, 0, 0),
        (0, 0, 0),
        1000,
        10,
        (2, 2, 2),
        static=False,
    )
    boxB = box_3D(
        (0.8, 1.0, 1.2),
        (0.0, 0.0, 0.0, 0.0),
        (0, 0, 0),
        (0, 0, 0),
        1000,
        10,
        (2, 2, 2),
        static=False,
    )
    boxA.body_id = 1
    boxB.body_id = 2
    return boxA, boxB


def unit(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float).reshape(-1)
    return arr / (np.linalg.norm(arr) + 1e-12)


def choose_perpendicular(axis: np.ndarray, preferred: np.ndarray | None = None) -> np.ndarray:
    axis = unit(axis)
    ref = np.asarray(preferred if preferred is not None else [0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(axis, unit(ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    perp = np.cross(axis, ref)
    if np.linalg.norm(perp) < 1e-10:
        perp = np.cross(axis, np.array([1.0, 0.0, 0.0], dtype=float))
    return unit(perp)


def make_translation(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    trans_mat = np.eye(4, dtype=float)
    trans_mat[:3, :3] = R
    trans_mat[:3, 3] = t
    return trans_mat


def create_cube_mesh(body: box_3D):
    w, h, d = body.size
    cube = pv.Cube(center=(0, 0, 0), x_length=w, y_length=h, z_length=d)
    cube.transform(make_translation(body.rotmat(), body.position[:3]), inplace=True)
    return cube


def cube_from_aabb(aabb: AABB3):
    center = (
        0.5 * (aabb.min_x + aabb.max_x),
        0.5 * (aabb.min_y + aabb.max_y),
        0.5 * (aabb.min_z + aabb.max_z),
    )
    lengths = (
        aabb.max_x - aabb.min_x,
        aabb.max_y - aabb.min_y,
        aabb.max_z - aabb.min_z,
    )
    return pv.Cube(center=center, x_length=lengths[0], y_length=lengths[1], z_length=lengths[2])


def cube_from_bounds(mins: np.ndarray, maxs: np.ndarray):
    mins = np.asarray(mins, dtype=float)
    maxs = np.asarray(maxs, dtype=float)
    center = 0.5 * (mins + maxs)
    lengths = np.maximum(maxs - mins, 1e-9)
    return pv.Cube(center=center, x_length=lengths[0], y_length=lengths[1], z_length=lengths[2])


def add_body(plotter: pv.Plotter, body: box_3D, color: str, *, opacity: float = 0.28, show_edges: bool = False):
    plotter.add_mesh(
        create_cube_mesh(body),
        color=color,
        opacity=opacity,
        show_edges=show_edges,
        smooth_shading=True,
    )


def add_aabb(plotter: pv.Plotter, aabb: AABB3, color: str, *, fill_opacity: float = 0.08, line_width: float = 4.0):
    mesh = cube_from_aabb(aabb)
    plotter.add_mesh(mesh, color=color, opacity=fill_opacity, smooth_shading=True)
    plotter.add_mesh(mesh, color=color, style="wireframe", line_width=line_width)


def add_labels(
    plotter: pv.Plotter,
    points,
    labels,
    *,
    text_color: str = "black",
    font_size: int = 18,
    enabled: bool = True,
):
    if not enabled:
        return
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return
    plotter.add_point_labels(
        pts,
        labels,
        text_color=text_color,
        font_size=font_size,
        show_points=False,
        fill_shape=False,
        always_visible=True,
    )


def add_labeled_vector(
    plotter: pv.Plotter,
    origin: np.ndarray,
    direction: np.ndarray,
    label: str,
    color: str,
    *,
    length: float,
    label_offset: float,
    font_size: int = 20,
):
    origin = np.asarray(origin, dtype=float).reshape(1, 3)
    vec = unit(direction).reshape(1, 3) * float(length)
    plotter.add_arrows(origin, vec, mag=1.0, color=color)
    label_pos = origin[0] + vec[0] + unit(vec[0]) * float(label_offset)
    add_labels(plotter, [label_pos], [label], text_color=color, font_size=font_size)


def add_tube_segment(
    plotter: pv.Plotter,
    p0: np.ndarray,
    p1: np.ndarray,
    color: str,
    *,
    radius: float,
):
    line = pv.Line(np.asarray(p0, dtype=float), np.asarray(p1, dtype=float), resolution=1)
    plotter.add_mesh(line.tube(radius=float(radius)), color=color, smooth_shading=True)


def scene_scale(*point_sets) -> float:
    pts = np.vstack([np.asarray(p, dtype=float).reshape(-1, 3) for p in point_sets if np.asarray(p).size > 0])
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    return max(float(np.linalg.norm(maxs - mins)), 1.0)
