"""
L-bar fracture fixture loaded from STL.
"""

from __future__ import annotations

import numpy as np

import geometry.octree as oct
import geometry.voxelizer as vox
from util.pyvista_visualizer import SimulationSetup
from util.voxel_assembly import VoxelAssembly


# -------------------- Geometry (meters) -------------------- #
MM = 1.0e-3
STL_PATH = r"C:\Users\aidan\Documents\TUM\Thesis\L bar fracture.stl"
VOX_RESOLUTION = 200


# -------------------- Boundary-condition controls -------------------- #
OUTER_DIM = 500.0 * MM
INNER_STEP = 250.0 * MM
LOAD_OFFSET_FROM_RIGHT = 30.0 * MM
BOTTOM_FIX_DEPTH = 20.0 * MM
LOAD_PATCH_WIDTH = 20.0 * MM
LOAD_BAND_THICKNESS = 20.0 * MM
LOAD_VELOCITY = np.array([0.0, 3.0, 0.0], dtype=float)
# Optional explicit voxel-ID sets from util.selection_tool.
FIXED_VOXEL_IDS: tuple[int, ...] = ()
LOAD_VOXEL_IDS: tuple[int, ...] = ()
FIXED_VOXEL_IDS = [0, 17, 34, 59, 618, 643, 668, 709, 1620, 1645, 1670, 1711, 2622, 2647, 2672, 2713, 3624, 3649, 3674, 3715]
LOAD_VOXEL_IDS = [5760, 5801, 6138, 6179, 6516, 6557]
# -------------------- Material / solver -------------------- #
DENSITY = 1150.0
PENALTY_GAIN = 1.0e6
E_MODULUS = 2.0e9
NU = 0.30
TENSILE_STRENGTH = 8.0e7
FRACTURE_TOUGHNESS = 5.0e4

DT_PHYSICS = 1 / 4000
DT_RENDER = 1 / 60
STEPS_PER_EXPORT = max(1, int(DT_RENDER / DT_PHYSICS))
ITER = 80
GRAV = 0.0
FRICTION = 0.0
STEPS = 8000
MAX_REF_LEVEL = 2

PYTHON_SOLVER_PARAMS = {
    "mu": 0.2,
    "post_stabilize": False,
    "beta": 100.0,
    "alpha": 0.95,
    "gamma": 0.99,
    "debug_contacts": False,
}

def _select_bodies_by_center_box(
    assembly: VoxelAssembly,
    *,
    x: tuple[float | None, float | None] | None = None,
    y: tuple[float | None, float | None] | None = None,
    z: tuple[float | None, float | None] | None = None,
) -> list:
    def _in_range(val: float, limits: tuple[float | None, float | None] | None) -> bool:
        if limits is None:
            return True
        lo, hi = limits
        if lo is not None and val < lo:
            return False
        if hi is not None and val > hi:
            return False
        return True

    selected = []
    for body in assembly.bodies:
        cx, cy, cz = body.get_center()
        if _in_range(cx, x) and _in_range(cy, y) and _in_range(cz, z):
            selected.append(body)
    return selected


def _select_bodies_by_ids(
    assembly: VoxelAssembly,
    ids: tuple[int, ...],
    *,
    valid_mask: np.ndarray | None = None,
    label: str = "selection",
) -> list:
    total = len(assembly.bodies)
    unique_ids = sorted(set(int(i) for i in ids))
    out_of_range = [idx for idx in unique_ids if idx < 0 or idx >= total]
    if out_of_range:
        preview = out_of_range[:20]
        suffix = " ..." if len(out_of_range) > len(preview) else ""
        raise ValueError(f"{label} IDs out of range for {total} bodies: {preview}{suffix}")

    if valid_mask is not None:
        outside_valid = [idx for idx in unique_ids if not bool(valid_mask[idx])]
        if outside_valid:
            preview = outside_valid[:20]
            suffix = " ..." if len(outside_valid) > len(preview) else ""
            raise ValueError(f"{label} IDs include voxels outside the valid STL interior: {preview}{suffix}")

    return [assembly.bodies[idx] for idx in unique_ids]


def _set_targets_fixed(targets: list) -> None:
    for body in targets:
        body.set_static()
        body.velocity[:] = 0.0
        body.prev_vel = body.velocity.copy()


def _set_targets_kinematic_velocity(targets: list, velocity: np.ndarray) -> None:
    vel = np.asarray(velocity, dtype=float)
    for body in targets:
        # Match VoxelAssembly.set_boundary_velocity behavior for kinematic grips.
        body.static = False
        body.mass = np.inf
        body.velocity[:3] = vel
        body.velocity[3:] = 0.0
        body.prev_vel = body.velocity.copy()


def build_setup(sync_bodies: bool = True) -> SimulationSetup:
    stlvox = vox.STLVoxelizer(STL_PATH, flood_fill=False)
    occ, raw_origin, raw_h = stlvox.voxelize_to_resolution(VOX_RESOLUTION)

    # Keep hierarchy refinement clipped to the STL interior.
    def _contains_fn(pts: np.ndarray) -> np.ndarray:
        return vox._contains_points_chunked(
            stlvox.mesh,
            np.asarray(pts, dtype=float),
            chunk=200_000,
            show_progress=False,
        )

    boxes, bonds, amr_dict = oct.build_hierarchical_bodies_bonds_amr(
        coarse_occ=occ,
        hierarchy_origin=raw_origin,
        hierarchy_h_base=raw_h,
        max_level=MAX_REF_LEVEL,
        contains_fn=_contains_fn,
        body_origin=raw_origin,
        body_h_base=raw_h,
        density=DENSITY,
        penalty_gain=PENALTY_GAIN,
        static=False,
        show_progress=False,
        E=E_MODULUS,
        nu=NU,
        tensile_strength=TENSILE_STRENGTH,
        fracture_toughness=FRACTURE_TOUGHNESS,
    )

    lbar = VoxelAssembly(boxes, bonds)
    for idx, body in enumerate(lbar.bodies):
        body.body_id = idx

    valid_mask = np.asarray(
        amr_dict.get("valid_mask", np.ones(len(lbar.bodies), dtype=bool)),
        dtype=bool,
    )

    explicit_fixed_ids = tuple(sorted(set(int(i) for i in FIXED_VOXEL_IDS)))
    explicit_load_ids = tuple(sorted(set(int(i) for i in LOAD_VOXEL_IDS)))
    overlap_ids = sorted(set(explicit_fixed_ids).intersection(explicit_load_ids))
    if overlap_ids:
        raise ValueError(
            f"FIXED_VOXEL_IDS and LOAD_VOXEL_IDS overlap: {overlap_ids[:20]}"
            f"{' ...' if len(overlap_ids) > 20 else ''}"
        )

    bounds = lbar.bounds()

    x_min, _, z_min = bounds.mins
    x_max, _, z_max = bounds.maxs
    x_span = x_max - x_min
    z_span = z_max - z_min

    if x_span <= 0.0 or z_span <= 0.0:
        raise ValueError("Invalid L-bar bounds from voxelized STL.")

    # Drawn geometry ratios from 500/250/30 mm sketch.
    scale_x = x_span / OUTER_DIM
    scale_z = z_span / OUTER_DIM

    x_corner = x_min + INNER_STEP * scale_x
    z_inner = z_min + INNER_STEP * scale_z

    if explicit_fixed_ids:
        fixed_targets = _select_bodies_by_ids(
            lbar,
            explicit_fixed_ids,
            valid_mask=valid_mask,
            label="fixed",
        )
    else:
        fix_depth = max(BOTTOM_FIX_DEPTH * scale_z, 1.5 * raw_h)
        fixed_targets = _select_bodies_by_center_box(
            lbar,
            x=(x_min, x_corner + 0.5 * raw_h),
            z=(z_min, z_min + fix_depth),
        )

    # Displacement patch at inner horizontal edge, 30 mm from right outer edge.
    x_center = x_max - LOAD_OFFSET_FROM_RIGHT * scale_x
    patch_width = max(LOAD_PATCH_WIDTH * scale_x, 2.0 * raw_h)
    x_half = 0.5 * patch_width
    x_lo = max(x_corner + 0.5 * raw_h, x_center - x_half)
    x_hi = min(x_max - 0.5 * raw_h, x_center + x_half)
    z_hi = min(z_max, z_inner + max(LOAD_BAND_THICKNESS * scale_z, 1.5 * raw_h))

    if explicit_load_ids:
        load_targets = _select_bodies_by_ids(
            lbar,
            explicit_load_ids,
            valid_mask=valid_mask,
            label="load",
        )
    else:
        load_targets = _select_bodies_by_center_box(
            lbar,
            x=(x_lo, x_hi),
            z=(z_inner, z_hi),
        )

    if not fixed_targets:
        if explicit_fixed_ids:
            raise ValueError("No fixed voxels selected from explicit FIXED_VOXEL_IDS.")
        raise ValueError("No fixed support voxels selected. Increase BOTTOM_FIX_DEPTH or increase VOX_RESOLUTION.")
    if not load_targets:
        if explicit_load_ids:
            raise ValueError("No load voxels selected from explicit LOAD_VOXEL_IDS.")
        raise ValueError("No load-patch voxels selected. Adjust LOAD_OFFSET_FROM_RIGHT / LOAD_PATCH_WIDTH / VOX_RESOLUTION.")

    _set_targets_fixed(fixed_targets)
    _set_targets_kinematic_velocity(load_targets, LOAD_VELOCITY)

    print(f"L-bar voxels: {len(lbar.bodies)}")
    print(f"L-bar bonds: {len(bonds)}")
    if explicit_fixed_ids:
        print(f"Fixed voxels: {len(fixed_targets)} (explicit FIXED_VOXEL_IDS)")
    else:
        print(f"Fixed voxels: {len(fixed_targets)}")
    if explicit_load_ids:
        print(f"Load voxels: {len(load_targets)} (explicit LOAD_VOXEL_IDS)")
    else:
        print(
            f"Load voxels: {len(load_targets)} "
            f"(x=[{x_lo:.4f}, {x_hi:.4f}], z=[{z_inner:.4f}, {z_hi:.4f}])"
        )

    return SimulationSetup(
        bodies=lbar.bodies,
        constraints=bonds,
        dt=DT_PHYSICS,
        iterations=ITER,
        gravity=GRAV,
        friction=FRICTION,
        sync_bodies=sync_bodies,
        python_solver_params=PYTHON_SOLVER_PARAMS,
        amr_params=amr_dict,
        metadata={
            "stl_path": STL_PATH,
            "vox_resolution": VOX_RESOLUTION,
            "h_base": raw_h,
            "max_ref_level": MAX_REF_LEVEL,
            "outer_dim": OUTER_DIM,
            "inner_step": INNER_STEP,
            "load_offset_from_right": LOAD_OFFSET_FROM_RIGHT,
            "load_patch_width": LOAD_PATCH_WIDTH,
            "load_band_thickness": LOAD_BAND_THICKNESS,
            "bottom_fix_depth": BOTTOM_FIX_DEPTH,
            "fixed_voxel_ids": list(explicit_fixed_ids),
            "load_voxel_ids": list(explicit_load_ids),
        },
        headless_steps=STEPS,
        headless_kwargs={
            "steps_per_export": STEPS_PER_EXPORT,
            "show_progress": True,
        },
    )


if __name__ == "__main__":
    from util.pyvista_visualizer import run_simulation

    run_simulation(build_setup())
