"""
Legacy L-bar setup based on output/L_bar/run_034, rerun with physical scaling.

This keeps the legacy direct-loading style and the legacy support/load voxel-ID
lists from run_034, but it uses the current physical-unit scaling and metadata
conventions so postprocessing reports lengths in meters instead of exported
coordinate units.
"""

from __future__ import annotations

import numpy as np

import geometry.octree as oct
import geometry.voxelizer as vox
from util.engine import SimulationSetup
from util.voxel_assembly import VoxelAssembly


# -------------------- Geometry (meters) -------------------- #
MM = 1.0e-3
STL_PATH = r"C:\Users\aidan\Documents\TUM\Thesis\L bar fracture.stl"
VOX_RESOLUTION = 200
FLOOD_FILL = False


# -------------------- Boundary-condition controls -------------------- #
OUTER_DIM = 500.0 * MM
INNER_STEP = 250.0 * MM
LOAD_OFFSET_FROM_RIGHT = 30.0 * MM
BOTTOM_FIX_DEPTH = 20.0 * MM
LOAD_PATCH_WIDTH = 20.0 * MM
LOAD_BAND_THICKNESS = 20.0 * MM

# Legacy run_034 used direct specimen loading rather than the newer contact
# indenter. Keep that direct-loading mode here.
LOAD_VELOCITY = np.array([0.0, 5.0 * MM, 0.0], dtype=float)

# Legacy run_034 explicit selections.
FIXED_VOXEL_IDS = (
    0,
    17,
    34,
    59,
    618,
    643,
    668,
    709,
    1620,
    1645,
    1670,
    1711,
    2622,
    2647,
    2672,
    2713,
    3624,
    3649,
    3674,
    3715,
)
LOAD_VOXEL_IDS = (
    5760,
    5801,
    5842,
    5883,
    6138,
    6179,
    6220,
    6261,
    6516,
    6557,
    6598,
    6639,
    6894,
    6935,
    6976,
    7017,
)

# If the old IDs no longer match the current voxelization exactly, fall back to
# the old coordinate-box logic so the script remains runnable.
ALLOW_SELECTION_FALLBACK = True


# -------------------- Material / solver -------------------- #
DENSITY = 1150.0
PENALTY_GAIN = 1.0e6
E_MODULUS = 2.0e9
NU = 0.30
TENSILE_STRENGTH = 8.0e7
FRACTURE_TOUGHNESS = 5.0e4
REFINE_STRESS_THRESHOLD = 0.05 * TENSILE_STRENGTH

DT_PHYSICS = 1 / 4000
DT_RENDER = 1 / 60
STEPS_PER_EXPORT = max(1, int(DT_RENDER / DT_PHYSICS))
ITER = 80
GRAV = 0.0
FRICTION = 0.0
STEPS = 7000
MAX_REF_LEVEL = 2

PYTHON_SOLVER_PARAMS = {
    "mu": 0.2,
    "post_stabilize": False,
    "beta": 100.0,
    "alpha": 0.95,
    "gamma": 0.99,
    "debug_contacts": False,
    "criteria_refine_stress_threshold": REFINE_STRESS_THRESHOLD,
    "criteria_refine_stress_exclude_kinematic": True,
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
        body.static = False
        body.mass = np.inf
        body.velocity[:3] = vel
        body.velocity[3:] = 0.0
        body.prev_vel = body.velocity.copy()


def _legacy_fixed_box_targets(lbar: VoxelAssembly, phys_h: float) -> list:
    bounds = lbar.bounds()
    x_min, _, z_min = bounds.mins
    x_corner = x_min + INNER_STEP
    fix_depth = max(BOTTOM_FIX_DEPTH, 1.5 * phys_h)
    return _select_bodies_by_center_box(
        lbar,
        x=(x_min, x_corner + 0.5 * phys_h),
        z=(z_min, z_min + fix_depth),
    )


def _legacy_load_box_targets(lbar: VoxelAssembly, phys_h: float) -> list:
    bounds = lbar.bounds()
    x_min, _, z_min = bounds.mins
    x_max, _, z_max = bounds.maxs
    x_corner = x_min + INNER_STEP
    z_inner = z_min + INNER_STEP

    x_center = x_max - LOAD_OFFSET_FROM_RIGHT
    patch_width = max(LOAD_PATCH_WIDTH, 2.0 * phys_h)
    x_half = 0.5 * patch_width
    x_lo = max(x_corner + 0.5 * phys_h, x_center - x_half)
    x_hi = min(x_max - 0.5 * phys_h, x_center + x_half)
    z_hi = min(z_max, z_inner + max(LOAD_BAND_THICKNESS, 1.5 * phys_h))

    return _select_bodies_by_center_box(
        lbar,
        x=(x_lo, x_hi),
        z=(z_inner, z_hi),
    )


def _resolve_targets(
    *,
    assembly: VoxelAssembly,
    ids: tuple[int, ...],
    valid_mask: np.ndarray,
    label: str,
    fallback_targets_fn,
) -> tuple[list, str]:
    try:
        return (
            _select_bodies_by_ids(
                assembly,
                ids,
                valid_mask=valid_mask,
                label=label,
            ),
            "explicit_ids",
        )
    except ValueError as exc:
        if not ALLOW_SELECTION_FALLBACK:
            raise
        targets = fallback_targets_fn()
        if not targets:
            raise ValueError(f"{label} fallback selection also failed after ID error: {exc}") from exc
        print(f"{label.title()} selection fell back to coordinate-box logic: {exc}")
        return targets, "coordinate_box_fallback"


def build_setup(sync_bodies: bool = True) -> SimulationSetup:
    stlvox = vox.STLVoxelizer(STL_PATH, flood_fill=FLOOD_FILL)
    occ, raw_origin, raw_h = stlvox.voxelize_to_resolution(VOX_RESOLUTION)
    raw_extents = np.asarray(stlvox.mesh.extents, dtype=float)
    raw_longest = float(np.max(raw_extents))
    if raw_longest <= 0.0:
        raise ValueError(f"Invalid STL extents for legacy L-panel benchmark: {raw_extents}")

    # Use the current physical scaling convention: map the longest STL span to
    # the prescribed 500 mm outer size.
    panel_scale = OUTER_DIM / raw_longest
    phys_origin = raw_origin * panel_scale
    phys_h = raw_h * panel_scale

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
        body_origin=phys_origin,
        body_h_base=phys_h,
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

    fixed_targets, fixed_selection_strategy = _resolve_targets(
        assembly=lbar,
        ids=FIXED_VOXEL_IDS,
        valid_mask=valid_mask,
        label="fixed",
        fallback_targets_fn=lambda: _legacy_fixed_box_targets(lbar, phys_h),
    )
    load_targets, load_selection_strategy = _resolve_targets(
        assembly=lbar,
        ids=LOAD_VOXEL_IDS,
        valid_mask=valid_mask,
        label="load",
        fallback_targets_fn=lambda: _legacy_load_box_targets(lbar, phys_h),
    )

    overlap_ids = sorted(
        set(int(body.body_id) for body in fixed_targets).intersection(
            int(body.body_id) for body in load_targets
        )
    )
    if overlap_ids:
        raise ValueError(
            f"Legacy fixed and load selections overlap after resolution: {overlap_ids[:20]}"
            f"{' ...' if len(overlap_ids) > 20 else ''}"
        )

    _set_targets_fixed(fixed_targets)
    _set_targets_kinematic_velocity(load_targets, LOAD_VELOCITY)

    fixed_body_ids = sorted(int(body.body_id) for body in fixed_targets)
    load_body_ids = sorted(int(body.body_id) for body in load_targets)

    print(f"L-bar legacy run_034 bodies: {len(lbar.bodies)}")
    print(f"L-bar legacy run_034 bonds: {len(bonds)}")
    print(f"Fixed targets: {len(fixed_targets)} ({fixed_selection_strategy})")
    print(f"Load targets: {len(load_targets)} ({load_selection_strategy})")

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
            "benchmark_name": "L-panel benchmark",
            "configuration_name": "legacy_run_034_direct_load_physical",
            "legacy_reference_run": "output/L_bar/run_034",
            "stl_path": STL_PATH,
            "vox_resolution": VOX_RESOLUTION,
            "raw_h_base": raw_h,
            "h_base": phys_h,
            "raw_length_scale_to_m": panel_scale,
            "geometry_scaled_to_physical_units": True,
            "length_unit_label": "m",
            "displacement_unit_label": "m",
            "area_unit_label": "m^2",
            "stress_unit_label": "Pa",
            "energy_unit_label": "J",
            "max_ref_level": MAX_REF_LEVEL,
            "outer_dim": OUTER_DIM,
            "inner_step": INNER_STEP,
            "load_offset_from_right": LOAD_OFFSET_FROM_RIGHT,
            "load_patch_width": LOAD_PATCH_WIDTH,
            "load_band_thickness": LOAD_BAND_THICKNESS,
            "bottom_fix_depth": BOTTOM_FIX_DEPTH,
            "loading_velocity": LOAD_VELOCITY.tolist(),
            "loading_strategy": "legacy_direct_patch",
            "fixed_selection_strategy": fixed_selection_strategy,
            "load_selection_strategy": load_selection_strategy,
            "legacy_selection_fallback_used": bool(
                fixed_selection_strategy != "explicit_ids" or load_selection_strategy != "explicit_ids"
            ),
            "legacy_explicit_fixed_voxel_ids": list(FIXED_VOXEL_IDS),
            "legacy_explicit_load_voxel_ids": list(LOAD_VOXEL_IDS),
            "selected_fixed_body_ids": fixed_body_ids,
            "selected_load_body_ids": load_body_ids,
            "fixed_body_ids": fixed_body_ids,
            "load_body_ids": load_body_ids,
            "fixed_voxel_ids": list(FIXED_VOXEL_IDS),
            "load_voxel_ids": list(LOAD_VOXEL_IDS),
            "refine_stress_threshold": REFINE_STRESS_THRESHOLD,
            "dt_physics": DT_PHYSICS,
            "dt_render": DT_RENDER,
            "steps": STEPS,
            "steps_per_export": STEPS_PER_EXPORT,
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
