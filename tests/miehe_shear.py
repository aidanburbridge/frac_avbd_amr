"""
Miehe-style shear fracture fixture loaded from STL.
"""

from __future__ import annotations

import numpy as np

import geometry.octree as oct
import geometry.voxelizer as vox
from util.pyvista_visualizer import SimulationSetup
from util.timestep import estimate_timestep
from util.voxel_assembly import VoxelAssembly


# -------------------- Geometry (meters) -------------------- #
MM = 1.0e-3
STL_PATH = r"C:\Users\aidan\Documents\TUM\Thesis\miehe shear.stl"
# Small 1 mm plate: keep the base grid fine enough to retain several voxels
# through the 0.1 mm thickness before AMR refinement.
VOX_RESOLUTION = 160

# Keep L_bar-style axis usage here: x = in-plane width, z = in-plane height,
# y = out-of-plane thickness.
# Nominal specimen size: 1.0 mm (x) x 1.0 mm (z) x 0.1 mm (y).
SPECIMEN_WIDTH = 1.0 * MM
SPECIMEN_HEIGHT = 1.0 * MM
SPECIMEN_THICKNESS = 0.1 * MM


# -------------------- Boundary-condition controls -------------------- #
BOTTOM_FIX_DEPTH = 0.10 * MM
TOP_LOAD_DEPTH = 0.10 * MM
LOAD_VELOCITY = np.array([5.0e-3, 0.0, 0.0], dtype=float)
# Optional explicit voxel-ID sets from util.selection_tool.
FIXED_VOXEL_IDS: tuple[int, ...] = ()
LOAD_VOXEL_IDS: tuple[int, ...] = ()
# FIXED_VOXEL_IDS = [ ... ]
# LOAD_VOXEL_IDS = [ ... ]

FIXED_VOXEL_IDS = [0, 730, 1460, 2190, 2920, 3650, 4380, 5110, 5840, 6570]
LOAD_VOXEL_IDS = [657, 1387, 2117, 2847, 3577, 4307, 5037, 5767, 6497, 7227]


# -------------------- Material / solver -------------------- #
DENSITY = 7800.0
PENALTY_GAIN = 1.0e7
E_MODULUS = 210000.0e6
NU = 0.30
# Bond-model calibration values chosen to approximate a brittle Miehe-style
# shear benchmark in this cohesive-bond voxel framework, not literal phase-field parameters.
GC_TARGET = 2.7e3
TENSILE_STRENGTH = 3.0e8
FRACTURE_TOUGHNESS = np.sqrt(E_MODULUS * GC_TARGET / (1.0 - NU ** 2))
# TODO: Calibrate TENSILE_STRENGTH and FRACTURE_TOUGHNESS against crack onset
# and force-displacement response for the final benchmark target.
REFINE_STRESS_THRESHOLD = 0.05 * TENSILE_STRENGTH

DT_RENDER = 1 / 60
ITER = 120
GRAV = 0.0
FRICTION = 0.0
STEPS = 10000
MAX_REF_LEVEL = 2
TIME_STEP_POLICY = "load"
TIME_STEP_USE_REFINED_SIZE = False
TIME_STEP_WAVE_SPEED = "dilatational"
TIME_STEP_CFL_SAFETY = 0.30
TIME_STEP_LOAD_SAFETY = 0.25

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

    shear = VoxelAssembly(boxes, bonds)
    for idx, body in enumerate(shear.bodies):
        body.body_id = idx

    valid_mask = np.asarray(
        amr_dict.get("valid_mask", np.ones(len(shear.bodies), dtype=bool)),
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

    bounds = shear.bounds()

    x_min, y_min, z_min = bounds.mins
    x_max, y_max, z_max = bounds.maxs
    x_span = x_max - x_min
    y_span = y_max - y_min
    z_span = z_max - z_min

    if x_span <= 0.0 or y_span <= 0.0 or z_span <= 0.0:
        raise ValueError("Invalid shear-specimen bounds from voxelized STL.")

    scale_z = z_span / SPECIMEN_HEIGHT

    if explicit_fixed_ids:
        fixed_targets = _select_bodies_by_ids(
            shear,
            explicit_fixed_ids,
            valid_mask=valid_mask,
            label="fixed",
        )
    else:
        fix_depth = max(BOTTOM_FIX_DEPTH * scale_z, 1.5 * raw_h)
        fixed_targets = _select_bodies_by_center_box(
            shear,
            x=(x_min, x_max),
            y=(y_min, y_max),
            z=(z_min, z_min + fix_depth),
        )

    if explicit_load_ids:
        load_targets = _select_bodies_by_ids(
            shear,
            explicit_load_ids,
            valid_mask=valid_mask,
            label="load",
        )
    else:
        load_depth = max(TOP_LOAD_DEPTH * scale_z, 1.5 * raw_h)
        z_lo = max(z_min, z_max - load_depth)
        load_targets = _select_bodies_by_center_box(
            shear,
            x=(x_min, x_max),
            y=(y_min, y_max),
            z=(z_lo, z_max),
        )

    if not fixed_targets:
        if explicit_fixed_ids:
            raise ValueError("No fixed voxels selected from explicit FIXED_VOXEL_IDS.")
        raise ValueError("No fixed support voxels selected. Increase BOTTOM_FIX_DEPTH or increase VOX_RESOLUTION.")
    if not load_targets:
        if explicit_load_ids:
            raise ValueError("No load voxels selected from explicit LOAD_VOXEL_IDS.")
        raise ValueError("No top-band load voxels selected. Adjust TOP_LOAD_DEPTH or increase VOX_RESOLUTION.")

    _set_targets_fixed(fixed_targets)
    _set_targets_kinematic_velocity(load_targets, LOAD_VELOCITY)

    time_step = estimate_timestep(
        density=DENSITY,
        young_modulus=E_MODULUS,
        poisson=NU,
        h_base=raw_h,
        max_ref_level=MAX_REF_LEVEL,
        load_velocity=LOAD_VELOCITY,
        tensile_strength=TENSILE_STRENGTH,
        use_refined_size=TIME_STEP_USE_REFINED_SIZE,
        policy=TIME_STEP_POLICY,
        wave_speed=TIME_STEP_WAVE_SPEED,
        cfl_safety=TIME_STEP_CFL_SAFETY,
        load_safety=TIME_STEP_LOAD_SAFETY,
    )
    dt_physics = time_step.recommended_dt
    steps_per_export = max(1, int(DT_RENDER / dt_physics))

    print(f"Miehe shear voxels: {len(shear.bodies)}")
    print(f"Miehe shear bonds: {len(bonds)}")
    if explicit_fixed_ids:
        print(f"Fixed voxels: {len(fixed_targets)} (explicit FIXED_VOXEL_IDS)")
    else:
        print(f"Fixed voxels: {len(fixed_targets)}")
    if explicit_load_ids:
        print(f"Load voxels: {len(load_targets)} (explicit LOAD_VOXEL_IDS)")
    else:
        print(
            f"Load voxels: {len(load_targets)} "
            f"(z=[{z_lo:.4f}, {z_max:.4f}], vx={LOAD_VELOCITY[0]:.4f})"
        )
    print(
        f"Time step: {dt_physics:.6e} s "
        f"({time_step.chosen_limit}; wave={time_step.dt_wave:.6e}, "
        f"load={time_step.dt_load if time_step.dt_load is not None else float('nan'):.6e})"
    )

    return SimulationSetup(
        bodies=shear.bodies,
        constraints=bonds,
        dt=dt_physics,
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
            "specimen_width": SPECIMEN_WIDTH,
            "specimen_height": SPECIMEN_HEIGHT,
            "specimen_thickness": SPECIMEN_THICKNESS,
            "bottom_fix_depth": BOTTOM_FIX_DEPTH,
            "top_load_depth": TOP_LOAD_DEPTH,
            "support_band_thickness": BOTTOM_FIX_DEPTH,
            "load_band_thickness": TOP_LOAD_DEPTH,
            "loading_velocity": LOAD_VELOCITY.tolist(),
            "refine_stress_threshold": REFINE_STRESS_THRESHOLD,
            "fixed_voxel_ids": list(explicit_fixed_ids),
            "load_voxel_ids": list(explicit_load_ids),
            "dt_physics": dt_physics,
            "dt_render": DT_RENDER,
            **time_step.to_metadata(),
        },
        headless_steps=STEPS,
        headless_kwargs={
            "steps_per_export": steps_per_export,
            "show_progress": True,
        },
    )


if __name__ == "__main__":
    from util.pyvista_visualizer import run_simulation

    run_simulation(build_setup())
