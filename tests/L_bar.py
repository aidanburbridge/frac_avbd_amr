"""
L-bar fracture fixture loaded from STL.
"""

from __future__ import annotations

import copy
import math
import numpy as np

import geometry.octree as oct
import geometry.voxelizer as vox
from util.engine import SimulationSetup
from util.voxel_assembly import Bounds, VoxelAssembly


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
LOAD_CONTACT_GAP_FACTOR = 0.05
LOAD_VELOCITY = np.array([0.0, 30 * MM, 0.0], dtype=float)
# Optional explicit voxel-ID sets from util.selection_tool.
FIXED_VOXEL_IDS = [0, 17, 568, 593, 1488, 1513, 2408, 2433, 3328, 3353, 4248, 4273]
#LOAD_VOXEL_IDS = [6548, 6589, 6630, 6671, 7008, 7049, 7090, 7131]

LOAD_VOXEL_IDS = [7008, 7049]

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
#ITER = 80
ITER = 20
GRAV = 0.0
FRICTION = 0.0
STEPS = 10000
MAX_REF_LEVEL = 2

def _build_solver_params() -> dict[str, float | bool]:
    return {
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


def _axis_label(axis: int) -> str:
    return "xyz"[int(axis)]


def _bounds_from_targets(targets: list) -> Bounds:
    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = -mins
    for body in targets:
        aabb = body.get_aabb()
        mins[0] = min(mins[0], aabb.min_x)
        maxs[0] = max(maxs[0], aabb.max_x)
        mins[1] = min(mins[1], aabb.min_y)
        maxs[1] = max(maxs[1], aabb.max_y)
        mins[2] = min(mins[2], aabb.min_z)
        maxs[2] = max(maxs[2], aabb.max_z)
    return Bounds(mins=mins, maxs=maxs)


def _compute_axis_gap(bounds_a: Bounds, bounds_b: Bounds, axis: int) -> float:
    if bounds_a.maxs[axis] <= bounds_b.mins[axis]:
        return float(bounds_b.mins[axis] - bounds_a.maxs[axis])
    if bounds_b.maxs[axis] <= bounds_a.mins[axis]:
        return float(bounds_a.mins[axis] - bounds_b.maxs[axis])
    return -float(
        min(
            bounds_a.maxs[axis] - bounds_b.mins[axis],
            bounds_b.maxs[axis] - bounds_a.mins[axis],
        )
    )


def _select_loading_surface_targets(
    targets: list,
    velocity: np.ndarray,
    voxel_size: float,
) -> tuple[list, int]:
    vel = np.asarray(velocity, dtype=float).reshape(-1)
    if vel.shape != (3,):
        raise ValueError(f"Expected a 3D loading velocity, got shape {vel.shape}.")

    axis = int(np.argmax(np.abs(vel)))
    axis_speed = float(vel[axis])
    if abs(axis_speed) <= 0.0:
        raise ValueError("LOAD_VELOCITY must prescribe motion along at least one axis.")

    coords = np.asarray([body.get_center()[axis] for body in targets], dtype=float)
    surface_coord = float(np.min(coords) if axis_speed > 0.0 else np.max(coords))
    tol = max(0.55 * float(voxel_size), 1.0e-12)

    if axis_speed > 0.0:
        surface_targets = [body for body, coord in zip(targets, coords) if coord <= surface_coord + tol]
    else:
        surface_targets = [body for body, coord in zip(targets, coords) if coord >= surface_coord - tol]

    if not surface_targets:
        raise ValueError("Failed to extract a surface slice from the selected load voxels.")
    return surface_targets, axis


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


def _build_loading_indenter(
    load_targets: list,
    velocity: np.ndarray,
    voxel_size: float,
    contact_gap: float,
) -> tuple[VoxelAssembly, list, int, float]:
    surface_targets, load_axis = _select_loading_surface_targets(load_targets, velocity, voxel_size)
    axis_speed = float(np.asarray(velocity, dtype=float)[load_axis])

    shift = np.zeros(3, dtype=float)
    shift[load_axis] = -math.copysign(float(voxel_size) + float(contact_gap), axis_speed)

    indenter = VoxelAssembly([copy.deepcopy(body) for body in surface_targets], constraints=[])
    indenter.translate(shift)
    _set_targets_kinematic_velocity(indenter.bodies, velocity)

    initial_gap = _compute_axis_gap(_bounds_from_targets(surface_targets), indenter.bounds(), load_axis)
    if initial_gap <= 0.0:
        raise ValueError(
            f"Loading indenter is initialized intersecting the specimen (gap={initial_gap:.6e} m). "
            "Increase LOAD_CONTACT_GAP_FACTOR or adjust the loading patch."
        )

    return indenter, surface_targets, load_axis, initial_gap


def build_setup(sync_bodies: bool = True) -> SimulationSetup:
    stlvox = vox.STLVoxelizer(STL_PATH, flood_fill=False)
    occ, raw_origin, raw_h = stlvox.voxelize_to_resolution(VOX_RESOLUTION)
    raw_extents = np.asarray(stlvox.mesh.extents, dtype=float)
    raw_longest = float(np.max(raw_extents))
    if raw_longest <= 0.0:
        raise ValueError(f"Invalid STL extents for L-panel benchmark: {raw_extents}")
    # Keep the current STL orientation and boundary-selection logic intact.
    # Only enforce that the longest STL axis maps to the prescribed outer size.
    panel_scale = OUTER_DIM / raw_longest
    phys_origin = raw_origin * panel_scale
    phys_h = raw_h * panel_scale

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

    x_corner = x_min + INNER_STEP
    z_inner = z_min + INNER_STEP

    if explicit_fixed_ids:
        fixed_targets = _select_bodies_by_ids(
            lbar,
            explicit_fixed_ids,
            valid_mask=valid_mask,
            label="fixed",
        )
    else:
        fix_depth = max(BOTTOM_FIX_DEPTH, 1.5 * phys_h)
        fixed_targets = _select_bodies_by_center_box(
            lbar,
            x=(x_min, x_corner + 0.5 * phys_h),
            z=(z_min, z_min + fix_depth),
        )

    # Contact loading patch at the inner horizontal edge, 30 mm from the right outer edge.
    x_center = x_max - LOAD_OFFSET_FROM_RIGHT
    patch_width = max(LOAD_PATCH_WIDTH, 2.0 * phys_h)
    x_half = 0.5 * patch_width
    x_lo = max(x_corner + 0.5 * phys_h, x_center - x_half)
    x_hi = min(x_max - 0.5 * phys_h, x_center + x_half)
    z_hi = min(z_max, z_inner + max(LOAD_BAND_THICKNESS, 1.5 * phys_h))

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

    specimen_load_ids = sorted(int(body.body_id) for body in load_targets)
    contact_gap = max(LOAD_CONTACT_GAP_FACTOR * phys_h, 1.0e-9)
    load_indenter, load_surface_targets, load_axis, initial_load_gap = _build_loading_indenter(
        load_targets,
        LOAD_VELOCITY,
        phys_h,
        contact_gap,
    )

    all_bodies = list(lbar.bodies) + list(load_indenter.bodies)
    all_bonds = list(bonds)
    for idx, body in enumerate(all_bodies):
        body.body_id = idx

    load_body_ids = [int(body.body_id) for body in load_indenter.bodies]
    amr_dict = oct.merge_amr_blocks([amr_dict, oct.make_flat_amr_block(len(load_indenter.bodies))])

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
    print(
        f"Load indenter voxels: {len(load_indenter.bodies)} "
        f"(surface slice={len(load_surface_targets)}, axis={_axis_label(load_axis)}, gap={initial_load_gap:.6e} m)"
    )

    return SimulationSetup(
        bodies=all_bodies,
        constraints=all_bonds,
        dt=DT_PHYSICS,
        iterations=ITER,
        gravity=GRAV,
        friction=FRICTION,
        sync_bodies=sync_bodies,
        python_solver_params=_build_solver_params(),
        amr_params=amr_dict,
        metadata={
            "benchmark_name": "L-panel benchmark",
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
            "density": DENSITY,
            "penalty_gain": PENALTY_GAIN,
            "E": E_MODULUS,
            "nu": NU,
            "tensile_strength": TENSILE_STRENGTH,
            "fracture_toughness": FRACTURE_TOUGHNESS,
            "max_ref_level": MAX_REF_LEVEL,
            "outer_dim": OUTER_DIM,
            "inner_step": INNER_STEP,
            "load_offset_from_right": LOAD_OFFSET_FROM_RIGHT,
            "load_patch_width": LOAD_PATCH_WIDTH,
            "load_band_thickness": LOAD_BAND_THICKNESS,
            "load_contact_gap_factor": LOAD_CONTACT_GAP_FACTOR,
            "load_contact_gap_m": initial_load_gap,
            "load_contact_axis": _axis_label(load_axis),
            "bottom_fix_depth": BOTTOM_FIX_DEPTH,
            "loading_velocity": LOAD_VELOCITY.tolist(),
            "loading_strategy": "contact_indenter",
            "refine_stress_threshold": REFINE_STRESS_THRESHOLD,
            "refine_stress_threshold_factor": (
                float(REFINE_STRESS_THRESHOLD) / float(TENSILE_STRENGTH)
                if float(TENSILE_STRENGTH) > 0.0
                else 0.0
            ),
            "fixed_voxel_ids": list(explicit_fixed_ids),
            "load_voxel_ids": specimen_load_ids,
            "load_body_ids": load_body_ids,
            "load_surface_voxel_ids": [int(body.body_id) for body in load_surface_targets],
            "load_indenter_body_count": len(load_indenter.bodies),
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
