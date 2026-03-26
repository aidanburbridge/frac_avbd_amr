"""
Concrete wall projectile-impact fixture loaded from STL.
"""

from __future__ import annotations

import numpy as np

import geometry.octree as oct
import geometry.voxelizer as vox
from geometry.bond_data import BondData
from util.pyvista_visualizer import SimulationSetup
from util.voxel_assembly import VoxelAssembly


# -------------------- Geometry (meters) -------------------- #
MM = 1.0e-3
WALL_STL_PATH = r"C:\Users\aidan\Documents\TUM\Thesis\concrete wall.stl"
PROJECTILE_STL_PATH = r"C:\Users\aidan\Documents\TUM\Thesis\150 mm sphere.stl"

WALL_WIDTH = 1500.0 * MM
WALL_HEIGHT = 1500.0 * MM
WALL_THICKNESS = 300.0 * MM
PROJECTILE_DIAMETER = 150.0 * MM

# Selection-tool compatibility aliases for wall support picking.
STL_PATH = WALL_STL_PATH
WALL_TARGET_RESOLUTION = 200
PROJECTILE_TARGET_RESOLUTION = 360
FLOOD_FILL = False
REPAIR_MESH = True


# -------------------- Boundary-condition controls -------------------- #
# The impact direction is inferred from the wall STL bounding box:
# the smallest wall extent is treated as the thickness axis and therefore the
# impact normal. The remaining two axes span the wall plane.
IMPACT_FROM_NEGATIVE_THICKNESS_SIDE = True
INITIAL_PROJECTILE_GAP = 15.0 * MM

# Keep the wall mostly free; only a narrow rear perimeter frame is fixed to
# suppress rigid-body drift while leaving the impact face and most of the wall
# thickness unconstrained.
WALL_SUPPORT_EDGE_WIDTH = 75.0 * MM
WALL_SUPPORT_BACK_DEPTH = 40.0 * MM

# Conservative debugging baseline for fracture onset; increase later once the
# setup is stable and the contact/fracture response looks reasonable.
IMPACT_VELOCITY = 35.0

# Optional explicit wall-support voxel IDs from util.selection_tool.
FIXED_VOXEL_IDS: tuple[int, ...] = ()
# FIXED_VOXEL_IDS = [ ... ]


# -------------------- Material / solver -------------------- #
# Concrete-like cohesive-bond parameters for a localized impact benchmark.
# These are reasonable starting values for this voxel/bond model, not a fully
# calibrated constitutive law for a specific concrete mix.
WALL_DENSITY = 2400.0
WALL_PENALTY_GAIN = 1.0e7
WALL_E = 30.0e9
WALL_NU = 0.20
WALL_TENSILE_STRENGTH = 3.5e6
WALL_FRACTURE_TOUGHNESS = 1.5e6

# Give the projectile a much stiffer/stronger bond network and disable AMR so
# it behaves as an effectively rigid impactor while still using the same STL
# voxel-body pipeline as the wall.
PROJECTILE_DENSITY = 7850.0
PROJECTILE_PENALTY_GAIN = 5.0e7
PROJECTILE_E = 200.0e9
PROJECTILE_NU = 0.30
PROJECTILE_TENSILE_STRENGTH = 1.0e9
PROJECTILE_FRACTURE_TOUGHNESS = 5.0e7

# Coarse base grids keep this large wall benchmark practical. Start here for a
# first run, then increase resolution and refinement once the setup is stable.
MAX_REF_LEVEL = 2
PROJECTILE_MAX_REF_LEVEL = 0
REFINE_STRESS_THRESHOLD = 0.05 * WALL_TENSILE_STRENGTH

DT_PHYSICS = 5.81e-7
DT_RENDER  = 1.0e-4
STEPS_PER_EXPORT = max(1, int(DT_RENDER / DT_PHYSICS))
ITER = 50
GRAV = 0.0
FRICTION = 0.0
STEPS = 40000

PYTHON_SOLVER_PARAMS = {
    "mu": 0.0,
    "post_stabilize": False,
    "beta": 100.0,
    "alpha": 0.95,
    "gamma": 0.99,
    "debug_contacts": False,
    "criteria_refine_stress_threshold": REFINE_STRESS_THRESHOLD,
    "criteria_refine_stress_exclude_kinematic": True,
}


def _axis_label(axis: int) -> str:
    return "xyz"[int(axis)]


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


def _select_bodies_by_axis_box(
    assembly: VoxelAssembly,
    axis_limits: dict[int, tuple[float | None, float | None]],
) -> list:
    return _select_bodies_by_center_box(
        assembly,
        x=axis_limits.get(0),
        y=axis_limits.get(1),
        z=axis_limits.get(2),
    )


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


def _move_center_to(assembly: VoxelAssembly, target_center) -> None:
    delta = np.asarray(target_center, dtype=float) - assembly.bounds().center
    assembly.translate(delta)


def _set_targets_fixed(targets: list) -> None:
    for body in targets:
        body.set_static()
        body.velocity[:] = 0.0
        body.prev_vel = body.velocity.copy()


def _set_targets_dynamic_velocity(targets: list, velocity: np.ndarray) -> None:
    vel = np.asarray(velocity, dtype=float)
    for body in targets:
        body.static = False
        body.velocity[:3] = vel
        body.velocity[3:] = 0.0
        body.prev_vel = body.velocity.copy()


def _flatten_with_global_bond_indices(assemblies: list[VoxelAssembly]):
    all_bodies = []
    all_bonds = []

    body_offset = 0
    for assembly in assemblies:
        all_bodies.extend(assembly.bodies)
        for bond in assembly.constraints:
            if isinstance(bond, BondData):
                bond.idxA += body_offset
                bond.idxB += body_offset
            all_bonds.append(bond)
        body_offset += len(assembly.bodies)

    return all_bodies, all_bonds


def _contains_fn_factory(stl_voxelizer: vox.STLVoxelizer):
    def _contains_fn(points: np.ndarray) -> np.ndarray:
        return vox._contains_points_chunked(
            stl_voxelizer.mesh,
            np.asarray(points, dtype=float),
            chunk=200_000,
            show_progress=False,
        )

    return _contains_fn


def _voxelize_stl(
    stl_path: str,
    *,
    target_resolution: int,
    label: str,
) -> tuple[vox.STLVoxelizer, np.ndarray, np.ndarray, float, np.ndarray, int]:
    stlvox = vox.STLVoxelizer(
        stl_path,
        flood_fill=FLOOD_FILL,
        repair=REPAIR_MESH,
    )
    occ, raw_origin, raw_h = stlvox.voxelize_to_resolution(target_resolution)
    occ_count = int(np.count_nonzero(occ))
    if occ_count <= 0:
        raise ValueError(f"{label} voxelization produced zero active bodies.")

    raw_extents = np.asarray(stlvox.mesh.extents, dtype=float)
    if raw_extents.shape != (3,) or np.any(raw_extents <= 0.0):
        raise ValueError(f"Invalid STL extents for {label}: {raw_extents}")

    return (
        stlvox,
        occ,
        np.asarray(raw_origin, dtype=float),
        float(raw_h),
        raw_extents,
        occ_count,
    )


def _build_hierarchical_assembly(
    *,
    stlvox: vox.STLVoxelizer,
    occ: np.ndarray,
    raw_origin: np.ndarray,
    raw_h: float,
    body_scale: float,
    max_ref_level: int,
    density: float,
    penalty_gain: float,
    E: float,
    nu: float,
    tensile_strength: float,
    fracture_toughness: float,
    label: str,
) -> tuple[VoxelAssembly, list[BondData], dict, dict[str, object]]:
    contains_fn = _contains_fn_factory(stlvox)
    body_origin = np.asarray(raw_origin, dtype=float) * float(body_scale)
    body_h_base = float(raw_h) * float(body_scale)

    boxes, bonds, amr_dict = oct.build_hierarchical_bodies_bonds_amr(
        coarse_occ=occ,
        hierarchy_origin=raw_origin,
        hierarchy_h_base=raw_h,
        max_level=max_ref_level,
        contains_fn=contains_fn,
        body_origin=body_origin,
        body_h_base=body_h_base,
        density=density,
        penalty_gain=penalty_gain,
        static=False,
        show_progress=False,
        E=E,
        nu=nu,
        tensile_strength=tensile_strength,
        fracture_toughness=fracture_toughness,
    )

    if not boxes:
        raise ValueError(f"{label} hierarchy build produced zero bodies.")

    assembly = VoxelAssembly(boxes, bonds)
    for idx, body in enumerate(assembly.bodies):
        body.body_id = idx

    return (
        assembly,
        bonds,
        amr_dict,
        {
            "scale": float(body_scale),
            "raw_h": float(raw_h),
            "h_base": float(body_h_base),
            "raw_extents": np.asarray(stlvox.mesh.extents, dtype=float).tolist(),
            "mesh_bounds": np.asarray(stlvox.mesh.bounds, dtype=float).tolist(),
        },
    )


def _select_wall_support_targets(
    assembly: VoxelAssembly,
    *,
    impact_axis: int,
    support_face: str,
    edge_width: float,
    back_depth: float,
) -> list:
    bounds = assembly.bounds()
    mins = bounds.mins
    maxs = bounds.maxs
    inplane_axes = [axis for axis in range(3) if axis != int(impact_axis)]
    axis_a, axis_b = inplane_axes

    if support_face == "min":
        impact_limits = (mins[impact_axis], mins[impact_axis] + back_depth)
    elif support_face == "max":
        impact_limits = (maxs[impact_axis] - back_depth, maxs[impact_axis])
    else:
        raise ValueError(f"Unknown support face '{support_face}'")

    edge_boxes = [
        {
            impact_axis: impact_limits,
            axis_a: (mins[axis_a], mins[axis_a] + edge_width),
            axis_b: (mins[axis_b], maxs[axis_b]),
        },
        {
            impact_axis: impact_limits,
            axis_a: (maxs[axis_a] - edge_width, maxs[axis_a]),
            axis_b: (mins[axis_b], maxs[axis_b]),
        },
        {
            impact_axis: impact_limits,
            axis_a: (mins[axis_a], maxs[axis_a]),
            axis_b: (mins[axis_b], mins[axis_b] + edge_width),
        },
        {
            impact_axis: impact_limits,
            axis_a: (mins[axis_a], maxs[axis_a]),
            axis_b: (maxs[axis_b] - edge_width, maxs[axis_b]),
        },
    ]

    selected_map = {}
    for axis_limits in edge_boxes:
        for body in _select_bodies_by_axis_box(assembly, axis_limits):
            selected_map[id(body)] = body
    return list(selected_map.values())


def _compute_axis_gap(bounds_a, bounds_b, axis: int) -> float:
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


def build_setup(sync_bodies: bool = True) -> SimulationSetup:
    wall_vox, wall_occ, wall_raw_origin, wall_raw_h, wall_raw_extents, wall_occ_count = _voxelize_stl(
        WALL_STL_PATH,
        target_resolution=WALL_TARGET_RESOLUTION,
        label="wall",
    )
    wall_impact_axis_raw = int(np.argmin(wall_raw_extents))
    wall_inplane_axes_raw = [axis for axis in range(3) if axis != wall_impact_axis_raw]
    wall_scale_candidates = np.array(
        [
            WALL_WIDTH / wall_raw_extents[wall_inplane_axes_raw[0]],
            WALL_HEIGHT / wall_raw_extents[wall_inplane_axes_raw[1]],
            WALL_THICKNESS / wall_raw_extents[wall_impact_axis_raw],
        ],
        dtype=float,
    )
    wall_scale = float(np.median(wall_scale_candidates))

    wall, wall_bonds, wall_amr, wall_info = _build_hierarchical_assembly(
        stlvox=wall_vox,
        occ=wall_occ,
        raw_origin=wall_raw_origin,
        raw_h=wall_raw_h,
        body_scale=wall_scale,
        max_ref_level=MAX_REF_LEVEL,
        density=WALL_DENSITY,
        penalty_gain=WALL_PENALTY_GAIN,
        E=WALL_E,
        nu=WALL_NU,
        tensile_strength=WALL_TENSILE_STRENGTH,
        fracture_toughness=WALL_FRACTURE_TOUGHNESS,
        label="wall",
    )

    projectile_vox, projectile_occ, projectile_raw_origin, projectile_raw_h, projectile_raw_extents, projectile_occ_count = _voxelize_stl(
        PROJECTILE_STL_PATH,
        target_resolution=PROJECTILE_TARGET_RESOLUTION,
        label="projectile",
    )
    projectile_scale = float(PROJECTILE_DIAMETER / np.max(projectile_raw_extents))

    projectile, projectile_bonds, projectile_amr, projectile_info = _build_hierarchical_assembly(
        stlvox=projectile_vox,
        occ=projectile_occ,
        raw_origin=projectile_raw_origin,
        raw_h=projectile_raw_h,
        body_scale=projectile_scale,
        max_ref_level=PROJECTILE_MAX_REF_LEVEL,
        density=PROJECTILE_DENSITY,
        penalty_gain=PROJECTILE_PENALTY_GAIN,
        E=PROJECTILE_E,
        nu=PROJECTILE_NU,
        tensile_strength=PROJECTILE_TENSILE_STRENGTH,
        fracture_toughness=PROJECTILE_FRACTURE_TOUGHNESS,
        label="projectile",
    )

    _move_center_to(wall, np.zeros(3))
    wall_bounds = wall.bounds()
    wall_extents = wall_bounds.maxs - wall_bounds.mins
    impact_axis = int(np.argmin(wall_extents))
    inplane_axes = [axis for axis in range(3) if axis != impact_axis]

    if np.any(wall_extents <= 0.0):
        raise ValueError("Invalid wall bounds from voxelized STL.")

    support_edge_width = max(WALL_SUPPORT_EDGE_WIDTH, 1.5 * float(wall_info["h_base"]))
    support_back_depth = max(WALL_SUPPORT_BACK_DEPTH, 1.5 * float(wall_info["h_base"]))
    impact_face = "min" if IMPACT_FROM_NEGATIVE_THICKNESS_SIDE else "max"
    support_face = "max" if IMPACT_FROM_NEGATIVE_THICKNESS_SIDE else "min"

    wall_valid_mask = np.asarray(
        wall_amr.get("valid_mask", np.ones(len(wall.bodies), dtype=bool)),
        dtype=bool,
    )
    explicit_fixed_ids = tuple(sorted(set(int(i) for i in FIXED_VOXEL_IDS)))
    if explicit_fixed_ids:
        fixed_targets = _select_bodies_by_ids(
            wall,
            explicit_fixed_ids,
            valid_mask=wall_valid_mask,
            label="fixed",
        )
        support_strategy = "explicit_fixed_voxel_ids"
    else:
        fixed_targets = _select_wall_support_targets(
            wall,
            impact_axis=impact_axis,
            support_face=support_face,
            edge_width=support_edge_width,
            back_depth=support_back_depth,
        )
        support_strategy = "rear_perimeter_frame"

    if not fixed_targets:
        if explicit_fixed_ids:
            raise ValueError("No fixed voxels selected from explicit FIXED_VOXEL_IDS.")
        raise ValueError(
            "No wall support voxels selected. Increase WALL_SUPPORT_EDGE_WIDTH / "
            "WALL_SUPPORT_BACK_DEPTH or increase WALL_TARGET_RESOLUTION."
        )

    _set_targets_fixed(fixed_targets)

    projectile_bounds = projectile.bounds()
    projectile_half_extent = 0.5 * (projectile_bounds.maxs[impact_axis] - projectile_bounds.mins[impact_axis])
    wall_center = wall_bounds.center
    projectile_center = wall_center.copy()
    projectile_velocity = np.zeros(3, dtype=float)

    if IMPACT_FROM_NEGATIVE_THICKNESS_SIDE:
        projectile_center[impact_axis] = wall_bounds.mins[impact_axis] - INITIAL_PROJECTILE_GAP - projectile_half_extent
        projectile_velocity[impact_axis] = abs(IMPACT_VELOCITY)
    else:
        projectile_center[impact_axis] = wall_bounds.maxs[impact_axis] + INITIAL_PROJECTILE_GAP + projectile_half_extent
        projectile_velocity[impact_axis] = -abs(IMPACT_VELOCITY)

    _move_center_to(projectile, projectile_center)
    projectile_bounds = projectile.bounds()
    initial_gap = _compute_axis_gap(wall_bounds, projectile_bounds, impact_axis)
    if initial_gap <= 0.0:
        raise ValueError(
            f"Projectile is initialized intersecting the wall (gap={initial_gap:.6e} m). "
            "Increase INITIAL_PROJECTILE_GAP or adjust the impact-side convention."
        )

    _set_targets_dynamic_velocity(projectile.bodies, projectile_velocity)

    assemblies = [wall, projectile]
    all_bodies, all_bonds = _flatten_with_global_bond_indices(assemblies)
    for idx, body in enumerate(all_bodies):
        body.body_id = idx

    assembly_ids = np.array([int(getattr(b, "assembly_id", -1)) for b in all_bodies], dtype=np.int32)
    wall_mask = assembly_ids == int(wall.assembly_tag)
    max_ref_level_per_body = np.where(wall_mask, MAX_REF_LEVEL, PROJECTILE_MAX_REF_LEVEL).astype(np.int32)

    amr_dict = oct.merge_amr_blocks([wall_amr, projectile_amr])
    amr_dict["can_refine"] = np.logical_and(amr_dict["can_refine"], wall_mask)
    amr_dict["max_ref_level"] = int(max_ref_level_per_body.max())
    amr_dict["max_ref_level_per_body"] = max_ref_level_per_body

    if len(all_bodies) != len(amr_dict["level"]):
        raise ValueError("AMR/body count mismatch after wall/projectile block merge.")

    projectile_extents = projectile_bounds.maxs - projectile_bounds.mins

    print(f"Concrete wall bodies: {len(wall.bodies)}")
    print(f"Concrete wall bonds: {len(wall_bonds)}")
    print(f"Projectile bodies: {len(projectile.bodies)}")
    print(f"Projectile bonds: {len(projectile_bonds)}")
    if explicit_fixed_ids:
        print(f"Wall support voxels: {len(fixed_targets)} (explicit FIXED_VOXEL_IDS)")
    else:
        print(
            f"Wall support voxels: {len(fixed_targets)} "
            f"({support_strategy}, face={support_face}, edge={support_edge_width:.4f} m, back={support_back_depth:.4f} m)"
        )
    print(
        f"Impact axis: {_axis_label(impact_axis)} "
        f"(face={impact_face}, v0={projectile_velocity.tolist()}, gap={initial_gap:.4f} m)"
    )

    return SimulationSetup(
        bodies=all_bodies,
        constraints=all_bonds,
        dt=DT_PHYSICS,
        iterations=ITER,
        gravity=GRAV,
        friction=FRICTION,
        sync_bodies=sync_bodies,
        python_solver_params=PYTHON_SOLVER_PARAMS,
        amr_params=amr_dict,
        metadata={
            "wall_stl_path": WALL_STL_PATH,
            "projectile_stl_path": PROJECTILE_STL_PATH,
            "wall_nominal_dimensions_m": [WALL_WIDTH, WALL_HEIGHT, WALL_THICKNESS],
            "projectile_nominal_diameter_m": PROJECTILE_DIAMETER,
            "wall_actual_dimensions_m": wall_extents.tolist(),
            "projectile_actual_dimensions_m": projectile_extents.tolist(),
            "wall_vox_resolution": WALL_TARGET_RESOLUTION,
            "projectile_vox_resolution": PROJECTILE_TARGET_RESOLUTION,
            "wall_voxel_count_target": WALL_TARGET_RESOLUTION,
            "projectile_voxel_count_target": PROJECTILE_TARGET_RESOLUTION,
            "wall_occ_count": wall_occ_count,
            "projectile_occ_count": projectile_occ_count,
            "wall_h_base": wall_info["h_base"],
            "projectile_h_base": projectile_info["h_base"],
            "wall_raw_h_base": wall_info["raw_h"],
            "projectile_raw_h_base": projectile_info["raw_h"],
            "wall_scale": wall_info["scale"],
            "projectile_scale": projectile_info["scale"],
            "wall_raw_extents": wall_info["raw_extents"],
            "projectile_raw_extents": projectile_info["raw_extents"],
            "dt_physics": DT_PHYSICS,
            "dt_render": DT_RENDER,
            "steps": STEPS,
            "steps_per_export": STEPS_PER_EXPORT,
            "iterations": ITER,
            "impact_velocity_m_per_s": float(IMPACT_VELOCITY),
            "impact_velocity_vector_m_per_s": projectile_velocity.tolist(),
            "impact_axis": _axis_label(impact_axis),
            "wall_inplane_axes": [_axis_label(axis) for axis in inplane_axes],
            "impact_face": impact_face,
            "support_face": support_face,
            "support_strategy": support_strategy,
            "support_edge_width_m": support_edge_width,
            "support_back_depth_m": support_back_depth,
            "initial_projectile_gap_m": initial_gap,
            "impact_from_negative_thickness_side": IMPACT_FROM_NEGATIVE_THICKNESS_SIDE,
            "friction": FRICTION,
            "gravity": GRAV,
            "max_ref_level": MAX_REF_LEVEL,
            "projectile_max_ref_level": PROJECTILE_MAX_REF_LEVEL,
            "refine_stress_threshold": REFINE_STRESS_THRESHOLD,
            "fixed_voxel_ids": list(explicit_fixed_ids),
            "wall_density": WALL_DENSITY,
            "wall_penalty_gain": WALL_PENALTY_GAIN,
            "wall_E": WALL_E,
            "wall_nu": WALL_NU,
            "wall_tensile_strength": WALL_TENSILE_STRENGTH,
            "wall_fracture_toughness": WALL_FRACTURE_TOUGHNESS,
            "projectile_density": PROJECTILE_DENSITY,
            "projectile_penalty_gain": PROJECTILE_PENALTY_GAIN,
            "projectile_E": PROJECTILE_E,
            "projectile_nu": PROJECTILE_NU,
            "projectile_tensile_strength": PROJECTILE_TENSILE_STRENGTH,
            "projectile_fracture_toughness": PROJECTILE_FRACTURE_TOUGHNESS,
            "wall_scale_candidates": wall_scale_candidates.tolist(),
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
