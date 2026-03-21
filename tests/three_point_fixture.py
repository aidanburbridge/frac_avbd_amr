import geometry.voxelizer as vox
import geometry.octree as oct
import numpy as np

from geometry.bond_data import BondData
from util.timestep import calc_damping, estimate_timestep
from util.voxel_assembly import VoxelAssembly
from util.simulate import SimulationSetup


# STL geometry
BEAM_STL = r"C:\Users\aidan\Documents\TUM\Thesis\beam_3pt.stl"
ROLLER_STL = r"C:\Users\aidan\Documents\TUM\Thesis\roller_3pt.stl"

# Physical fixture dimensions
MM = 1.0e-3
BEAM_LENGTH = 35.0 * MM
BEAM_WIDTH = 4.0 * MM
BEAM_HEIGHT = 3.0 * MM
ROLLER_LENGTH = 6.0 * MM
ROLLER_DIAMETER = 5.0 * MM
SUPPORT_SPAN = 30.0 * MM

# Initial clearances
SUPPORT_GAP = 0.2 * MM
TOP_GAP = 0.2 * MM

# Voxelization settings (separate per part)
BEAM_VOXEL_RES = 140
ROLLER_VOXEL_RES = 90

# Shared solver params
DT_RENDER = 1 / 60
ITER = 180
GRAV = 0.0
FRICTION = 0.2
PULL_RATE = 0.002
E_MODULUS = 2e9
NU = 0.3
TENSILE_STRENGTH = 80e5
FRACTURE_TOUGHNESS = 5e4
DENSITY = 1150.0
PENALTY_GAIN = 1e6
STEPS = 1000
ZETA_DAMP = 0.1

# Per-body AMR caps
BEAM_MAX_REF_LEVEL = 2
ROLLER_MAX_REF_LEVEL = 0
TIME_STEP_POLICY = "load"
TIME_STEP_USE_REFINED_SIZE = False
TIME_STEP_WAVE_SPEED = "dilatational"
TIME_STEP_CFL_SAFETY = 0.30
TIME_STEP_LOAD_SAFETY = 0.25

PYTHON_SOLVER_PARAMS = {
    "mu": 0.3,
    "post_stabilize": True,
    "beta": 5.0e4,
    "alpha": 0.95,
    "gamma": 1.0,
    "debug_contacts": False,
}


def _build_voxel_assembly(
    stl_path: str,
    target_length: float,
    voxel_res: int,
    align_axis: str,
) -> tuple[VoxelAssembly, float]:
    stlvox = vox.STLVoxelizer(stl_path, flood_fill=False)
    occ, raw_origin, raw_h = stlvox.voxelize_to_resolution(voxel_res)

    raw_len = float(np.max(stlvox.mesh.extents))
    if raw_len <= 0.0:
        raise ValueError(f"Invalid STL extents for '{stl_path}'")

    scale = target_length / raw_len
    phys_h = raw_h * scale
    phys_origin = raw_origin * scale

    leaves, _ = oct.octree_from_occ(occ, h_base=raw_h)
    boxes, mapping, _ = oct.instantiate_boxes_from_tree(
        leaves,
        origin=phys_origin,
        h_base=phys_h,
        density=DENSITY,
        penalty_gain=PENALTY_GAIN,
        static=False,
        show_progress=False,
    )

    visco_val = calc_damping(DENSITY, phys_h, E_MODULUS, ZETA_DAMP)
    bonds = oct.build_constraints_from_tree(
        leaves,
        boxes,
        mapping,
        E=E_MODULUS,
        nu=NU,
        tensile_strength=TENSILE_STRENGTH,
        fracture_toughness=FRACTURE_TOUGHNESS,
        damping_val=visco_val,
    )

    assembly = VoxelAssembly(boxes, bonds)
    target_idx = {"x": 0, "y": 1, "z": 2}[align_axis.lower()]
    current_idx = int(np.argmax(np.asarray(stlvox.mesh.extents, dtype=float)))
    if current_idx != target_idx:
        v_curr = np.zeros(3, dtype=float)
        v_target = np.zeros(3, dtype=float)
        v_curr[current_idx] = 1.0
        v_target[target_idx] = 1.0
        assembly.rotate(axis=np.cross(v_curr, v_target), angle=90.0, degrees=True)

    return assembly, phys_h


def _move_center_to(assembly: VoxelAssembly, target_center) -> None:
    delta = np.asarray(target_center, dtype=float) - assembly.bounds().center
    assembly.translate(delta)


def _set_assembly_static(assembly: VoxelAssembly) -> None:
    for body in assembly.bodies:
        body.set_static()
        body.velocity[:] = 0.0
        body.prev_vel = body.velocity.copy()


def _set_assembly_kinematic(assembly: VoxelAssembly, linear_velocity) -> None:
    vel = np.asarray(linear_velocity, dtype=float)
    for body in assembly.bodies:
        body.static = False
        body.mass = np.inf
        body.inv_mass = 0.0
        body.inertia = np.diag([np.inf, np.inf, np.inf])
        body.inv_inertia = np.zeros((3, 3), dtype=float)
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


def build_setup() -> SimulationSetup:
    beam_vox = vox.STLVoxelizer(BEAM_STL, flood_fill=False)
    beam_occ, beam_raw_origin, beam_raw_h = beam_vox.voxelize_to_resolution(BEAM_VOXEL_RES)
    beam_raw_len = float(np.max(beam_vox.mesh.extents))
    if beam_raw_len <= 0.0:
        raise ValueError(f"Invalid STL extents for '{BEAM_STL}'")
    beam_scale = BEAM_LENGTH / beam_raw_len
    beam_h = beam_raw_h * beam_scale
    beam_origin = beam_raw_origin * beam_scale

    def _beam_contains_fn(pts: np.ndarray) -> np.ndarray:
        return vox._contains_points_chunked(
            beam_vox.mesh,
            np.asarray(pts, dtype=float),
            chunk=200_000,
            show_progress=False,
        )

    beam_boxes, beam_bonds, beam_amr = oct.build_hierarchical_bodies_bonds_amr(
        coarse_occ=beam_occ,
        hierarchy_origin=beam_raw_origin,
        hierarchy_h_base=beam_raw_h,
        max_level=BEAM_MAX_REF_LEVEL,
        contains_fn=_beam_contains_fn,
        body_origin=beam_origin,
        body_h_base=beam_h,
        density=DENSITY,
        penalty_gain=PENALTY_GAIN,
        E=E_MODULUS,
        nu=NU,
        tensile_strength=TENSILE_STRENGTH,
        fracture_toughness=FRACTURE_TOUGHNESS,
        damping_val=calc_damping(DENSITY, beam_h, E_MODULUS, ZETA_DAMP),
        show_progress=False,
    )
    beam = VoxelAssembly(beam_boxes, beam_bonds)
    beam.align_longest_axis("x")

    roller_base, roller_h = _build_voxel_assembly(
        ROLLER_STL,
        target_length=ROLLER_LENGTH,
        voxel_res=ROLLER_VOXEL_RES,
        align_axis="y",
    )

    _move_center_to(beam, np.zeros(3))

    left_support = roller_base.copy()
    right_support = roller_base.copy()
    top_indenter = roller_base.copy()

    beam_bounds = beam.bounds()
    beam_center = beam_bounds.center
    beam_length_actual = beam_bounds.maxs[0] - beam_bounds.mins[0]
    if SUPPORT_SPAN >= beam_length_actual:
        raise ValueError(
            f"SUPPORT_SPAN ({SUPPORT_SPAN}) must be less than beam length ({beam_length_actual})."
        )

    roller_radius = 0.5 * ROLLER_DIAMETER
    z_support = beam_bounds.mins[2] - roller_radius - SUPPORT_GAP
    z_top = beam_bounds.maxs[2] + roller_radius + TOP_GAP

    _move_center_to(left_support, (-0.5 * SUPPORT_SPAN, beam_center[1], z_support))
    _move_center_to(right_support, (0.5 * SUPPORT_SPAN, beam_center[1], z_support))
    _move_center_to(top_indenter, (0.0, beam_center[1], z_top))

    _set_assembly_static(left_support)
    _set_assembly_static(right_support)
    _set_assembly_kinematic(top_indenter, (0.0, 0.0, -abs(PULL_RATE)))

    assemblies = [beam, left_support, right_support, top_indenter]
    all_bodies, all_bonds = _flatten_with_global_bond_indices(assemblies)
    n_bodies = len(all_bodies)

    assembly_ids = np.array([int(getattr(b, "assembly_id", -1)) for b in all_bodies], dtype=np.int32)
    beam_tag = int(beam.assembly_tag)
    beam_mask = assembly_ids == beam_tag
    max_ref_level_per_body = np.where(beam_mask, BEAM_MAX_REF_LEVEL, ROLLER_MAX_REF_LEVEL).astype(np.int32)

    amr_blocks = [
        beam_amr,
        oct.make_flat_amr_block(len(left_support.bodies)),
        oct.make_flat_amr_block(len(right_support.bodies)),
        oct.make_flat_amr_block(len(top_indenter.bodies)),
    ]
    amr_dict = oct.merge_amr_blocks(amr_blocks)
    amr_dict["can_refine"] = np.logical_and(amr_dict["can_refine"], beam_mask)
    amr_dict["max_ref_level"] = int(max_ref_level_per_body.max())
    amr_dict["max_ref_level_per_body"] = max_ref_level_per_body

    if n_bodies != len(amr_dict["level"]):
        raise ValueError("AMR/body count mismatch after block merge")

    beam_time_step = estimate_timestep(
        density=DENSITY,
        young_modulus=E_MODULUS,
        poisson=NU,
        h_base=beam_h,
        max_ref_level=BEAM_MAX_REF_LEVEL,
        load_velocity=[0.0, 0.0, abs(PULL_RATE)],
        tensile_strength=TENSILE_STRENGTH,
        use_refined_size=TIME_STEP_USE_REFINED_SIZE,
        policy=TIME_STEP_POLICY,
        wave_speed=TIME_STEP_WAVE_SPEED,
        cfl_safety=TIME_STEP_CFL_SAFETY,
        load_safety=TIME_STEP_LOAD_SAFETY,
    )
    roller_time_step = estimate_timestep(
        density=DENSITY,
        young_modulus=E_MODULUS,
        poisson=NU,
        h_base=roller_h,
        max_ref_level=ROLLER_MAX_REF_LEVEL,
        load_velocity=None,
        tensile_strength=TENSILE_STRENGTH,
        use_refined_size=TIME_STEP_USE_REFINED_SIZE,
        policy="wave",
        wave_speed=TIME_STEP_WAVE_SPEED,
        cfl_safety=TIME_STEP_CFL_SAFETY,
        load_safety=TIME_STEP_LOAD_SAFETY,
    )
    dt_physics = beam_time_step.recommended_dt
    steps_per = max(1, int(DT_RENDER / dt_physics))

    print(f"Fixture bodies: beam={len(beam.bodies)} left={len(left_support.bodies)} right={len(right_support.bodies)} top={len(top_indenter.bodies)}")
    print(f"Fixture bonds: total={len(all_bonds)}")
    print(
        f"Beam time step: {dt_physics:.6e} s "
        f"({beam_time_step.chosen_limit}; wave={beam_time_step.dt_wave:.6e}, "
        f"load={beam_time_step.dt_load if beam_time_step.dt_load is not None else float('nan'):.6e})"
    )
    print(f"Roller wave limit: {roller_time_step.dt_wave:.6e} s")

    return SimulationSetup(
        bodies=all_bodies,
        constraints=all_bonds,
        dt=dt_physics,
        iterations=ITER,
        gravity=GRAV,
        friction=FRICTION,
        sync_bodies=True,
        python_solver_params=PYTHON_SOLVER_PARAMS,
        amr_params=amr_dict,
        metadata={
            "dt_physics": dt_physics,
            "dt_render": DT_RENDER,
            "beam_dims_m": [BEAM_LENGTH, BEAM_WIDTH, BEAM_HEIGHT],
            "roller_dims_m": [ROLLER_LENGTH, ROLLER_DIAMETER],
            "support_span_m": SUPPORT_SPAN,
            "pull_rate_m_per_s": PULL_RATE,
            "E": E_MODULUS,
            "nu": NU,
            "tensile_strength": TENSILE_STRENGTH,
            "fracture_toughness": FRACTURE_TOUGHNESS,
            "density": DENSITY,
            "penalty_gain": PENALTY_GAIN,
            "zeta_damp": ZETA_DAMP,
            "beam_voxel_h": beam_h,
            "roller_voxel_h": roller_h,
            **{f"beam_{key}": value for key, value in beam_time_step.to_metadata().items()},
            **{f"roller_{key}": value for key, value in roller_time_step.to_metadata().items()},
        },
        headless_steps=STEPS,
        headless_kwargs={
            "steps_per_export": steps_per,
            "show_progress": True,
            "profile_timings": True,
        },
    )
