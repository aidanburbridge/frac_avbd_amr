import geometry.voxelizer as vox
import geometry.octree as oct
import numpy as np

from geometry.bond_data import BondData
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
DT_PHYSICS = 1 / 4000
DT_RENDER = 1 / 60
STEPS_PER = max(1, int(DT_RENDER / DT_PHYSICS))
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
STEPS = 2000
ZETA_DAMP = 0.1

PYTHON_SOLVER_PARAMS = {
    "mu": 0.3,
    "post_stabilize": True,
    "beta": 5.0e4,
    "alpha": 0.95,
    "gamma": 1.0,
    "debug_contacts": False,
}


def calc_cfl(density, young_mod, poisson, vox_size):
    shear_mod = young_mod / (2 * (1 + poisson))
    wave_speed = np.sqrt(shear_mod / density)
    return vox_size / wave_speed


def calc_damping(density, h, stiffness, zeta):
    mass = h * h * density
    return 2 * zeta * np.sqrt(mass * stiffness)


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
    assembly.align_longest_axis(align_axis)

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
    beam, beam_h = _build_voxel_assembly(
        BEAM_STL,
        target_length=BEAM_LENGTH,
        voxel_res=BEAM_VOXEL_RES,
        align_axis="x",
    )
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

    # Non-AMR scene: keep every body active and disable refinement explicitly.
    amr_dict = {
        "parent_list": np.full(n_bodies, -1, dtype=np.int32),
        "child_start": np.full(n_bodies, -1, dtype=np.int32),
        "child_count": np.zeros(n_bodies, dtype=np.int32),
        "level": np.zeros(n_bodies, dtype=np.int8),
        "active": np.ones(n_bodies, dtype=np.bool_),
        "valid_mask": np.ones(n_bodies, dtype=np.bool_),
        "neighbor_map": np.full((n_bodies, 6), -1, dtype=np.int32),
        "can_refine": np.zeros(n_bodies, dtype=np.bool_),
        "max_ref_level": 0,
        "max_ref_level_per_body": np.zeros(n_bodies, dtype=np.int32),
    }

    cfl = min(calc_cfl(DENSITY, E_MODULUS, NU, beam_h), calc_cfl(DENSITY, E_MODULUS, NU, roller_h))
    print(f"Fixture bodies: beam={len(beam.bodies)} left={len(left_support.bodies)} right={len(right_support.bodies)} top={len(top_indenter.bodies)}")
    print(f"Fixture bonds: total={len(all_bonds)}")
    print(f"Estimated CFL: {cfl}")

    return SimulationSetup(
        bodies=all_bodies,
        constraints=all_bonds,
        dt=DT_PHYSICS,
        iterations=ITER,
        gravity=GRAV,
        friction=FRICTION,
        sync_bodies=True,
        python_solver_params=PYTHON_SOLVER_PARAMS,
        amr_params=amr_dict,
        metadata={
            "dt_physics": DT_PHYSICS,
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
        },
        headless_steps=STEPS,
        headless_kwargs={
            "steps_per_export": STEPS_PER,
            "show_progress": True,
            "profile_timings": True,
        },
    )
