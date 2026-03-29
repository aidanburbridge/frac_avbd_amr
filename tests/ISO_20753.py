"""
ISO 20753 tensile test with dog bone STL.

"""

import geometry.voxelizer as vox
import geometry.octree as oct
import numpy as np

from util.timestep import calc_damping
from util.voxel_assembly import VoxelAssembly
from util.simulate import SimulationSetup

STL_PATH = r"C:\Users\aidan\Documents\TUM\Thesis\ISO 20753 Type A1 v1.stl"
LENGTH = 0.170
VOXEL_RES = 100

# Build full hierarchy potential and lists for refinement
MAX_REF_LEVEL = 2

# Shared solver params
DT_PHYSICS = 1 / 4000
DT_RENDER = 1 / 60
STEPS_PER = max(1, int(DT_RENDER / DT_PHYSICS))
ITER = 80
GRAV = 0.0
FRICTION = 0.0
PULL_RATE = 0.005
# PULL_RATE = 10
GRIP_DISTANCE = 0.01
# GRIP_DISTANCE = 20
E_MODULUS = 2e9
NU = 0.3
TENSILE_STRENGTH = 80e5
FRACTURE_TOUGHNESS = 5e4
DENSITY = 1150.0
PENALTY_GAIN = 1e6
STEPS = 3500
ZETA_DAMP = 0.1
REFINE_STRESS_THRESHOLD = 0.05 * TENSILE_STRENGTH

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


def build_setup() -> SimulationSetup:

    stlvox = vox.STLVoxelizer(STL_PATH, flood_fill=False)

    occ, raw_origin, raw_h = stlvox.voxelize_to_resolution(VOXEL_RES)

    raw_len = np.max(stlvox.mesh.extents)
    scale_factor = LENGTH / raw_len
    phys_h = raw_h * scale_factor
    phys_origin = raw_origin * scale_factor

    visco_val = calc_damping(DENSITY, phys_h, E_MODULUS, ZETA_DAMP)

    def _contains_fn(pts: np.ndarray) -> np.ndarray:
        return vox._contains_points_chunked(
            stlvox.mesh,
            np.asarray(pts, dtype=float),
            chunk=200_000,
            show_progress=False,
        )

    boxes, beam_bonds, amr_dict = oct.build_hierarchical_bodies_bonds_amr(
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
        damping_val=visco_val,
    )

    print(f"DEBUG: Instantiated {len(boxes)} bodies")
    print(f"Number of beam bonds: {len(beam_bonds)}")
    print(f"The damping value used for this sim: {visco_val}")

    dog_bone = VoxelAssembly(boxes, beam_bonds)

    dog_bone.align_longest_axis("z")
    for idx, body in enumerate(dog_bone.bodies):
        body.body_id = idx

    fixed_targets = dog_bone.select_boundary(
        ["bottom"],
        distance=GRIP_DISTANCE,
    )
    load_targets = dog_bone.select_boundary(
        ["top"],
        distance=GRIP_DISTANCE,
    )

    dog_bone.set_boundary_fixed(
        faces=["bottom"],
        distance=GRIP_DISTANCE,
        debug=True,
    )

    dog_bone.set_boundary_velocity(
        faces=["top"],
        velocity=[0.0, 0.0, PULL_RATE],
        # velocity = [0.0, PULL_RATE, 0.0],
        distance=GRIP_DISTANCE,
        debug=True,
    )

    return SimulationSetup(
        bodies=dog_bone.bodies,
        constraints=beam_bonds,
        dt=DT_PHYSICS,
        iterations=ITER,
        gravity=GRAV,
        friction=FRICTION,
        sync_bodies=True,
        python_solver_params=PYTHON_SOLVER_PARAMS,
        amr_params=amr_dict,
        metadata={
            "benchmark_name": "ISO 20753 test",
            "dt_physics": DT_PHYSICS,
            "dt_render": DT_RENDER,
            "E": E_MODULUS,
            "nu": NU,
            "tensile_strength": TENSILE_STRENGTH,
            "fracture_toughness": FRACTURE_TOUGHNESS,
            "density": DENSITY,
            "penalty_gain": PENALTY_GAIN,
            "zeta_damp": ZETA_DAMP,
            "geometry_scaled_to_physical_units": True,
            "length_unit_label": "m",
            "displacement_unit_label": "m",
            "area_unit_label": "m^2",
            "stress_unit_label": "Pa",
            "energy_unit_label": "J",
            "raw_length_scale_to_m": scale_factor,
            "h_base": phys_h,
            "raw_h_base": raw_h,
            "loading_velocity": [0.0, 0.0, PULL_RATE],
            "fixed_body_ids": [int(body.body_id) for body in fixed_targets],
            "load_body_ids": [int(body.body_id) for body in load_targets],
            "refine_stress_threshold": REFINE_STRESS_THRESHOLD,
            "steps": STEPS,
            "steps_per_export": STEPS_PER,
        },
        headless_steps=STEPS,
        headless_kwargs={
            "steps_per_export": STEPS_PER,
            "show_progress": True,
            "profile_timings": True,
        },
    )
