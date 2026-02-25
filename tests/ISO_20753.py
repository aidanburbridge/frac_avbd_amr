"""
ISO 20753 tensile test with dog bone STL.

"""

import geometry.voxelizer as vox
import geometry.octree as oct
import numpy as np

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
STEPS_PER = int(DT_RENDER / DT_PHYSICS)
ITER = 50
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
STEPS = 1000
ZETA_DAMP = 0.1

PYTHON_SOLVER_PARAMS = {
    "mu": 0.3,
    "post_stabilize": False,
    "beta": 10,
    "alpha": 0.95,
    "gamma": 0.99,
    "debug_contacts": False,
}

def calc_cfl(density, young_mod, poisson, vox_size):
    shear_mod = young_mod / (2 * (1 + poisson))
    wave_speed = np.sqrt(shear_mod / density)
    dt_cfl = vox_size / wave_speed
    return dt_cfl


def calc_damping(density, h, stiffness, zeta):
    mass = h * h * density
    return 2 * zeta * np.sqrt(mass * stiffness)


def build_setup() -> SimulationSetup:

    stlvox = vox.STLVoxelizer(STL_PATH, flood_fill=False)

    occ, raw_origin, raw_h = stlvox.voxelize_to_resolution(VOXEL_RES)

    raw_len = np.max(stlvox.mesh.extents)
    scale_factor = LENGTH / raw_len
    phys_h = raw_h * scale_factor
    phys_origin = raw_origin * scale_factor

    visco_val = calc_damping(DENSITY, phys_h, E_MODULUS, ZETA_DAMP)
    cfl = calc_cfl(DENSITY, E_MODULUS, NU, phys_h)  # 2.9330379512351167e-06

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
    print(f"The cfl value used for this sim: {cfl}")

    dog_bone = VoxelAssembly(boxes, beam_bonds)

    dog_bone.align_longest_axis("z")

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
            "dt_physics": DT_PHYSICS,
            "dt_render": DT_RENDER,
            "E": E_MODULUS,
            "nu": NU,
            "tensile_strength": TENSILE_STRENGTH,
            "fracture_toughness": FRACTURE_TOUGHNESS,
            "density": DENSITY,
            "penalty_gain": PENALTY_GAIN,
            "zeta_damp": ZETA_DAMP,
            "h_base": phys_h,
        },
        headless_steps=STEPS,
        headless_kwargs={
            "steps_per_export": STEPS_PER,
            "show_progress": True,
            "profile_timings": True,
        },
    )
