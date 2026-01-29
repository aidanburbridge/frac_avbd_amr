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
VOXEL_RES = 1000

# Shared solver params
DT_PHYSICS = 1/4000
DT_RENDER = 1/60
STEPS_PER = int(DT_RENDER / DT_PHYSICS)
ITER = 50
GRAV = 0.0
FRICTION = 0.0
PULL_RATE = 0.002
#PULL_RATE = 10
GRIP_DISTANCE = 0.02
#GRIP_DISTANCE = 20
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
    "post_stabilize": False,
    "beta": 10,
    "alpha": 0.95,
    "gamma": 0.99,
    "debug_contacts": False,
}

def calc_cfl(density, young_mod, poisson, vox_size):
    shear_mod = young_mod / (2* ( 1+poisson))
    wave_speed = np.sqrt(shear_mod/density)
    dt_cfl = vox_size / wave_speed
    return dt_cfl

def calc_damping(density, h, stiffness, zeta):
    mass = h*h*density
    return 2 * zeta * np.sqrt(mass*stiffness)

def build_setup()-> SimulationSetup:

    stlvox = vox.STLVoxelizer(STL_PATH, flood_fill=False)
    #occ, raw_origin, raw_h = stlvox.voxelize_to_h(2)
    occ, raw_origin, raw_h = stlvox.voxelize_to_resolution(VOXEL_RES)
    # print(f"DEBUG: Trimesh generated {occ.sum()} raw voxels")


    raw_len = np.max(stlvox.mesh.extents)

    scale_factor = LENGTH / raw_len

    phys_h = raw_h * scale_factor
    phys_origin = raw_origin * scale_factor

    leaves, h_base = oct.octree_from_occ(occ, phys_h)
    #leaves, h_base = oct.octree_from_occ(occ, raw_h)


    print(f"DEBUG: Octree generated {len(leaves)} leaves")

    boxes, mapping = oct.instantiate_boxes_from_tree(
        leaves,
        #raw_origin,
        phys_origin,
        h_base,
        density=DENSITY,
        penalty_gain=PENALTY_GAIN,
        static=False
    )

    print(f"DEBUG: Instantiated {len(boxes)} bodies")

    visco_val = calc_damping(DENSITY, h_base, E_MODULUS, ZETA_DAMP)
    cfl = calc_cfl(DENSITY, E_MODULUS, NU, h_base) #2.9330379512351167e-06

    beam_bonds = oct.build_constraints_from_tree(
        leaves,
        boxes,
        mapping,
        E=E_MODULUS,
        #E = 1e5,
        nu=NU,
        tensile_strength=TENSILE_STRENGTH,
        fracture_toughness=FRACTURE_TOUGHNESS,
        damping_val=visco_val,
    )
    print(f"Number of beam bonds: {len(beam_bonds)}")
    print(f"The damping value used for this sim: {visco_val}")
    print(f"The cfl value used for this sim: {cfl}")

    dog_bone = VoxelAssembly(boxes, beam_bonds)

    dog_bone.align_longest_axis('z')

    dog_bone.set_boundary_fixed(
        faces = ["bottom"],
        distance = GRIP_DISTANCE,
        debug=True
        )
    
    dog_bone.set_boundary_velocity(
        faces = ["top"],
        velocity = [0.0, 0.0, PULL_RATE],
        distance = GRIP_DISTANCE,
        debug=True
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
            "h_base": h_base,
        },
        headless_steps=STEPS,
        headless_kwargs={
            "steps_per_export": STEPS_PER,
            "show_progress": True,
            "profile_timings": True,
        }
    )
