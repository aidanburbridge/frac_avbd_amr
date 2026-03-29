"""
ISO 20753 tensile test with dog bone STL.

"""

import geometry.voxelizer as vox
import geometry.octree as oct
import numpy as np

from util.voxel_assembly import VoxelAssembly
from util.engine import SimulationSetup

STL_PATH = r"C:\Users\aidan\Documents\TUM\Thesis\D3Q26 Test.stl"
LENGTH = 0.170
VOXEL_RES = 5

# Shared solver params
DT_PHYSICS = 1 / 100
DT_RENDER = 1/60
STEPS_PER = int(DT_RENDER / DT_PHYSICS)
ITER = 15
GRAV = 0.0
FRICTION = 0.0
PULL_RATE = 0.010
GRIP_DISTANCE = 0.2
PYTHON_SOLVER_PARAMS = {
    "mu": 0.3,
    "post_stabilize": True,
    "beta": 10000,
    "alpha": 0.95,
    "gamma": 0.99,
    "debug_contacts": False,
}

def build_setup()-> SimulationSetup:

    stlvox = vox.STLVoxelizer(STL_PATH)
    occ, raw_origin, raw_h = stlvox.voxelize_to_resolution(VOXEL_RES)

    raw_len = np.max(stlvox.mesh.extents)
    scale_factor = LENGTH / raw_len

    phys_h = raw_h * scale_factor
    phys_origin = raw_origin * scale_factor

    leaves, h_base = oct.octree_from_occ(occ, phys_h)
    boxes, mapping, _ = oct.instantiate_boxes_from_tree(
        leaves,
        phys_origin,
        h_base,
        density=1150.0,
        penalty_gain=1e6,
        static=False
    )

    beam_bonds = oct.build_constraints_from_tree(
        leaves,
        boxes,
        mapping,
        E=2e9,
        nu=0.3,
        tensile_strength=80e6,
        fracture_toughness=1000,
    )
    print(f"Number of beam bonds: {len(beam_bonds)}")

    dog_bone = VoxelAssembly(boxes, beam_bonds)

    dog_bone.align_longest_axis('z')
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
            "benchmark_name": "Bond-26 tensile test",
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
            "dt_physics": DT_PHYSICS,
            "dt_render": DT_RENDER,
            "steps_per_export": STEPS_PER,
        },
        headless_steps=20,
        headless_kwargs={
            "steps_per_export": STEPS_PER,
            "show_progress": True,
        }
    )
