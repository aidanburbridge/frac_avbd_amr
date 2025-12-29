"""
ISO 20753 tensile test with dog bone STL.

"""

import geometry.voxelizer as vox
import geometry.octree as oct

from util.voxel_assembly import VoxelAssembly
from util.pyvista_visualizer import SimulationSetup

STL_PATH = r"C:\Users\aidan\Documents\TUM\Thesis\ISO 20753 Type A1 v1.stl"
LENGTH = 0.170
VOXEL_RES = 500

# Shared solver params
DT_PHYSICS = 1 / 2000
DT_RENDER = 1/60
STEPS_PER = int(DT_RENDER / DT_PHYSICS)
ITER = 50
GRAV = 0.0
FRICTION = 0.0
#PULL_RATE = 0.010
PULL_RATE = 10
#GRIP_DISTANCE = 0.02
GRIP_DISTANCE = 20
PYTHON_SOLVER_PARAMS = {
    "mu": 0.3,
    "post_stabilize": True,
    "beta": 10000,
    "alpha": 0.95,
    "gamma": 0.99,
    "debug_contacts": False,
}

def build_setup()-> SimulationSetup:

    stlvox = vox.STLVoxelizer(STL_PATH, flood_fill=False)
    occ, raw_origin, raw_h = stlvox.voxelize_to_h(2)

    # print(f"DEBUG: Trimesh generated {occ.sum()} raw voxels")


    # raw_len = np.max(stlvox.mesh.extents)

    # scale_factor = LENGTH / raw_len

    # phys_h = raw_h * scale_factor
    # phys_origin = raw_origin * scale_factor

    #leaves, h_base = oct.octree_from_occ(occ, phys_h)
    leaves, h_base = oct.octree_from_occ(occ, raw_h)


    print(f"DEBUG: Octree generated {len(leaves)} leaves")

    boxes, mapping = oct.instantiate_boxes_from_tree( # error here...
        leaves,
        raw_origin,
        #phys_origin,
        h_base,
        density=1150.0,
        penalty_gain=1e6,
        static=False
    )

    print(f"DEBUG: Instantiated {len(boxes)} bodies")

    beam_bonds = oct.build_constraints_from_tree(
        leaves,
        boxes,
        mapping,
        E=2e9,
        nu=0.3,
        tensile_strength=80e6,
        fracture_toughness=5e5,
    )
    print(f"Number of beam bonds: {len(beam_bonds)}")

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
        headless_steps=1000,
        headless_kwargs={
            "steps_per_export": STEPS_PER,
            "show_progress": True,
        }
    )
