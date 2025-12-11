from util.pyvista_visualizer import SimulationSetup
from util.voxel_assembly import VoxelAssembly
import geometry.voxelizer as vox
import geometry.octree as oct


# make a beam from a few boxes
# manually add bonds between voxels
# simulate beam bending under gravity

# Shared solver params
DT_PHYSICS = 1 / 400
DT_RENDER = 1/60
STEPS_PER_EXPORT = int(DT_RENDER / DT_PHYSICS)
ITER = 60
GRAV = -9.81
FRICTION = 0.3
PYTHON_SOLVER_PARAMS = {
    "mu": 0.3,
    "post_stabilize": True,
    "beta": 10000,
    "alpha": 0.95,
    "gamma": 0.99,
    "debug_contacts": False,
}

STL_PATH = r"C:\Users\aidan\Documents\TUM\Thesis\10x10x50_beam.stl"

def build_setup(sync_bodies: bool = True) -> SimulationSetup:
    # Voxelization
    stlvox = vox.STLVoxelizer(STL_PATH)
    occ, origin, h_cube = stlvox.voxelize_to_resolution(200)
    print("Origin: ", origin)

    # Octree & Cubes
    leaves, h_base = oct.octree_from_occ(occ, h_cube)
    boxes, mapping = oct.instantiate_boxes_from_tree(
        leaves,
        origin=origin,
        h_base=h_cube,
        density=.10,
        penalty_gain=1e5,
        static=False,
    )
    beam_bonds = oct.build_constraints_from_tree(
        leaves,
        boxes,
        mapping,
        E=10e9,
        nu=0.25,
        tensile_strength=1e6,
        fracture_toughness=1e6,
    )

    # Build a base assembly that we can duplicate and transform
    base_beam = VoxelAssembly(boxes, beam_bonds)

    # Cantilever reference: fix the bottom face so it acts as an anchored beam
    cantilever = base_beam.copy()
    cantilever.fix_faces(["bottom"])

    # Free-falling beam: duplicate, rotate 90 degrees, raise it, and keep it dynamic
    falling_beam = (
        base_beam.copy()
        .rotate(axis=(0.0, 1.0, 0.0), angle=90.0, degrees=True)
        .rotate(axis=(0.0, 0.0, 1.0), angle=30.0, degrees=True)
        .translate((0.0, 20.0, 20.0))
        .set_all_static(False)
        .set_bond_material(E=10e7, nu= 0.3)
    )

    all_bodies = cantilever.bodies + falling_beam.bodies
    all_bonds = cantilever.constraints + falling_beam.constraints

    num_stat = sum(1 for body in cantilever.bodies if getattr(body, "static", False))
    print("Total static: ", num_stat)

    return SimulationSetup(
        bodies=all_bodies,
        constraints=all_bonds,
        dt=DT_PHYSICS,
        iterations=ITER,
        gravity=GRAV,
        friction=FRICTION,
        sync_bodies=sync_bodies,
        python_solver_params=PYTHON_SOLVER_PARAMS,

        headless_steps=2000,
        headless_kwargs={
            "steps_per_export": 6,
            "show_progress": True,
        }
    )


if __name__ == "__main__":
    from util.pyvista_visualizer import run_simulation

    run_simulation(build_setup())
