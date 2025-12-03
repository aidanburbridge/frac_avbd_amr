import argparse
import sys


def _parse_args():
    parser = argparse.ArgumentParser(description="Double beam fall benchmark.")
    parser.add_argument(
        "--solver",
        choices=["hybrid", "python"],
        default="hybrid",
        help="Choose Julia-backed hybrid solver or pure Python solver_4.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="For hybrid: do not copy poses back to Python each frame (for timing Julia-only).",
    )
    return parser.parse_known_args()


_args, _unknown = _parse_args() if __name__ == "__main__" else (None, None)
USE_HYBRID = (_args.solver == "hybrid") if _args else True
SYNC_BODIES = not (_args.no_sync if _args else False)

if USE_HYBRID:
    from solver.hybrid_solver import HybridWorld
else:
    from solver.solver_4 import Solver
from util.pyvista_visualizer import run_visualizer, run_visualizer_headless
from util.voxel_assembly import VoxelAssembly
import geometry.voxelizer as vox
import geometry.octree as oct


# make a beam from a few boxes
# manually add bonds between voxels
# simulate beam bending under gravity

# Shared solver params
DT = 1 / 240
ITER = 60
GRAV = -9.81

#STL
STL_PATH = r"C:\Users\aidan\Documents\TUM\Thesis\10x10x50_beam.stl"

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
    E=10e5,
    nu=0.25,
    tensile_strength=5e8,
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
    .set_bond_material(E=10e3)
)

# Add both assemblies and their bonds to the solver
if USE_HYBRID:
    all_bodies = cantilever.bodies + falling_beam.bodies
    all_bonds = cantilever.constraints + falling_beam.constraints
    solver = HybridWorld(
        all_bodies,
        all_bonds,
        dt=DT,
        iterations=ITER,
        gravity=GRAV,
        sync_bodies=SYNC_BODIES,  # set False to time pure Julia without pose round-trip
        friction=0.3,
    )
else:
    solver = Solver(dt=DT, num_iterations=ITER, gravity=GRAV)
    solver.mu = 0.3
    solver.post_stabilize = True
    solver.beta = 10000
    solver.alpha = 0.95
    solver.gamma = 0.99
    solver.debug_contacts = False

    for beam in (cantilever, falling_beam):
        for body in beam.bodies:
            solver.add_body(body)
        solver.add_persistent_constraints(beam.constraints)

num_stat = sum(1 for body in cantilever.bodies if getattr(body, "static", False))
print("Total static: ", num_stat)

run_visualizer_headless(solver, cantilever.bodies + falling_beam.bodies, num_steps=1000)#, save_path="beam_test_05")

if __name__ == "__main__" and _unknown:
    # If extra args were provided, show a warning so users know they were ignored.
    print(f"[double_beam_fall] Ignored extra args: {' '.join(_unknown)}")
