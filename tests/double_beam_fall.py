from solver.solver_4 import Solver
from util.pyvista_visualizer import run_visualizer, run_visualizer_headless
from util.voxel_assembly import VoxelAssembly
import geometry.voxelizer as vox
import geometry.octree as oct


# make a beam from a few boxes
# manually add bonds between voxels
# simulate beam bending under gravity

# SOLVER
solver = Solver(dt=1/60, num_iterations=15, gravity=-9.81)
solver.mu = 0.3
solver.post_stabilize = True
solver.beta = 10000
solver.alpha = 0.95
solver.gamma = 0.99
solver.debug_contacts = False

#STL
STL_PATH = r"C:\Users\aidan\Documents\TUM\Thesis\10x10x50_beam.stl"

# Voxelization
stlvox = vox.STLVoxelizer(STL_PATH)
occ, origin, h_cube = stlvox.voxelize_to_resolution(40)
print("Origin: ", origin)

# Octree & Cubes
leaves, h_base = oct.octree_from_occ(occ, h_cube)
boxes, mapping = oct.instantiate_boxes_from_tree(
    leaves,
    origin=origin,
    h_base=h_cube,
    density=1.0,
    penalty_gain=1e5,
    static=False,
)
beam_bonds = oct.build_constraints_from_tree(
    leaves,
    boxes,
    mapping,
    E=10e5,
    nu=0.25,
    tensile_strength=5e4,
    fracture_toughness=1e4,
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
for beam in (cantilever, falling_beam):
    for body in beam.bodies:
        solver.add_body(body)
    solver.add_persistent_constraints(beam.constraints)

num_stat = sum(1 for body in cantilever.bodies if body.static)
print("Total static: ", num_stat)

run_visualizer_headless(solver, solver.bodies, num_steps=500)#, save_path="beam_test_05")
