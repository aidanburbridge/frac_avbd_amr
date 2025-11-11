from geometry.primitives import box_3D
from solver.solver_3 import Solver
from util.pyvista_visualizer import run_visualizer
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
occ, origin, h_cube = stlvox.voxelize_to_resolution(10)
print("Origin: ",origin)

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
beam_bonds = oct.build_constraints_from_tree(leaves, boxes, mapping, E=10e7, nu=0.25)
solver.add_persistent_constraints(beam_bonds)

fall_cube = box_3D(
    (0, 50, 40),
    (0, 0, 0,0),
    (0,0,0),
    (0,0,0),
    100,
    10e5,
    (10,10,10),
    False
)

# Add bodies to solver and set fixed-end static
for b in boxes:
    solver.add_body(b)

solver.add_body(fall_cube)

# Set fixed end: mark voxels at global min Z as static
num_stat = 0
z_min = min(b.position[2] for b in boxes)
eps = 1e-9
for b in boxes:
    if abs(b.position[2] - z_min) <= eps:
        b.set_static()
        num_stat += 1


print("Total static: ", num_stat)

run_visualizer(solver, solver.bodies)


# BEAM DIMENSIONS
# length = 10
# width = 1
# height = 1
# cube_side_length = 1

# for i in len(length):
#     for j in len(width):
#         for k in len(height):
             
#             new_cube= box_3D(
#                 trans_pos=(i * cube_side_length, j * cube_side_length, k * cube_side_length),
#                 quat_pos=(1.0, 0.0, 0.0, 0.0),
#                 linear_vel=(0., 0.0, 0.0),
#                 ang_vel=(0.0, 0.0, 0.0),
#                 density=100.0,
#                 penalty_gain=1e5,
#                 size=(cube_side_length, cube_side_length, cube_side_length),
#                 static=False
#             )

#             cube_arr.append(new_cube)
#             solver.add_body(new_cube)
            
