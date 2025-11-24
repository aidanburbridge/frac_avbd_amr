
# Project imports (your current layout)
from geometry.primitives import box_3D
from solver.solver_3 import Solver

from util.pyvista_visualizer import run_visualizer_headless


# SOLVER
solver = Solver(dt=1/60, num_iterations=15, gravity=-9.81)
solver.mu = 0.3
solver.post_stabilize = True
solver.beta = 10000
solver.alpha = 0.95
solver.gamma = 0.99
solver.debug_contacts = False


ground = box_3D(
    trans_pos=(0.0, -1.0, 0.0),
    quat_pos=(1.0, 0.0, 0.0, 0.0),
    linear_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    density=100.0,
    penalty_gain=1e6,
    size=(20.0, .50, 20.0),
    static=True
)
cube1 = box_3D(
    trans_pos=(3.0, 1.0, 0.0),
    quat_pos=(.1, 0.2, 0.3, 0.5),
    linear_vel=(0., 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    density=100.0,
    penalty_gain=1e5,
    size=(0.8, 0.8, 0.8),
    static=False
)
cube2 = box_3D(
    trans_pos=(0.0, 3.0, 0.0),
    quat_pos=(1.0, 0.0, 0.0, 0.0),
    linear_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    density=100.0,
    penalty_gain=1e5,
    size=(1.0, 1.0, 1.0),
    static=False
)
cube3 = box_3D(
    trans_pos=(0.5, 8.0, 0.0),
    quat_pos=(1.0, 0.0, 0.0, 0.0),
    linear_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    density=100.0,
    penalty_gain=1e5,
    size=(1.0, 1.0, 1.0),
    static=False
)
cube4 = box_3D(
    trans_pos=(0., 4, .1),
    quat_pos=(1.0, 0.0, 0.0, 0.0),
    linear_vel=(0, 5, 0),
    ang_vel=(0.0, 0.0, 0.0),
    density=100.0,
    penalty_gain=1e5,
    size=(1.0, 1.0, 1.0),
    static=False
)

for b in (ground, cube1, cube2, cube3, cube4):
    solver.add_body(b)

run_visualizer_headless(solver, solver.bodies, num_steps=500)
