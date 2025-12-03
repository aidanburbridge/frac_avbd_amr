from __future__ import annotations

import argparse

# Project imports (your current layout)
from geometry.primitives import box_3D
from solver.hybrid_solver import HybridWorld
from solver.solver_4 import Solver

from util.pyvista_visualizer import run_visualizer_headless


def _parse_args():
    parser = argparse.ArgumentParser(description="Simple headless drop test.")
    parser.add_argument(
        "--solver",
        choices=["hybrid", "python"],
        default="hybrid",
        help="Choose Julia hybrid or pure Python solver_4.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="For hybrid: skip copying poses back each frame (Julia-only timing).",
    )
    return parser.parse_args()


ARGS = _parse_args()


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

if ARGS.solver == "hybrid":
    bodies = [ground, cube1, cube2, cube3, cube4]
    solver = HybridWorld(
        bodies,
        constraints=[],
        dt=1 / 60,
        iterations=15,
        gravity=-9.81,
        sync_bodies=not ARGS.no_sync,
    )
    body_list = bodies
else:
    solver = Solver(dt=1 / 60, num_iterations=15, gravity=-9.81)
    solver.mu = 0.3
    solver.post_stabilize = True
    solver.beta = 10000
    solver.alpha = 0.95
    solver.gamma = 0.99
    solver.debug_contacts = False
    for b in (ground, cube1, cube2, cube3, cube4):
        solver.add_body(b)
    body_list = solver.bodies

run_visualizer_headless(solver, body_list, num_steps=500)
