from __future__ import annotations

from geometry.primitives import box_3D
from util.pyvista_visualizer import SimulationSetup


DT = 1 / 60
ITER = 15
GRAVITY = -9.81
FRICTION = 0.5
PYTHON_SOLVER_PARAMS = {
    "mu": 0.3,
    "post_stabilize": True,
    "beta": 10000,
    "alpha": 0.95,
    "gamma": 0.99,
    "debug_contacts": False,
}


def build_setup(sync_bodies: bool = True) -> SimulationSetup:
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

    return SimulationSetup(
        bodies=[ground, cube1, cube2, cube3, cube4],
        constraints=[],
        dt=DT,
        iterations=ITER,
        gravity=GRAVITY,
        friction=FRICTION,
        sync_bodies=sync_bodies,
        python_solver_params=PYTHON_SOLVER_PARAMS,
        headless_steps=500,
    )


if __name__ == "__main__":
    from util.pyvista_visualizer import run_simulation

    run_simulation(build_setup())
