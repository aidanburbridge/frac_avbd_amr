import py_solver.constraints as const
from geometry.primitives import box_3D

cube1 = box_3D(
    trans_pos=(0.0, 0.0, 0.0),
    quat_pos=(1, 0., 0., 0.),
    linear_vel=(0., 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    density=100.0,
    penalty_gain=1e5,
    size=(1, 1, 1),
    static=False
)

cube2 = box_3D(
    trans_pos=(0.0, -1.0, 0.0),
    quat_pos=(1, 0., 0., 0.),
    linear_vel=(0., 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    density=100.0,
    penalty_gain=1e5,
    size=(1, 1, 1),
    static=False
)

print("Test output: ", const.build_face_bonds(cube1, cube2, 1000, 0.5))