import time
import sys
import numpy as np

# Project imports (your current layout)
from geometry.primitives import rect_2D, box_3D
from solver.solver import Solver

# Third-party
import pyvista as pv
import vtk


# ---------- tiny helpers that adapt to old/new PyVista APIs ----------

def try_enable_msaa(plotter: pv.Plotter):
    """Enable MSAA if this PyVista build supports the signature."""
    try:
        plotter.enable_anti_aliasing("msaa")
    except TypeError:
        try:
            plotter.enable_anti_aliasing()
        except Exception:
            pass

def add_floor_compat(plotter: pv.Plotter):
    """
    Older PyVista can't use add_floor(origin=...). Build a plane instead.
    Newer builds may have Plotter.add_floor without 'origin'.
    """
    # Try native add_floor with minimal args
    try:
        # Many versions accept just sizes/resolution, origin is optional/absent.
        return plotter.add_floor(i_size=40, j_size=40, i_resolution=20, j_resolution=20)
    except TypeError:
        # Fallback: a light gray plane on the XZ plane (normal +Y)
        floor = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0),
                         i_size=40, j_size=40, i_resolution=20, j_resolution=20)
        return plotter.add_mesh(floor, color="#eaeaea", show_edges=False)

def start_interactive_compat(plotter: pv.Plotter):
    """
    Old PyVista: no plotter.is_active; drive the window with interactive_update=True.
    """
    # Newer PyVista can just return; older needs interactive_update
    try:
        plotter.show(interactive_update=True, auto_close=False)
        return True
    except TypeError:
        # Some versions use different kwargs; try safest call
        plotter.show(interactive_update=True)
        return True

def window_is_open(plotter: pv.Plotter):
    """
    Cross-version check for an open window.
    """
    # Newer versions sometimes have .closed
    if hasattr(plotter, "closed"):
        return not plotter.closed
    # Older versions keep an internal flag
    if hasattr(plotter, "_closed"):
        return not plotter._closed
    # Last resort: assume open until VTK says otherwise
    rw = getattr(plotter, "ren_win", None)
    try:
        return bool(rw) and rw is not None
    except Exception:
        return True


# ---------- transforms ----------

def rot_from_axes(axes_rt: np.ndarray) -> np.ndarray:
    """Your get_axes() returns R^T (rows = world axes). Convert back to R."""
    return axes_rt.T

def vtk_matrix_from_RT(R: np.ndarray, t: np.ndarray) -> vtk.vtkMatrix4x4:
    M = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            M.SetElement(i, j, float(R[i, j]))
        M.SetElement(i, 3, float(t[i]))
    M.SetElement(3, 0, 0.0); M.SetElement(3, 1, 0.0); M.SetElement(3, 2, 0.0); M.SetElement(3, 3, 1.0)
    return M

def set_actor_pose(actor: vtk.vtkActor, R: np.ndarray, t: np.ndarray):
    T = vtk.vtkTransform()
    T.SetMatrix(vtk_matrix_from_RT(R, t))
    actor.SetUserTransform(T)


# ---------- scene construction ----------

def make_world_3d():
    """
    Build a small 3D scene: ground + a couple of cubes.
    """
    solver = Solver(dt=1/120, num_iterations=40, gravity=-9.81)
    solver.mu = 0.6
    solver.post_stabilize = True

    ground = box_3D(
        trans_pos=(0.0, -1.0, 0.0),
        quat_pos=(1.0, 0.0, 0.0, 0.0),
        linear_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        density=1000.0,
        penalty_gain=1e6,
        size=(20.0, 2.0, 20.0),
        static=True
    )

    cube1 = box_3D(
        trans_pos=(0.0, 2.0, 0.0),
        quat_pos=(1.0, 0.0, 0.0, 0.0),
        linear_vel=(0.0, 10.0, 0.0),
        ang_vel=(2.0, 0.0, 0.3),
        density=500.0,
        penalty_gain=1e5,
        size=(0.8, 0.8, 0.8),
        static=False
    )
    cube2 = box_3D(
        trans_pos=(0.0, 10.0, 0.0),
        quat_pos=(1.0, 0.0, 0.0, 0.0),
        linear_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.3, 0.0),
        density=500.0,
        penalty_gain=1e5,
        size=(1.0, 1.0, 1.0),
        static=False
    )

    for b in (ground, cube1, cube2):
        solver.add_body(b)

    return solver, [ground, cube1, cube2]

def build_actor_for_body(plotter: pv.Plotter, body, color=None):
    """
    Create a mesh/actor for a Body and return an updater closure that applies pose.
    """
    if isinstance(body, box_3D):
        w, h, d = body.size
        mesh = pv.Cube(center=(0, 0, 0), x_length=w, y_length=h, z_length=d)
        actor = plotter.add_mesh(mesh, color=(color or "royalblue"),
                                 smooth_shading=True, show_edges=False)
        def update():
            R = rot_from_axes(body.get_axes())
            t = body.position[:3] if body.position.shape[0] >= 3 else body.pos
            set_actor_pose(actor, R, t)
        return actor, update

    if isinstance(body, rect_2D):
        # Show 2D rect as a thin 3D prism
        w, h = body.size
        zt = 0.1
        mesh = pv.Cube(center=(0, 0, 0), x_length=w, y_length=h, z_length=zt)
        actor = plotter.add_mesh(mesh, color=(color or "tomato"),
                                 smooth_shading=True, show_edges=False)
        def update():
            theta = float(body.position[2])
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]], dtype=float)
            t = np.array([body.position[0], body.position[1], 0.0], dtype=float)
            set_actor_pose(actor, R, t)
        return actor, update

    # Fallback: unit cube
    mesh = pv.Cube()
    actor = plotter.add_mesh(mesh, color=(color or "gray"))
    return actor, (lambda: None)


# ---------- main loop ----------

def run_realtime():
    solver, bodies = make_world_3d()

    pv.set_plot_theme("document")
    plotter = pv.Plotter(window_size=(1280, 720))
    plotter.add_axes()
    plotter.set_background("white")
    try_enable_msaa(plotter)

    # Camera
    try:
        plotter.camera_position = "iso"
    except Exception:
        pass

    # Floor / grid
    #add_floor_compat(plotter)

    # Actors
    updaters = []
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B"]
    for i, b in enumerate(bodies):
        _, update = build_actor_for_body(plotter, b, color=palette[i % len(palette)])
        updaters.append(update)

    # HUD and controls
    plotter.add_text(f"PyVista {pv.__version__}  |  space=pause  q=quit", font_size=10)
    paused = {"flag": True}
    plotter.add_key_event("space", lambda: paused.update(flag=not paused["flag"]))
    plotter.add_key_event("q", lambda: plotter.close())

    # Start window (compat)
    start_interactive_compat(plotter)

    last = time.perf_counter()
    frames = 0

    while window_is_open(plotter):
        if not paused["flag"]:
            solver.step()

            # If your 6-DOF bodies rely on velocity[3:6] for rotation, integrate here.
            for b in bodies:
                if hasattr(b, "pos") and b.position.shape[0] >= 3:
                    b.pos = b.position[:3].copy()
                if hasattr(b, "integrate_orientation") and b.velocity.shape[0] >= 6 and not b.static:
                    b.linear_vel = b.velocity[:3]
                    b.ang_vel    = b.velocity[3:6]
                    b.integrate_orientation(solver.dt)

        # Update visuals
        for update in updaters:
            update()

        frames += 1
        now = time.perf_counter()
        if now - last > 0.5:
            fps = frames / (now - last)
            # plotter.add_text(f"PyVista {pv.__version__}  |  FPS: {fps:5.1f}   (space=pause, q=quit)",
            #                  position="upper_right", font_size=10)
            frames = 0
            last = now

        # Drive rendering
        try:
            plotter.update()
        except Exception:
            pass
        plotter.render()

    try:
        plotter.close()
    except Exception:
        pass


if __name__ == "__main__":
    print(f"[viz] Using PyVista {pv.__version__} on Python {sys.version.split()[0]}")
    run_realtime()
