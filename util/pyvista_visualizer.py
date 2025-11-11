"""
Reusable PyVista real-time visualizer for AVBD simulations.

Usage (inside a test):

    from util.pyvista_visualizer import run_visualizer

    solver, bodies = make_world()  # returns (Solver, list[Body])
    run_visualizer(solver, bodies,
                   window_size=(1280, 720),
                   background="white",
                   camera_position="iso",
                   show_contacts=True)

You can also pass a custom actor builder mapping to support more body types:

    def build_my_actor(plotter, body):
        actor = plotter.add_mesh(...)
        def update():
            ... # set pose from body
        return actor, update

    run_visualizer(solver, bodies, actor_builders={MyBodyType: build_my_actor})
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np
import pyvista as pv
import vtk

from geometry.primitives import rect_2D, box_3D
from solver.constraints import ContactConstraint


# ---------- tiny helpers that adapt to old/new PyVista APIs ----------

def _try_enable_msaa(plotter: pv.Plotter):
    try:
        plotter.enable_anti_aliasing("msaa")
    except TypeError:
        try:
            plotter.enable_anti_aliasing()
        except Exception:
            pass


def _start_interactive_compat(plotter: pv.Plotter):
    try:
        plotter.show(interactive_update=True, auto_close=False)
        return True
    except TypeError:
        plotter.show(interactive_update=True)
        return True


def _window_is_open(plotter: pv.Plotter):
    if hasattr(plotter, "closed"):
        return not plotter.closed
    if hasattr(plotter, "_closed"):
        return not plotter._closed
    rw = getattr(plotter, "ren_win", None)
    try:
        return bool(rw) and rw is not None
    except Exception:
        return True


# ---------- transforms ----------

def _rot_from_axes(axes_rt: np.ndarray) -> np.ndarray:
    return axes_rt.T


def _vtk_matrix_from_RT(R: np.ndarray, t: np.ndarray) -> vtk.vtkMatrix4x4:
    M = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            M.SetElement(i, j, float(R[i, j]))
        M.SetElement(i, 3, float(t[i]))
    M.SetElement(3, 0, 0.0)
    M.SetElement(3, 1, 0.0)
    M.SetElement(3, 2, 0.0)
    M.SetElement(3, 3, 1.0)
    return M


def _set_actor_pose(actor: vtk.vtkActor, R: np.ndarray, t: np.ndarray):
    T = vtk.vtkTransform()
    T.SetMatrix(_vtk_matrix_from_RT(R, t))
    actor.SetUserTransform(T)


# ---------- default actor builders ----------

def _build_actor_box(plotter: pv.Plotter, body: box_3D, color: Optional[str] = None):
    w, h, d = body.size
    mesh = pv.Cube(center=(0, 0, 0), x_length=w, y_length=h, z_length=d)
    actor = plotter.add_mesh(
        mesh,
        color=(color or "royalblue"),
        smooth_shading=True,
        show_edges=False,
        opacity=1,
    )

    def update():
        R = _rot_from_axes(body.get_axes())
        t = body.position[:3] if body.position.shape[0] >= 3 else getattr(body, "pos", np.zeros(3))
        _set_actor_pose(actor, R, t)

    return actor, update


def _build_actor_rect2d(plotter: pv.Plotter, body: rect_2D, color: Optional[str] = None):
    w, h = body.size
    zt = 0.1
    mesh = pv.Cube(center=(0, 0, 0), x_length=w, y_length=h, z_length=zt)
    actor = plotter.add_mesh(mesh, color=(color or "tomato"), smooth_shading=True, show_edges=False)

    def update():
        theta = float(body.position[2])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
        t = np.array([body.position[0], body.position[1], 0.0], dtype=float)
        _set_actor_pose(actor, R, t)

    return actor, update


def _fallback_actor(plotter: pv.Plotter, color: Optional[str] = None):
    mesh = pv.Cube()
    actor = plotter.add_mesh(mesh, color=(color or "gray"))
    return actor, (lambda: None)


# ---------- contact overlay ----------

def _paint_contacts(
    plotter: pv.Plotter,
    contact_constraints: List[ContactConstraint],
    point_radius: float = 0.03,
    point_color: str = "yellow",
):
    actors = []
    contacts = [
        cc.contact for cc in contact_constraints if hasattr(cc, "contact") and cc.contact is not None
    ]
    for c in contacts:
        p = np.asarray(c.point, dtype=float)
        n = np.asarray(c.normal, dtype=float)
        d = float(c.depth)
        if p.shape != (3,) or n.shape != (3,):
            continue
        n_norm = np.linalg.norm(n) + 1e-12
        n = n / n_norm
        # Point
        sph = pv.Sphere(radius=point_radius, center=p)
        actors.append(plotter.add_mesh(sph, color=point_color, smooth_shading=True))
        # Arrow
        centers = p[None, :]
        dirs = n[None, :]
        actors.append(plotter.add_arrows(centers, dirs, mag=d, color=point_color))
    return actors


def _clear_contact_overlay(plotter: pv.Plotter, overlay: dict):
    if overlay["actors"]:
        for a in overlay["actors"]:
            try:
                plotter.remove_actor(a)
            except Exception:
                pass
        overlay["actors"].clear()


# ---------- public API ----------

def run_visualizer(
    solver,
    bodies: Iterable[object],
    *,
    window_size: Tuple[int, int] = (1280, 720),
    background: str = "white",
    theme: Optional[str] = "document",
    camera_position: Optional[str] = "iso",
    show_axes: bool = True,
    palette: Optional[List[str]] = None,
    initial_paused: bool = True,
    enable_msaa: bool = True,
    show_contacts: bool = False,
    contact_point_radius: float = 0.03,
    contact_point_color: str = "yellow",
    update_interval: int = 2,
    pause_key: str = "space",
    step_key: Optional[str] = "n",
    toggle_contacts_key: str = "c",
    quit_key: str = "q",
    actor_builders: Optional[Dict[Type[object], Callable[..., Tuple[object, Callable[[], None]]]]] = None,
):
    """
    Launch a real-time PyVista visualizer for a given solver+bodies.

    - solver: object with .step() and .contact_constraints (optional)
    - bodies: iterable of body objects; supported: box_3D, rect_2D (fallback cube otherwise)
    - actor_builders: optional mapping {BodyType: (plotter, body, color)->(actor, update)}
    """

    if theme:
        try:
            pv.set_plot_theme(theme)
        except Exception:
            pass

    plotter = pv.Plotter(window_size=window_size)
    if show_axes:
        plotter.add_axes()
    plotter.set_background(background)
    if enable_msaa:
        _try_enable_msaa(plotter)
    try:
        if camera_position:
            plotter.camera_position = camera_position
    except Exception:
        pass

    # Colors
    if palette is None:
        palette = [
            "#4C78A8",
            "#F58518",
            "#54A24B",
            "#E45756",
            "#72B7B2",
            "#EECA3B",
        ]

    # Actor builders registry
    registry: Dict[Type[object], Callable[..., Tuple[object, Callable[[], None]]]] = {
        box_3D: _build_actor_box,
        rect_2D: _build_actor_rect2d,
    }
    if actor_builders:
        registry.update(actor_builders)

    def make_actor(plotter: pv.Plotter, body, color: Optional[str]):
        for t, builder in registry.items():
            if isinstance(body, t):
                return builder(plotter, body, color)
        return _fallback_actor(plotter, color)

    # Actors
    updaters: List[Callable[[], None]] = []
    bodies_list = list(bodies)
    for i, b in enumerate(bodies_list):
        _, update = make_actor(plotter, b, palette[i % len(palette)])
        updaters.append(update)

    # HUD and controls
    plotter.add_text(
        f"PyVista {pv.__version__}  |  {pause_key}=pause  {quit_key}=quit  {toggle_contacts_key}=contacts",
        font_size=10,
    )
    paused = {"flag": bool(initial_paused)}
    contact_overlay = {"show": bool(show_contacts), "actors": []}

    def _toggle_pause():
        paused["flag"] = not paused["flag"]
        if not paused["flag"]:
            contact_overlay["show"] = False
            _clear_contact_overlay(plotter, contact_overlay)

    def _toggle_contacts():
        if paused["flag"]:
            contact_overlay["show"] = not contact_overlay["show"]
        else:
            contact_overlay["show"] = False

    def _step_once():
        if paused["flag"]:
            solver.step()

    plotter.add_key_event(pause_key, _toggle_pause)
    if step_key:
        plotter.add_key_event(step_key, _step_once)
    plotter.add_key_event(toggle_contacts_key, _toggle_contacts)
    plotter.add_key_event(quit_key, lambda: plotter.close())

    # Start window (compat)
    _start_interactive_compat(plotter)

    last = time.perf_counter()
    frames = 0
    iter_count = 0

    while _window_is_open(plotter):
        if not paused["flag"]:
            solver.step()

            # Optional compatibility hooks if user bodies expose these
            for b in bodies_list:
                if hasattr(b, "pos") and getattr(b, "position", None) is not None and b.position.shape[0] >= 3:
                    b.pos = b.position[:3].copy()
                if (
                    hasattr(b, "integrate_orientation")
                    and getattr(b, "velocity", None) is not None
                    and b.velocity.shape[0] >= 6
                    and not getattr(b, "static", False)
                ):
                    b.linear_vel = b.velocity[:3]
                    b.ang_vel = b.velocity[3:6]
                    b.integrate_orientation(getattr(solver, "dt", 0.0))

        iter_count += 1
        if iter_count % max(1, int(update_interval)) == 0:
            # Contacts overlay
            if paused["flag"] and contact_overlay["show"]:
                _clear_contact_overlay(plotter, contact_overlay)
                contacts = getattr(solver, "contact_constraints", []) or []
                if contacts:
                    new_actors = _paint_contacts(
                        plotter,
                        contacts,
                        point_radius=contact_point_radius,
                        point_color=contact_point_color,
                    )
                    contact_overlay["actors"].extend(new_actors)
            else:
                if contact_overlay["actors"]:
                    _clear_contact_overlay(plotter, contact_overlay)

            # Update visuals
            for update in updaters:
                update()

            frames += 1
            now = time.perf_counter()
            if now - last > 0.5:
                fps = frames / (now - last)

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


__all__ = ["run_visualizer"]

