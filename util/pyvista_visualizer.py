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

import copy
import pickle
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
import sys
from types import SimpleNamespace

import numpy as np
import pyvista as pv
import vtk

from geometry.primitives import rect_2D, box_3D
from solver.constraints import ContactConstraint

from tqdm import tqdm


_DEFAULT_RECORDING_DIR = Path("output/frames_list")
_SAVE_INDEX_WIDTH = 2


def _derive_default_stem(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 and argv0 not in {"-m", ""}:
        stem = Path(argv0).stem
        if stem:
            return stem
    return "recording"


def _base_save_path(target: Optional[Union[str, Path]], default_stem: Optional[str]) -> Path:
    if target is None:
        stem = _derive_default_stem(default_stem)
        return (_DEFAULT_RECORDING_DIR / stem).with_suffix(".pkl")

    path = Path(target).expanduser()
    if not path.suffix:
        path = path.with_suffix(".pkl")
    if path.parent == Path("."):
        path = _DEFAULT_RECORDING_DIR / path.name
    return path


def _split_numbered_stem(stem: str) -> tuple[str, Optional[int], int]:
    idx = stem.rfind("_")
    if idx != -1:
        suffix = stem[idx + 1 :]
        if suffix.isdigit():
            return stem[:idx], int(suffix), len(suffix)
    return stem, None, _SAVE_INDEX_WIDTH


def _unique_save_path(base: Path, *, always_number: bool = False) -> tuple[Path, Optional[Path]]:
    base = base.expanduser()
    if not base.suffix:
        base = base.with_suffix(".pkl")
    parent = base.parent if base.parent not in (Path(""), Path(".")) else _DEFAULT_RECORDING_DIR
    parent.mkdir(parents=True, exist_ok=True)

    root, number, width = _split_numbered_stem(base.stem)

    if always_number:
        number = (number or 0) + 1
        candidate = parent / f"{root}_{number:0{width}d}{base.suffix}"
        conflict: Optional[Path] = None
        while candidate.exists():
            conflict = candidate
            number += 1
            candidate = parent / f"{root}_{number:0{width}d}{base.suffix}"
        return candidate, conflict

    candidate = parent / base.name
    conflict = None
    while candidate.exists():
        conflict = candidate
        if number is None:
            number = 1
            width = max(width, _SAVE_INDEX_WIDTH)
        else:
            number += 1
        candidate = parent / f"{root}_{number:0{width}d}{base.suffix}"
    return candidate, conflict


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
        opacity=0.7,
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


# ---------- contact serialization helpers ----------

def _vector3(data) -> Optional[np.ndarray]:
    if data is None:
        return None
    try:
        arr = np.asarray(data, dtype=float)
    except Exception:
        return None
    if arr.size == 0:
        return None
    arr = np.ravel(arr)
    vec = np.zeros(3, dtype=float)
    count = min(3, arr.size)
    vec[:count] = arr[:count]
    return vec


def _snapshot_contact_constraints(contact_constraints: Optional[Iterable[ContactConstraint]]) -> List[Dict[str, object]]:
    snapshot: List[Dict[str, object]] = []
    if not contact_constraints:
        return snapshot
    for cc in contact_constraints:
        contact = getattr(cc, "contact", None)
        if contact is None:
            continue
        point = _vector3(getattr(contact, "point", None))
        normal = _vector3(getattr(contact, "normal", None))
        if point is None or normal is None:
            continue
        depth = float(getattr(contact, "depth", 0.0))
        snapshot.append(
            {
                "point": point,
                "normal": normal,
                "depth": depth,
            }
        )
    return snapshot


def _snapshot_contact_frames(contact_frames: Optional[List[List[Dict[str, object]]]]) -> List[List[Dict[str, object]]]:
    if not contact_frames:
        return []
    return [
        [
            {
                "point": np.asarray(entry["point"], dtype=float).copy(),
                "normal": np.asarray(entry["normal"], dtype=float).copy(),
                "depth": float(entry.get("depth", 0.0)),
            }
            for entry in frame
            if "point" in entry and "normal" in entry
        ]
        for frame in contact_frames
    ]


def _deserialize_contact_frames(
    serialized_frames: Optional[List[List[Dict[str, object]]]],
    frame_count: int,
) -> List[List[SimpleNamespace]]:
    if serialized_frames is None:
        serialized_frames = []
    playback_frames: List[List[SimpleNamespace]] = []
    for idx in range(frame_count):
        raw_frame: List[Dict[str, object]] = serialized_frames[idx] if idx < len(serialized_frames) else []
        contacts: List[SimpleNamespace] = []
        for entry in raw_frame or []:
            point = _vector3(entry.get("point"))
            normal = _vector3(entry.get("normal"))
            if point is None or normal is None:
                continue
            depth = float(entry.get("depth", 0.0))
            contact = SimpleNamespace(point=point, normal=normal, depth=depth)
            contacts.append(SimpleNamespace(contact=contact))
        playback_frames.append(contacts)
    return playback_frames


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
    step_key: Optional[str] = "Right",
    back_key: Optional[str] = "Left",
    restart_key: Optional[str] = "r",
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
    info_line = [
        f"PyVista {pv.__version__}",
        f"{pause_key}=pause",
        f"{quit_key}=quit",
        f"{toggle_contacts_key}=contacts",
    ]
    if restart_key:
        info_line.append(f"{restart_key}=restart")
    plotter.add_text("  |  ".join(info_line), font_size=10)
    paused = {"flag": bool(initial_paused)}
    contact_overlay = {"show": bool(show_contacts), "actors": []}

    initial_states: List[Dict[str, np.ndarray]] = []
    for b in bodies_list:
        state: Dict[str, np.ndarray] = {}
        if hasattr(b, "position") and getattr(b, "position") is not None:
            state["position"] = b.position.copy()
        if hasattr(b, "velocity") and getattr(b, "velocity") is not None:
            state["velocity"] = b.velocity.copy()
        if hasattr(b, "linear_vel") and getattr(b, "linear_vel") is not None:
            state["linear_vel"] = np.asarray(b.linear_vel).copy()
        if hasattr(b, "ang_vel") and getattr(b, "ang_vel") is not None:
            state["ang_vel"] = np.asarray(b.ang_vel).copy()
        if hasattr(b, "pos") and getattr(b, "pos") is not None:
            state["pos"] = np.asarray(b.pos).copy()
        initial_states.append(state)

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

    has_step_back = callable(getattr(solver, "step_back", None))

    def _step_back_once():
        if paused["flag"] and has_step_back:
            try:
                solver.step_back()
            except Exception:
                pass

    def _reset_solver_state():
        reset_method = None
        for attr in ("reset", "restart", "reset_simulation"):
            cand = getattr(solver, attr, None)
            if callable(cand):
                reset_method = cand
                break
        if not reset_method:
            cand = getattr(solver, "reset_state", None)
            if callable(cand):
                reset_method = cand

        if reset_method:
            try:
                reset_method()
                return True
            except Exception:
                pass

        for b, state in zip(bodies_list, initial_states):
            pos = state.get("position")
            if pos is not None and hasattr(b, "position"):
                np.copyto(b.position, pos)
            vel = state.get("velocity")
            if vel is not None and hasattr(b, "velocity"):
                np.copyto(b.velocity, vel)
            lin_vel = state.get("linear_vel")
            if lin_vel is not None and hasattr(b, "linear_vel"):
                b.linear_vel = lin_vel.copy()
            ang_vel = state.get("ang_vel")
            if ang_vel is not None and hasattr(b, "ang_vel"):
                b.ang_vel = ang_vel.copy()
            pos_attr = state.get("pos")
            if pos_attr is not None and hasattr(b, "pos"):
                b.pos = pos_attr.copy()
        if hasattr(solver, "contact_constraints"):
            solver.contact_constraints = []
        if hasattr(solver, "manifolds") and isinstance(getattr(solver, "manifolds"), dict):
            solver.manifolds.clear()
        if hasattr(solver, "_frame_id"):
            solver._frame_id = 0
        return True

    def _restart_simulation():
        if not _reset_solver_state():
            return
        paused["flag"] = True
        contact_overlay["show"] = False
        _clear_contact_overlay(plotter, contact_overlay)
        for update in updaters:
            update()
        try:
            plotter.render()
        except Exception:
            pass

    plotter.add_key_event(pause_key, _toggle_pause)
    if step_key:
        plotter.add_key_event(step_key, _step_once)
    if back_key and has_step_back:
        plotter.add_key_event(back_key, _step_back_once)
    if restart_key:
        plotter.add_key_event(restart_key, _restart_simulation)
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


class _PlaybackSolver:
    """Minimal solver shim that replays recorded frames."""

    def __init__(
        self,
        bodies: List[object],
        frames: List[List[np.ndarray]],
        dt: float,
        contact_frames: Optional[List[List[SimpleNamespace]]] = None,
    ):
        self.bodies = bodies
        self.frames = frames
        self.dt = float(dt)
        self.contact_constraints: List[ContactConstraint] = []
        self._contact_frames: List[List[SimpleNamespace]] = contact_frames or [[] for _ in frames]
        if len(self._contact_frames) < len(self.frames):
            missing = len(self.frames) - len(self._contact_frames)
            self._contact_frames.extend([] for _ in range(missing))
        elif len(self._contact_frames) > len(self.frames):
            self._contact_frames = self._contact_frames[: len(self.frames)]
        self._current_idx = 0
        if self.frames:
            self._apply_frame(0)

    def _apply_frame(self, idx: int):
        snapshot = self.frames[idx]
        for body, pose in zip(self.bodies, snapshot):
            body.position[: pose.shape[0]] = pose
        contacts = self._contact_frames[idx] if idx < len(self._contact_frames) else []
        self.contact_constraints = list(contacts)

    def step(self):
        if not self.frames:
            return
        if self._current_idx >= len(self.frames) - 1:
            return
        self._current_idx += 1
        self._apply_frame(self._current_idx)

    def step_back(self):
        if not self.frames:
            return
        if self._current_idx <= 0:
            return
        self._current_idx -= 1
        self._apply_frame(self._current_idx)

    def reset(self):
        self._current_idx = 0
        if self.frames:
            self._apply_frame(0)

def _snapshot_frames(frames: List[List[np.ndarray]]):
    return [[pose.copy() for pose in frame] for frame in frames]


def _save_recording(
    save_path: Union[str, Path],
    bodies: List[object],
    frames: List[List[np.ndarray]],
    metadata: Dict[str, object],
    contact_frames: Optional[List[List[Dict[str, object]]]] = None,
) -> str:
    path = Path(save_path).expanduser()
    if not path.suffix:
        path = path.with_suffix(".pkl")
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "frames": _snapshot_frames(frames),
        "bodies": [copy.deepcopy(b) for b in bodies],
        "metadata": metadata,
    }
    if contact_frames is not None:
        payload["contact_frames"] = _snapshot_contact_frames(contact_frames)
    with path.open("wb") as fh:
        pickle.dump(payload, fh)
    return str(path)


def _load_recording(save_path: str):
    path = Path(save_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Recording '{path}' does not exist")
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("Recording payload is invalid")
    if "frames" not in payload or "bodies" not in payload:
        raise ValueError("Recording file missing required data ('frames', 'bodies')")
    frames = list(payload["frames"])
    bodies = list(payload["bodies"])
    metadata = payload.get("metadata", {}) or {}
    contact_frames = payload.get("contact_frames")
    return bodies, frames, metadata, contact_frames


def run_visualizer_headless(
    solver,
    bodies: Iterable[object],
    num_steps: int,
    *,
    save_path: Optional[Union[str, Path, bool]] = None,
    default_save_stem: Optional[str] = None,
    record_interval: int = 1,
    show_progress: bool = True,
    prompt_viewer: bool = True,
    viewer_kwargs: Optional[Dict[str, object]] = None,
):
    """
    Run the solver headlessly for a fixed number of steps, recording body poses.

    Saving options:
    1. Leave `save_path` as None (default) to save using the calling script's name with
       an auto-incremented suffix (e.g., `cantilever_beam_test_01.pkl`).
    2. Provide a string/Path in `save_path` to use that name or location (relative names
       are placed in `output/frames_list`). If the target exists, another suffix is added.
       Pass `False` explicitly to skip saving.

    Returns a dict with the recorded frames so callers can persist or postprocess them.
    """

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if record_interval <= 0:
        raise ValueError("record_interval must be positive")

    save_target: Optional[Path] = None
    if save_path is not False:
        base = _base_save_path(None if save_path is None else save_path, default_save_stem)
        save_target, conflict = _unique_save_path(base, always_number=save_path is None)
        if conflict is not None:
            print(f"[pyvista] WARNING: '{conflict}' exists; saving as '{save_target}' instead.")

    bodies_list = list(bodies)
    if not bodies_list:
        raise ValueError("No bodies were provided to record")

    frames: List[List[np.ndarray]] = [[b.position.copy() for b in bodies_list]]
    contact_frames: List[List[Dict[str, object]]] = [
        _snapshot_contact_constraints(getattr(solver, "contact_constraints", []))
    ]

    progress_bar = None
    if show_progress:
        progress_bar = tqdm(total=num_steps, desc="Simulating", unit="step")

    for step_idx in range(num_steps):
        solver.step()
        if progress_bar:
            progress_bar.update(1)
        should_record = ((step_idx + 1) % record_interval == 0) or (step_idx + 1 == num_steps)
        if should_record:
            frames.append([b.position.copy() for b in bodies_list])
            contact_frames.append(
                _snapshot_contact_constraints(getattr(solver, "contact_constraints", []))
            )

    if progress_bar:
        progress_bar.close()

    result = {
        "frames": frames,
        "num_frames": len(frames),
        "num_steps": num_steps,
        "record_interval": record_interval,
        "contact_frames": contact_frames,
    }
    if save_target:
        metadata = {
            "num_frames": len(frames),
            "num_steps": num_steps,
            "record_interval": record_interval,
            "dt": getattr(solver, "dt", 0.0),
        }
        saved_path = _save_recording(save_target, bodies_list, frames, metadata, contact_frames)
        result["saved_file"] = saved_path

    if prompt_viewer and frames:
        try:
            answer = input("Open PyVista replay viewer now? [y/N]: ")
        except EOFError:
            answer = ""
        answer = (answer or "").strip().lower()
        if answer in {"y", "yes"}:
            viewer_kwargs = viewer_kwargs or {}
            replay_bodies = [copy.deepcopy(b) for b in bodies_list]
            playback_contacts = _deserialize_contact_frames(contact_frames, len(frames))
            playback_solver = _PlaybackSolver(
                replay_bodies,
                frames,
                getattr(solver, "dt", 0.0),
                playback_contacts,
            )
            run_visualizer(
                playback_solver,
                replay_bodies,
                initial_paused=True,
                **viewer_kwargs,
            )

    return result


def run_visualizer_from_save(
    save_path: str,
    *,
    viewer_kwargs: Optional[Dict[str, object]] = None,
):
    """
    Load a saved headless recording from disk and launch the PyVista viewer.
    """

    bodies, frames, metadata, contact_records = _load_recording(save_path)
    if not frames:
        raise ValueError(f"Recording '{save_path}' does not contain any frames")

    viewer_opts = dict(viewer_kwargs or {})
    viewer_opts.setdefault("initial_paused", True)
    playback_contacts = _deserialize_contact_frames(contact_records, len(frames))
    playback_solver = _PlaybackSolver(
        bodies,
        frames,
        float(metadata.get("dt", 0.0)),
        playback_contacts,
    )
    run_visualizer(playback_solver, bodies, **viewer_opts)

    return {
        "frames": frames,
        "metadata": metadata,
        "recording_path": str(Path(save_path).expanduser()),
        "contact_frames": contact_records,
    }

__all__ = ["run_visualizer", "run_visualizer_headless", "run_visualizer_from_save"]
