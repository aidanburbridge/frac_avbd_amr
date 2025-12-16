"""Utility helpers for transforming voxelized assemblies.

The :class:`VoxelAssembly` class wraps a collection of ``box_3D`` bodies and
their corresponding constraints (typically ``FaceBondPoint`` bonds).  The
class offers a small, chainable API to duplicate assemblies and apply rigid
body transforms or uniform scaling before they are registered with the
solver.  The helpers keep the constraints consistent by updating the cached
anchor data whenever the geometry changes.

Example
-------
    >>> assembly = VoxelAssembly(boxes, beam_bonds)
    >>> cantilever = assembly.copy().fix_faces(["bottom"])
    >>> falling = (
    ...     assembly.copy()
    ...     .rotate(axis=(0, 1, 0), angle=90.0, degrees=True)
    ...     .translate((0.0, 30.0, 40.0))
    ... )
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from geometry.primitives import (
    box_3D,
    quat_from_axis_angle,
    quat_mult,
    quat_normalize,
    quat_to_rotmat,
)
from geometry.bond_data import BondData
from py_solver.constraints import Constraint, FaceBondPoint


_FACE_KEY_MAP = {
    "left": (0, "min"),
    "right": (0, "max"),
    "back": (1, "min"),
    "front": (1, "max"),
    "bottom": (2, "min"),
    "top": (2, "max"),
}

_FACE_ALIASES = {
    "rear": "back",
    "forward": "front",
    "up": "top",
    "down": "bottom",
}

_AXIS_MAP = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}


def _as_vector(value: Sequence[float] | float) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        arr = np.full(3, float(arr))
    if arr.shape != (3,):
        raise ValueError("Expected a scalar or a 3-vector")
    return arr


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0.0:
        return v
    return v / n


def _clone_constraints(constraints: Iterable[Constraint], body_map: dict[int, box_3D]) -> list[Constraint]:
    clones: list[Constraint] = []
    for constr in constraints:
        new_c = copy.deepcopy(constr)
        if getattr(new_c, "bodyA", None) is not None:
            new_c.bodyA = body_map.get(id(constr.bodyA), new_c.bodyA)
        if getattr(new_c, "bodyB", None) is not None:
            new_c.bodyB = body_map.get(id(constr.bodyB), new_c.bodyB)
        # Drop cached state so the solver recomputes everything on first frame
        if hasattr(new_c, "_cache"):
            new_c._cache = None
        if hasattr(new_c, "_rest_initialized"):
            new_c._rest_initialized = False
        clones.append(new_c)
    return clones


def _reset_constraint_state(constraints: Iterable[Constraint]) -> None:
    for constr in constraints:
        if hasattr(constr, "rest"):
            constr.rest[...] = 0.0
        if hasattr(constr, "_rest_initialized"):
            constr._rest_initialized = False
        if hasattr(constr, "_cache"):
            constr._cache = None
        if hasattr(constr, "_frame_initialized"):
            constr._frame_initialized = False


def _rotate_constraint_frames(constraints: Iterable[Constraint], R: np.ndarray) -> None:
    for constr in constraints:
        if isinstance(constr, FaceBondPoint):
            n = getattr(constr, "n", None)
            t1 = getattr(constr, "t1", None)
            t2 = getattr(constr, "t2", None)
            if n is None or t1 is None or t2 is None:
                # Newly constructed bonds might not have cached frames yet; skip them.
                continue
            constr.n = _normalize(R @ n)
            constr.t1 = _normalize(R @ t1)
            constr.t2 = _normalize(R @ t2)
        elif isinstance(constr, BondData):
            constr.normal = _normalize(R @ np.asarray(constr.normal, dtype=float))


def _scale_constraint_anchors(constraints: Iterable[Constraint], scale: float) -> None:
    for constr in constraints:
        if isinstance(constr, FaceBondPoint):
            constr.pA_local = np.asarray(constr.pA_local, dtype=float) * scale
            constr.pB_local = np.asarray(constr.pB_local, dtype=float) * scale
        elif isinstance(constr, BondData):
            constr.pA_local = np.asarray(constr.pA_local, dtype=float) * scale
            constr.pB_local = np.asarray(constr.pB_local, dtype=float) * scale


def _update_mass_properties(body: box_3D) -> None:
    if body.static:
        return

    w, h, d = body.size
    volume = float(w * h * d)
    body.mass = volume * body.density
    body.inv_mass = 0.0 if body.mass <= 0.0 else 1.0 / body.mass

    Ixx = (1.0 / 12.0) * body.mass * (h * h + d * d)
    Iyy = (1.0 / 12.0) * body.mass * (w * w + d * d)
    Izz = (1.0 / 12.0) * body.mass * (w * w + h * h)
    body.inertia = np.diag([Ixx, Iyy, Izz])
    body.inv_inertia = np.diag([
        0.0 if Ixx <= 0.0 else 1.0 / Ixx,
        0.0 if Iyy <= 0.0 else 1.0 / Iyy,
        0.0 if Izz <= 0.0 else 1.0 / Izz,
    ])
    mI3 = body.mass * np.eye(3)
    body.mass_mat = np.block([
        [mI3, np.zeros((3, 3))],
        [np.zeros((3, 3)), body.I_world()],
    ])


@dataclass
class Bounds:
    mins: np.ndarray
    maxs: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.mins + self.maxs)


class VoxelAssembly:
    """Container that groups voxels and their bonds for easy manipulation."""

    _next_tag: int = 0

    def __init__(self, bodies: Iterable[box_3D], constraints: Iterable[Constraint] | None = None):
        bodies = list(bodies)
        if not bodies:
            raise ValueError("VoxelAssembly requires at least one body")
        self.bodies: list[box_3D] = bodies
        self.constraints: list[Constraint] = list(constraints or [])
        self.assembly_tag: int | None = None
        self.set_assembly_id()

    # ----- cloning & bookkeeping -----
    def copy(self) -> "VoxelAssembly":
        new_bodies = [copy.deepcopy(b) for b in self.bodies]
        body_map = {id(old): new for old, new in zip(self.bodies, new_bodies)}
        for body in new_bodies:
            body.body_id = None
        new_constraints = _clone_constraints(self.constraints, body_map)
        return VoxelAssembly(new_bodies, new_constraints)

    @classmethod
    def _allocate_tag(cls) -> int:
        tag = cls._next_tag
        cls._next_tag += 1
        return tag

    def set_assembly_id(self, tag: int | None = None) -> "VoxelAssembly":
        """Assign the same assembly id to every body in this group."""
        if tag is None:
            tag = self._allocate_tag()
        self.assembly_tag = int(tag)
        for body in self.bodies:
            body.assembly_id = self.assembly_tag
        return self

    # ----- aggregate info -----
    def bounds(self) -> Bounds:
        mins = np.array([np.inf, np.inf, np.inf], dtype=float)
        maxs = -mins
        for body in self.bodies:
            aabb = body.get_aabb()
            mins[0] = min(mins[0], aabb.min_x)
            maxs[0] = max(maxs[0], aabb.max_x)
            mins[1] = min(mins[1], aabb.min_y)
            maxs[1] = max(maxs[1], aabb.max_y)
            mins[2] = min(mins[2], aabb.min_z)
            maxs[2] = max(maxs[2], aabb.max_z)
        return Bounds(mins=mins, maxs=maxs)

    # ----- transforms -----
    def translate(self, offset: Sequence[float]) -> "VoxelAssembly":
        delta = np.asarray(offset, dtype=float)
        if delta.shape != (3,):
            raise ValueError("Translation offset must be a 3-vector")
        for body in self.bodies:
            body.position[:3] += delta
            body.initial_pos[:3] += delta
        _reset_constraint_state(self.constraints)
        return self

    def rotate(
        self,
        axis: Sequence[float],
        angle: float,
        *,
        pivot: Sequence[float] | None = None,
        degrees: bool = False,
    ) -> "VoxelAssembly":
        if degrees:
            angle = np.deg2rad(angle)
        quat = quat_from_axis_angle(np.asarray(axis, dtype=float), angle)
        R = quat_to_rotmat(quat)
        bounds = self.bounds()
        pivot_vec = np.asarray(pivot, dtype=float) if pivot is not None else bounds.center
        for body in self.bodies:
            rel = body.position[:3] - pivot_vec
            rotated = R @ rel + pivot_vec
            body.position[:3] = rotated
            body.initial_pos[:3] = rotated
            body.position[3:] = quat_normalize(quat_mult(quat, body.position[3:]))
            body.initial_pos[3:] = body.position[3:]
            body.velocity[:3] = R @ body.velocity[:3]
            body.velocity[3:] = R @ body.velocity[3:]
            body.prev_vel = body.velocity.copy()
        _rotate_constraint_frames(self.constraints, R)
        _reset_constraint_state(self.constraints)
        return self

    def scale(self, factor: float | Sequence[float], pivot: Sequence[float] | None = None) -> "VoxelAssembly":
        scale_vec = _as_vector(factor)
        if not np.allclose(scale_vec, scale_vec[0]):
            raise ValueError("Voxel voxels only support uniform scaling at the moment")
        s = float(scale_vec[0])
        bounds = self.bounds()
        pivot_vec = np.asarray(pivot, dtype=float) if pivot is not None else bounds.center
        for body in self.bodies:
            rel = body.position[:3] - pivot_vec
            body.position[:3] = pivot_vec + s * rel
            body.initial_pos[:3] = body.position[:3]
            body.size *= s
            _update_mass_properties(body)
        _scale_constraint_anchors(self.constraints, s)
        _reset_constraint_state(self.constraints)
        return self
    
    def align_longest_axis(self, target_axis: str = 'z') -> "VoxelAssembly":

        bounds = self.bounds()
        # Choose longest index from aabb bounds
        current_idx = np.argmax(bounds.maxs - bounds.mins)
        target_idx = _AXIS_MAP[target_axis.lower()]

        if current_idx == target_idx:
            return self
        
        v_curr = np.zeros(3)
        v_target = np.zeros(3)

        # Set target axis index as 1
        v_curr[current_idx] = 1.0
        v_target[target_idx] = 1.0

        # Find rotation axis with cross product
        rot_axis = np.cross(v_curr,v_target)
        return self.rotate(axis=rot_axis, angle=90, degrees=True)
    
    def set_boundary_velocity(
        self,
        faces: Sequence[str],
        velocity: list[float],
        ratio: float | None = None,
        distance: float | None = None,
        debug: bool = False,
    ) -> "VoxelAssembly":

        targets = self.select_boundary(faces, ratio=ratio, distance=distance)
        vel = np.array(velocity, dtype=float)

        if debug:
            self._debug_boundary_selection(targets, faces, vel, ratio, distance)
    
        for b in targets:
            b.mass = np.inf
            b.static = False
            b.velocity[:3] = vel
            b.velocity[3:] = 0.0

        return self

    def set_boundary_fixed(
        self,
        faces: Sequence[str],
        ratio: float | None = None,
        distance: float | None = None,
        debug: bool = False,
    ) -> "VoxelAssembly":
        return self.set_boundary_velocity(
            faces,
            [0.0, 0.0, 0.0],
            ratio=ratio,
            distance=distance,
            debug=debug,
        )

    def set_bond_material(
        self,
        E: float | None = None,
        nu: float | None = None,
    ) -> "VoxelAssembly":
        """Update FaceBondPoint material properties; pass ``None`` to keep a value."""

        if E is None and nu is None:
            return self

        new_E_value = None if E is None else float(E)
        new_nu_value = None if nu is None else float(nu)

        for constr in self.constraints:
            if not isinstance(constr, FaceBondPoint):
                continue

            old_E = getattr(constr, "material_E", None)
            old_nu = getattr(constr, "material_nu", None)

            target_E = new_E_value if new_E_value is not None else old_E
            target_nu = new_nu_value if new_nu_value is not None else old_nu

            n_scale = 1.0
            if target_E is not None and old_E not in (None, 0.0):
                n_scale = target_E / old_E

            if (
                target_E is not None
                and target_nu is not None
                and old_E is not None
                and old_nu is not None
            ):
                old_G = old_E / (2.0 * (1.0 + old_nu))
                new_G = target_E / (2.0 * (1.0 + target_nu))
                t_scale = 1.0 if old_G == 0.0 else new_G / old_G
            else:
                t_scale = 1.0

            constr.stiffness[0] *= n_scale
            constr.stiffness[1:] *= t_scale
            constr.penalty_k[0] *= n_scale
            constr.penalty_k[1:] *= t_scale
            constr.material_E = target_E
            constr.material_nu = target_nu

        _reset_constraint_state(self.constraints)
        return self

    def _debug_boundary_selection(
        self,
        targets: list[box_3D],
        faces: Sequence[str],
        velocity: np.ndarray,
        ratio: float | None,
        distance: float | None,
    ) -> None:
        """Print a summary of voxels tagged by boundary helpers."""
        total = len(self.bodies)
        selected = len(targets)
        fraction = 0.0 if total == 0 else selected / total
        spec = f"distance={distance}" if distance is not None else f"ratio={ratio}"
        vel_str = ", ".join(f"{v:.4g}" for v in velocity)
        print(
            f"[VoxelAssembly] boundary selection faces={list(faces)} {spec} "
            f"count={selected}/{total} ({fraction*100:.2f}%) velocity=({vel_str})"
        )
        # for idx, body in enumerate(targets):
        #     center = body.get_center()
        #     aabb = body.get_aabb()
        #     label = getattr(body, "body_id", None)
        #     if label is None:
        #         label = f"obj-{id(body)}"
        #     print(
        #         f"  [{idx}] body_id={label} assembly={getattr(body, 'assembly_id', None)} "
        #         f"center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}) "
        #         f"size=({body.size[0]:.4f}, {body.size[1]:.4f}, {body.size[2]:.4f}) "
        #         f"aabb=({aabb.min_x:.4f}, {aabb.max_x:.4f}, "
        #         f"{aabb.min_y:.4f}, {aabb.max_y:.4f}, "
        #         f"{aabb.min_z:.4f}, {aabb.max_z:.4f}) "
        #         f"static={body.static}"
        #     )
    
    def select_boundary(self, faces: Sequence[str], ratio: float | None = None, distance: float | None = None) -> list[box_3D]:

        if ratio is not None and distance is not None:
            raise ValueError("Specify either a ratio or distance, not both.")
        elif ratio is None and distance is None:
            raise ValueError("Did not provide either a ratio or a distance.")

        bounds = self.bounds()
        extents = bounds.maxs - bounds.mins
        selection = []

        face_list = []
        for face in faces:
            key = _FACE_ALIASES.get(face.lower(), face.lower())
            if key not in _FACE_KEY_MAP:
                raise ValueError(f"Unknown face '{face}'")
            face_list.append(_FACE_KEY_MAP[key])

        for body in self.bodies:
            aabb = body.get_aabb()
            center = (np.array([aabb.min_x, aabb.min_y, aabb.min_z]) + 
                      np.array([aabb.max_x, aabb.max_y, aabb.max_z])) * 0.5 
            
            for axis, side in face_list:
                limit = bounds.mins[axis] if side == "min" else bounds.maxs[axis]

                if distance is not None:
                    threshold = float(distance)
                else:
                    r = float(ratio) if ratio is not None else 0.1
                    threshold = extents[axis] * r
                
                dist = abs(center[axis] - limit)
                if dist <= threshold:
                    selection.append(body)
                    break
        return selection 
       
    def set_all_static(self, flag: bool) -> "VoxelAssembly":
        for body in self.bodies:
            body.static = bool(flag)
        return self

    def fix_faces(self, faces: Sequence[str], tolerance: float = 1e-6) -> "VoxelAssembly":
        if not faces:
            return self
        bounds = self.bounds()

        face_list: list[tuple[int, str]] = []
        for face in faces:
            key = _FACE_ALIASES.get(face.lower(), face.lower())
            if key not in _FACE_KEY_MAP:
                raise ValueError(f"Unknown face name '{face}'")
            face_list.append(_FACE_KEY_MAP[key])

        for body in self.bodies:
            aabb = body.get_aabb()
            face_hits = False
            for axis, side in face_list:
                limit = bounds.mins[axis] if side == "min" else bounds.maxs[axis]
                if axis == 0:
                    value = aabb.min_x if side == "min" else aabb.max_x
                elif axis == 1:
                    value = aabb.min_y if side == "min" else aabb.max_y
                else:
                    value = aabb.min_z if side == "min" else aabb.max_z
                if abs(value - limit) <= tolerance:
                    face_hits = True
                    break
            if face_hits:
                body.set_static()
        return self
    



__all__ = ["VoxelAssembly", "Bounds"]
