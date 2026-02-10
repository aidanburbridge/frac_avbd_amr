from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from geometry.bond_data import BondData, rows_from_bonds, bonds_from_facebondpoints


class HybridSolver:
    """
    Thin Python facade over the Julia solver.

    Python handles scene construction; Julia keeps the simulation state and
    performs the heavy math so we avoid per-frame data shuttling.
    """

    def __init__(self, dt: float, iterations: int, gravity: float = -9.81, friction: float = 0.5, project: Optional[str] = None) -> None:
        self.dt = float(dt)
        self.iterations = int(iterations)
        self.gravity = float(gravity)
        self.friction = float(friction)
        self._jl, self._bridge = self._load_bridge(project)
        self._sim = None

    # ------------------------------------------------------------------ #
    # Bridge bootstrap
    # ------------------------------------------------------------------ #
    def _load_bridge(self, project: Optional[str]):
        bridge_path = Path(__file__).resolve().parent / "physics_bridge.jl"

        # Prefer juliacall; fall back to the legacy PyJulia API if needed.
        try:
            from juliacall import Main as jl  # type: ignore

            if project:
                jl.seval(f'using Pkg; Pkg.activate("{project}")')

            jl.seval(f'include("{bridge_path.as_posix()}")')
            bridge_mod = jl.PhysicsBridge
            return jl, bridge_mod
        except ImportError:
            try:
                from julia.api import Julia  # type: ignore
                Julia(compiled_modules=False)
                from julia import Main  # type: ignore

                if project:
                    Main.eval(f'using Pkg; Pkg.activate("{project}")')

                Main.include(str(bridge_path))
                bridge_mod = Main.PhysicsBridge
                return Main, bridge_mod
            except ImportError as exc:  # pragma: no cover - environment specific
                raise RuntimeError(
                    "Julia bridge not available. Install `juliacall` or `julia` (PyJulia)."
                ) from exc

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def initialize(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        bonds: Optional[np.ndarray] = None,
        sizes: Optional[np.ndarray] = None,
        assembly_ids: Optional[np.ndarray] = None,
        amr: Optional[dict] = None,
    ):
        """
        Create or reset the Julia SimulationState in-place.

        positions: (N, 7) -> xyz + quaternion (wxyz)
        velocities: (N, 6) -> linear xyz, angular xyz
        masses: (N,)
        bonds: (M, 17) -> optional face bond table
        sizes: (N, 3) -> optional box extents per body
        """
        pos_arr = np.ascontiguousarray(positions, dtype=np.float64)
        vel_arr = np.ascontiguousarray(velocities, dtype=np.float64)
        mass_arr = np.ascontiguousarray(masses, dtype=np.float64)

        if bonds is None:
            bond_arr = np.zeros((0, 17), dtype=np.float64)
        else:
            bond_arr = np.ascontiguousarray(bonds, dtype=np.float64)
            if bond_arr.ndim == 2 and bond_arr.shape[1] == 16:
                bond_arr = np.hstack((bond_arr, np.zeros((bond_arr.shape[0], 1), dtype=bond_arr.dtype)))

        kwargs = {}
        if sizes is not None:
            kwargs["sizes"] = np.ascontiguousarray(sizes, dtype=np.float64)
        if assembly_ids is not None:
            kwargs["assembly_ids"] = np.ascontiguousarray(assembly_ids, dtype=np.int64)

        if amr:
            kwargs["parent_list"] = np.ascontiguousarray(amr["parent_list"], dtype=np.int32)
            if "children_start" in amr:
                kwargs["children_start"] = np.ascontiguousarray(amr["children_start"], dtype=np.int32)
            else:
                kwargs["children_start"] = np.ascontiguousarray(amr["child_start"], dtype=np.int32)
            if "children_count" in amr:
                kwargs["children_count"] = np.ascontiguousarray(amr["children_count"], dtype=np.int32)
            else:
                kwargs["children_count"] = np.ascontiguousarray(amr["child_count"], dtype=np.int32)
            kwargs["level"] = np.ascontiguousarray(amr["level"], dtype=np.int32)
            kwargs["active"] = np.ascontiguousarray(amr["active"], dtype=np.bool_)
            if "max_ref_level" in amr:
                kwargs["max_ref_level"] = int(amr["max_ref_level"])

        # Ensure Julia sees plain Array{T} rather than PyArray wrappers (juliacall requires this).
        def _to_julia_array(x):
            try:
                # juliacall path
                return self._jl.seval("Array")(x)
            except Exception:
                return x

        pos_jl = _to_julia_array(pos_arr)
        vel_jl = _to_julia_array(vel_arr)
        mass_jl = _to_julia_array(mass_arr)
        bond_jl = _to_julia_array(bond_arr)
        kwargs_jl = {k: _to_julia_array(v) for k, v in kwargs.items()}

        self._sim = self._bridge.init_system(
            pos_jl, vel_jl, mass_jl, bond_jl, self.dt, self.gravity, self.iterations, friction=self.friction, **kwargs_jl
        )
        return self._sim

    def step(self, steps: int = 1) -> None:
        if self._sim is None:
            raise RuntimeError("Call initialize(...) before stepping the simulation.")
        self._bridge.step_batch(self._sim, int(steps))

    def step_timed(self):
        if self._sim is None:
            raise RuntimeError("Call initialize(...) before stepping the simulation.")
        return self._bridge.step_timed(self._sim)

    def get_state(self) -> np.ndarray:

        # TODO add in stress data from bridge
        if self._sim is None:
            raise RuntimeError("No simulation initialized.")
        # Minimal transfer: only pull positions/quaternions to Python.
        return np.array(self._bridge.get_positions(self._sim))
    
    def write_frame(self, filename: str) -> None:
        if self._sim is None:
            return
        
        self._bridge.write_frame(self._sim, str(filename))
    
    def write_energy_csv(self, filename: str, frame_idx: int) -> None:
        if self._sim is None:
            return

        self._bridge.write_energy_csv(self._sim, str(filename), int(frame_idx))

    def get_visualization_data(self) -> tuple[np.ndarray, np.ndarray]:

        if self._sim is None:
            # Return empty arrays
            return np.zeros((0, 6)), np.zeros((0, 6))
        
        # Julia bridge function        
        stress_jl, bond_jl = self._bridge.get_visualization_data(self._sim)

        return np.array(stress_jl), np.array(bond_jl)

# --------------------------------------------------------------------------- #
# Python adapter that mirrors the solver_4 interface for easy swapping.
# --------------------------------------------------------------------------- #

def _body_arrays(
    bodies: Iterable[object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    bodies = list(bodies)
    n = len(bodies)
    pos = np.zeros((n, 7), dtype=np.float64)
    vel = np.zeros((n, 6), dtype=np.float64)
    masses = np.zeros((n,), dtype=np.float64)
    sizes = np.zeros((n, 3), dtype=np.float64)
    assembly_ids = np.full((n,), -1, dtype=np.int64)
    idx_map: dict[int, int] = {}

    for i, b in enumerate(bodies):
        idx_map[id(b)] = i
        p = np.asarray(getattr(b, "position", np.zeros(7)), dtype=float).reshape(-1)
        v = np.asarray(getattr(b, "velocity", np.zeros(6)), dtype=float).reshape(-1)
        s = np.asarray(getattr(b, "size", np.ones(3)), dtype=float).reshape(-1)
        pos[i, : min(7, p.shape[0])] = p[:7]
        vel[i, : min(6, v.shape[0])] = v[:6]
        masses[i] = float(np.inf if getattr(b, "static", False) else getattr(b, "mass", np.inf))
        sizes[i, : min(3, s.shape[0])] = s[:3]
        asm_id = getattr(b, "assembly_id", -1)
        if asm_id is None:
            asm_id = -1
        assembly_ids[i] = int(asm_id)
    return pos, vel, masses, sizes, assembly_ids, idx_map


def _bond_rows(constraints: Iterable[object], idx_map: dict[int, int]) -> np.ndarray:
    # Prefer directly provided BondData
    bonddata = [c for c in constraints if isinstance(c, BondData)]
    if bonddata:
        return rows_from_bonds(bonddata)

    # Fallback: convert FaceBondPoint constraints
    rows = []
    for con in constraints:
        # Late import to avoid hard dependency when only using BondData
        try:
            from py_solver.constraints import FaceBondPoint  # type: ignore
        except ImportError:
            FaceBondPoint = tuple()  # type: ignore

        if not isinstance(con, FaceBondPoint) or getattr(con, "is_broken", False):
            continue

        a_idx = idx_map.get(id(getattr(con, "bodyA", None)))
        b_idx = idx_map.get(id(getattr(con, "bodyB", None)))
        if a_idx is None or b_idx is None:
            continue

        # Convert via BondData helper to reuse flattening
        bond_list = bonds_from_facebondpoints([con], idx_map)
        if bond_list:
            rows.append(bond_list[0].as_row())

    if not rows:
        return np.zeros((0, 17), dtype=np.float64)
    return np.vstack(rows)


class HybridWorld:
    """
    Drop-in adapter for headless tests: exposes a `step()` like solver_4 but
    executes the math in Julia. Use `sync_bodies=False` to measure pure Julia
    speed without round-tripping poses every frame.
    """

    def __init__(
        self,
        bodies: Iterable[object],
        constraints: Iterable[object],
        dt: float,
        iterations: int,
        gravity: float = -9.81,
        friction: float = 0.5,
        project: Optional[str] = None,
        sync_bodies: bool = True,
        amr = None,
    ) -> None:
        self.dt = float(dt)
        self.iterations = int(iterations)
        self.gravity = float(gravity)
        self.friction = float(friction)
        self.bodies = list(bodies)
        self.constraints = list(constraints)
        self.contact_constraints = []  # placeholder for visualizer API
        self._sync_bodies = bool(sync_bodies)
        self._amr = amr or {}

        self._solver = HybridSolver(self.dt, self.iterations, self.gravity, friction=self.friction, project=project)

        pos, vel, masses, sizes, assembly_ids, idx_map = _body_arrays(self.bodies)
        bonds = _bond_rows(self.constraints, idx_map)
        self._solver.initialize(pos, vel, masses, bonds=bonds, sizes=sizes, assembly_ids=assembly_ids, amr=self._amr)

        if self._sync_bodies:
            self._update_bodies(self._solver.get_state())


    def _update_bodies(self, pose_arr: np.ndarray) -> None:
        for i, pose in enumerate(pose_arr):
            b = self.bodies[i]
            if hasattr(b, "position"):
                b.position[:7] = np.asarray(pose, dtype=float)[:7]

    def step(self) -> None:
        self._solver.step(1)
        if self._sync_bodies:
            self._update_bodies(self._solver.get_state())

    def step_timed(self):
        timings = self._solver.step_timed()
        if self._sync_bodies:
            self._update_bodies(self._solver.get_state())
        return timings

    def step_many(self, steps: int) -> None:
        self._solver.step(int(steps))
        if self._sync_bodies:
            self._update_bodies(self._solver.get_state())

    def get_state(self) -> np.ndarray:
        return self._solver.get_state()
    
    def write_frame(self, filename:str) -> None:
        self._solver.write_frame(filename)
    
    def write_energy_csv(self, filename: str, frame_idx: int) -> None:
        self._solver.write_energy_csv(filename, frame_idx)

    def get_visualization_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self._solver.get_visualization_data()

__all__ = ["HybridSolver", "HybridWorld"]
