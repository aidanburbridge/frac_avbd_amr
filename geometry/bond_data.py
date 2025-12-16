"""Solver-agnostic container for bond definitions.

`BondData` carries only primitives (indices, local anchors, normal, stiffness
values) so it can be forwarded to either the Python solver (converted to
FaceBondPoint) or the Julia solver (flattened to a numeric array).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class BondData:
    """Minimal bond description detached from solver-specific classes."""

    idxA: int
    idxB: int
    pA_local: np.ndarray
    pB_local: np.ndarray
    normal: np.ndarray
    k_n: float
    k_t: float
    area: float
    tensile_strength: float
    fracture_energy: float

    def as_row(self) -> np.ndarray:
        """Flatten into the 16-value row expected by the hybrid/Julia bridge."""
        row = np.zeros(16, dtype=float)
        row[0] = self.idxA
        row[1] = self.idxB
        row[2:5] = np.asarray(self.pA_local, dtype=float).reshape(3)
        row[5:8] = np.asarray(self.pB_local, dtype=float).reshape(3)
        row[8:11] = np.asarray(self.normal, dtype=float).reshape(3)
        row[11] = float(self.k_n)
        row[12] = float(self.k_t)
        row[13] = float(self.area)
        row[14] = float(self.tensile_strength)
        row[15] = float(self.fracture_energy)
        return row


def rows_from_bonds(bonds: Iterable[BondData]) -> np.ndarray:
    """Stack a sequence of BondData into the hybrid solver's row matrix."""
    rows = [b.as_row() for b in bonds]
    if not rows:
        return np.zeros((0, 16), dtype=float)
    return np.vstack(rows)


def bond_from_facebondpoint(con, idxA: int, idxB: int) -> BondData:
    """
    Create BondData from a FaceBondPoint and resolved body indices.

    Delays the import of constraints to avoid tying this module to the Python
    solver unless needed.
    """
    # Import inside to dodge top-level dependency for Julia-only users.
    from py_solver.constraints import FaceBondPoint  # type: ignore

    if not isinstance(con, FaceBondPoint):
        raise TypeError("Expected FaceBondPoint")

    return BondData(
        idxA=idxA,
        idxB=idxB,
        pA_local=np.asarray(con.pA_local, dtype=float).reshape(3),
        pB_local=np.asarray(con.pB_local, dtype=float).reshape(3),
        normal=np.asarray(con.n_local, dtype=float).reshape(3),
        k_n=float(con.stiffness[0]),
        k_t=float(con.stiffness[1]),
        area=float(getattr(con, "area", 1.0)),
        tensile_strength=float(getattr(con, "tensile_strength", con.stiffness[0])),
        fracture_energy=float(getattr(con, "fracture_energy", 0.0)),
    )


def facebondpoint_from_bonddata(bond: BondData, bodies: Sequence[object]):
    """
    Convert BondData back into a FaceBondPoint for the Python solver.
    """
    from py_solver.constraints import FaceBondPoint  # type: ignore

    bodyA = bodies[bond.idxA]
    bodyB = bodies[bond.idxB]
    return FaceBondPoint(
        bodyA,
        bodyB,
        bond.pA_local,
        bond.pB_local,
        bond.normal,
        bond.k_n,
        bond.k_t,
        bond.area,
        tensile_strength=bond.tensile_strength,
        fracture_energy=bond.fracture_energy,
    )


def bonds_from_facebondpoints(constrs: Iterable[object], idx_map: dict[int, int]) -> list[BondData]:
    """
    Convert FaceBondPoint constraints to BondData using the provided body id map.
    """
    bonds: list[BondData] = []
    for con in constrs:
        a_idx = idx_map.get(id(getattr(con, "bodyA", None)))
        b_idx = idx_map.get(id(getattr(con, "bodyB", None)))
        if a_idx is None or b_idx is None:
            continue
        try:
            bonds.append(bond_from_facebondpoint(con, a_idx, b_idx))
        except TypeError:
            continue
    return bonds
