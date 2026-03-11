"""
L-bar fixture test with an interior load patch.

This scene demonstrates how to target non-extreme regions (interior edges)
using coordinate windows over voxel centers.
"""

from __future__ import annotations

import numpy as np

import geometry.octree as oct
from util.pyvista_visualizer import SimulationSetup
from util.voxel_assembly import VoxelAssembly


# -------------------- Geometry (meters) -------------------- #
MM = 1.0e-3

LEG_WIDTH = 250.0 * MM            # left vertical leg width
TOTAL_WIDTH = 500.0 * MM          # full top width
TOTAL_HEIGHT = 500.0 * MM         # full height
INNER_EDGE_HEIGHT = 250.0 * MM    # z location of inner horizontal edge
THICKNESS = 20.0 * MM             # extrusion thickness (out-of-plane, y-axis)
VOX_SIZE = 10.0 * MM              # base voxel edge length


# -------------------- Boundary-condition controls -------------------- #
BOTTOM_FIX_DEPTH = 20.0 * MM

# Load is applied upward at the underside of the inner horizontal edge,
# centered at this x-offset from the inner corner.
LOAD_OFFSET_FROM_CORNER = 220.0 * MM
LOAD_PATCH_WIDTH = 20.0 * MM
LOAD_BAND_THICKNESS = 20.0 * MM
LOAD_VELOCITY = np.array([0.0, 0.0, 2.0e-3], dtype=float)


# -------------------- Material / solver -------------------- #
DENSITY = 1150.0
PENALTY_GAIN = 1.0e6
E_MODULUS = 2.0e9
NU = 0.30
TENSILE_STRENGTH = 8.0e7
FRACTURE_TOUGHNESS = 5.0e4

DT_PHYSICS = 1 / 4000
DT_RENDER = 1 / 60
STEPS_PER_EXPORT = max(1, int(DT_RENDER / DT_PHYSICS))
ITER = 80
GRAV = 0.0
FRICTION = 0.0
STEPS = 2500

PYTHON_SOLVER_PARAMS = {
    "mu": 0.2,
    "post_stabilize": False,
    "beta": 100.0,
    "alpha": 0.95,
    "gamma": 0.99,
    "debug_contacts": False,
}


def _cells(length: float, h: float) -> int:
    return max(1, int(round(length / h)))


def _build_lbar_occ() -> np.ndarray:
    nx = _cells(TOTAL_WIDTH, VOX_SIZE)
    ny = _cells(THICKNESS, VOX_SIZE)
    nz = _cells(TOTAL_HEIGHT, VOX_SIZE)

    leg_cells = min(nx, _cells(LEG_WIDTH, VOX_SIZE))
    inner_k = min(nz - 1, _cells(INNER_EDGE_HEIGHT, VOX_SIZE))

    occ = np.zeros((nx, ny, nz), dtype=bool)
    occ[:leg_cells, :, :] = True      # full-height vertical leg
    occ[:, :, inner_k:] = True        # top horizontal leg
    return occ


def _select_bodies_by_center_box(
    assembly: VoxelAssembly,
    *,
    x: tuple[float | None, float | None] | None = None,
    y: tuple[float | None, float | None] | None = None,
    z: tuple[float | None, float | None] | None = None,
) -> list:
    def _in_range(val: float, limits: tuple[float | None, float | None] | None) -> bool:
        if limits is None:
            return True
        lo, hi = limits
        if lo is not None and val < lo:
            return False
        if hi is not None and val > hi:
            return False
        return True

    selected = []
    for body in assembly.bodies:
        cx, cy, cz = body.get_center()
        if _in_range(cx, x) and _in_range(cy, y) and _in_range(cz, z):
            selected.append(body)
    return selected


def _set_targets_fixed(targets: list) -> None:
    for body in targets:
        body.set_static()
        body.velocity[:] = 0.0
        body.prev_vel = body.velocity.copy()


def _set_targets_kinematic_velocity(targets: list, velocity: np.ndarray) -> None:
    vel = np.asarray(velocity, dtype=float)
    for body in targets:
        # Match VoxelAssembly.set_boundary_velocity behavior for kinematic grips.
        body.static = False
        body.mass = np.inf
        body.velocity[:3] = vel
        body.velocity[3:] = 0.0
        body.prev_vel = body.velocity.copy()


def build_setup(sync_bodies: bool = True) -> SimulationSetup:
    occ = _build_lbar_occ()
    origin = np.zeros(3, dtype=float)

    leaves, h_base = oct.octree_from_occ(occ, h_base=VOX_SIZE)
    boxes, mapping, _ = oct.instantiate_boxes_from_tree(
        leaves,
        origin=origin,
        h_base=h_base,
        density=DENSITY,
        penalty_gain=PENALTY_GAIN,
        static=False,
        show_progress=False,
    )
    bonds = oct.build_constraints_from_tree(
        leaves,
        boxes,
        mapping,
        E=E_MODULUS,
        nu=NU,
        tensile_strength=TENSILE_STRENGTH,
        fracture_toughness=FRACTURE_TOUGHNESS,
    )

    lbar = VoxelAssembly(boxes, bonds)
    bounds = lbar.bounds()

    x_min, _, z_min = bounds.mins[0], bounds.mins[1], bounds.mins[2]
    x_max = bounds.maxs[0]

    x_corner = x_min + LEG_WIDTH
    z_inner = z_min + INNER_EDGE_HEIGHT

    # 1) Fixed bottom of the vertical leg
    fixed_targets = _select_bodies_by_center_box(
        lbar,
        x=(x_min, x_corner + 0.5 * VOX_SIZE),
        z=(z_min, z_min + BOTTOM_FIX_DEPTH),
    )

    # 2) Interior load patch on the underside of the inner horizontal edge
    x_center = min(x_corner + LOAD_OFFSET_FROM_CORNER, x_max - 0.5 * VOX_SIZE)
    x_half = 0.5 * LOAD_PATCH_WIDTH
    x_lo = max(x_corner + 0.5 * VOX_SIZE, x_center - x_half)
    x_hi = min(x_max - 0.5 * VOX_SIZE, x_center + x_half)

    load_targets = _select_bodies_by_center_box(
        lbar,
        x=(x_lo, x_hi),
        z=(z_inner, z_inner + LOAD_BAND_THICKNESS),
    )

    if not fixed_targets:
        raise ValueError("No fixed support voxels selected. Increase BOTTOM_FIX_DEPTH or reduce VOX_SIZE.")
    if not load_targets:
        raise ValueError(
            "No load-patch voxels selected. Adjust LOAD_OFFSET_FROM_CORNER / LOAD_PATCH_WIDTH / VOX_SIZE."
        )

    _set_targets_fixed(fixed_targets)
    _set_targets_kinematic_velocity(load_targets, LOAD_VELOCITY)

    print(f"L-bar voxels: {len(lbar.bodies)}")
    print(f"L-bar bonds: {len(bonds)}")
    print(f"Fixed voxels: {len(fixed_targets)}")
    print(
        f"Load voxels: {len(load_targets)} "
        f"(x=[{x_lo:.4f}, {x_hi:.4f}], z=[{z_inner:.4f}, {z_inner + LOAD_BAND_THICKNESS:.4f}])"
    )

    return SimulationSetup(
        bodies=lbar.bodies,
        constraints=bonds,
        dt=DT_PHYSICS,
        iterations=ITER,
        gravity=GRAV,
        friction=FRICTION,
        sync_bodies=sync_bodies,
        python_solver_params=PYTHON_SOLVER_PARAMS,
        metadata={
            "vox_size": VOX_SIZE,
            "leg_width": LEG_WIDTH,
            "total_width": TOTAL_WIDTH,
            "total_height": TOTAL_HEIGHT,
            "inner_edge_height": INNER_EDGE_HEIGHT,
            "load_offset_from_corner": LOAD_OFFSET_FROM_CORNER,
            "load_patch_width": LOAD_PATCH_WIDTH,
            "load_band_thickness": LOAD_BAND_THICKNESS,
        },
        headless_steps=STEPS,
        headless_kwargs={
            "steps_per_export": STEPS_PER_EXPORT,
            "show_progress": True,
        },
    )


if __name__ == "__main__":
    from util.pyvista_visualizer import run_simulation

    run_simulation(build_setup())
