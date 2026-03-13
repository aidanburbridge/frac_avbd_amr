"""
Interactive voxel-ID picker for STL-based fixtures.

This script voxelizes an STL using the same hierarchy convention as `tests/L_bar.py`,
then lets you tag voxel IDs as either:
    - fixed
    - load

Controls
--------
- `f`: set active selection group to fixed
- `l`: set active selection group to load
- `r`: rotate mode (temporarily disable selection edits)
- `x`: snap to X-normal view (X axis perpendicular to screen)
- `y`: snap to Y-normal view (Y axis perpendicular to screen)
- `z`: snap to Z-normal view (Z axis perpendicular to screen)
- `c`: clear all selections
- `p`: print copy/paste-ready ID lists
- `s`: save JSON selection file
- `q`: quit viewer

Notes
-----
- Picked voxel IDs are body indices in the instantiated `boxes` list.
- These IDs can be copied directly into `tests/L_bar.py` as explicit boundary IDs.

Usage
-----
- `python -m util.selection_tool L_bar`
  Loads STL/voxel settings from `tests/L_bar.py` (with CLI flags as explicit overrides).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import pyvista as pv
except ImportError as exc:  # pragma: no cover - runtime environment specific
    raise SystemExit(
        "pyvista is required for interactive voxel picking. Install it with: pip install pyvista vtk"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import geometry.octree as oct
import geometry.voxelizer as vox


DEFAULT_STL = r"C:\Users\aidan\Documents\TUM\Thesis\L bar fracture.stl"
DEFAULT_RESOLUTION = 1000
DEFAULT_MAX_REF_LEVEL = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive voxel-ID picker for STL fixtures.")
    parser.add_argument(
        "test",
        nargs="?",
        default=None,
        help="Optional test setup module under tests/ (e.g. L_bar or tests/L_bar.py).",
    )
    parser.add_argument("--stl", default=None, help="Path to input STL file (overrides test config).")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Target occupied voxel count for STL voxelization (overrides test config).",
    )
    parser.add_argument(
        "--max-ref-level",
        type=int,
        default=None,
        help="Maximum AMR hierarchy level (same meaning as in tests/L_bar.py; overrides test config).",
    )
    parser.add_argument(
        "--scope",
        choices=("highest", "all", "active"),
        default="highest",
        help=(
            "Selection scope. 'highest' shows only highest-level (coarsest) voxels. "
            "'all' and 'active' are accepted for backward compatibility and map to 'highest'."
        ),
    )
    parser.add_argument(
        "--flood-fill",
        dest="flood_fill",
        action="store_true",
        help="Enable flood-fill interior solidification during voxelization (overrides test config).",
    )
    parser.add_argument(
        "--no-flood-fill",
        dest="flood_fill",
        action="store_false",
        help="Disable flood-fill interior solidification during voxelization (overrides test config).",
    )
    parser.add_argument(
        "--pad-voxels",
        type=int,
        default=None,
        help="AABB padding (voxels) used during voxelization (overrides test config).",
    )
    parser.add_argument(
        "--repair",
        dest="repair",
        action="store_true",
        help="Enable trimesh repair when loading STL (overrides test config).",
    )
    parser.add_argument(
        "--no-repair",
        dest="repair",
        action="store_false",
        help="Disable trimesh repair when loading STL (overrides test config).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path for saved selection (default: <stl_stem>_voxel_selection.json).",
    )
    parser.set_defaults(flood_fill=None, repair=None)
    return parser.parse_args()


def _normalize_test_name(raw_name: str) -> str:
    name = (raw_name or "").strip()
    if not name:
        return ""
    name = name.replace("\\", "/")
    if name.endswith(".py"):
        name = name[:-3]
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    if name.startswith("tests."):
        name = name.split(".", 1)[1]
    return name


def _first_attr(module, names: tuple[str, ...]):
    for attr in names:
        if hasattr(module, attr):
            return getattr(module, attr)
    return None


def _voxelizer_default(name: str, fallback):
    param = inspect.signature(vox.STLVoxelizer.__init__).parameters.get(name)
    if param is None or param.default is inspect._empty:
        return fallback
    return param.default


def _load_test_config(test_name: str) -> dict[str, object]:
    normalized = _normalize_test_name(test_name)
    if not normalized:
        return {}

    module_name = f"tests.{normalized}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Could not load test module '{module_name}': {exc}") from exc

    cfg: dict[str, object] = {"test_name": normalized}

    stl_path = _first_attr(module, ("STL_PATH",))
    if stl_path is not None:
        cfg["stl_path"] = str(stl_path)

    vox_res = _first_attr(module, ("VOX_RESOLUTION", "VOXEL_RESOLUTION", "VOXEL_RES", "VOX_RES"))
    if vox_res is not None:
        try:
            cfg["resolution"] = int(vox_res)
        except Exception:
            pass

    max_ref_level = _first_attr(module, ("MAX_REF_LEVEL", "MAX_LEVEL", "MAX_REFINEMENT_LEVEL"))
    if max_ref_level is not None:
        try:
            cfg["max_ref_level"] = int(max_ref_level)
        except Exception:
            pass

    flood_fill = _first_attr(module, ("FLOOD_FILL",))
    if flood_fill is not None:
        cfg["flood_fill"] = bool(flood_fill)

    pad_voxels = _first_attr(module, ("PAD_VOXELS", "PAD"))
    if pad_voxels is not None:
        try:
            cfg["pad_voxels"] = int(pad_voxels)
        except Exception:
            pass

    repair = _first_attr(module, ("REPAIR_MESH", "REPAIR"))
    if repair is not None:
        cfg["repair"] = bool(repair)

    return cfg


def _contains_fn_factory(stl_voxelizer: vox.STLVoxelizer):
    def _contains_fn(points: np.ndarray) -> np.ndarray:
        return vox._contains_points_chunked(
            stl_voxelizer.mesh,
            np.asarray(points, dtype=float),
            chunk=200_000,
            show_progress=False,
        )

    return _contains_fn


def build_hierarchy_bodies(
    *,
    stl_path: Path,
    resolution: int,
    max_ref_level: int,
    flood_fill: bool,
    pad_voxels: int,
    repair: bool,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    stl_vox = vox.STLVoxelizer(
        str(stl_path),
        pad_voxels=int(pad_voxels),
        flood_fill=bool(flood_fill),
        repair=bool(repair),
    )
    occ, raw_origin, raw_h = stl_vox.voxelize_to_resolution(int(resolution))

    contains_fn = _contains_fn_factory(stl_vox)
    nodes, _, _, _, _, valid_mask, _, _ = oct.build_full_hierarchy(
        coarse_occ=occ,
        max_level=int(max_ref_level),
        origin=raw_origin,
        h_base=raw_h,
        contains_fn=contains_fn,
    )

    bodies, _, active_list = oct.instantiate_boxes_from_tree(
        nodes,
        origin=raw_origin,
        h_base=raw_h,
        density=1.0,
        penalty_gain=1.0e5,
        static=True,
        show_progress=False,
        valid_mask=valid_mask,
    )

    for idx, body in enumerate(bodies):
        body.body_id = idx

    levels = np.asarray([leaf.level for leaf in nodes], dtype=np.int32)
    return (
        bodies,
        np.asarray(valid_mask, dtype=bool),
        np.asarray(active_list, dtype=bool),
        levels,
        float(raw_h),
        np.asarray(raw_origin, dtype=float),
    )


def build_pick_mesh(bodies: list, selectable_ids: np.ndarray) -> tuple[pv.DataSet, dict[int, int]]:
    if selectable_ids.size == 0:
        raise ValueError("No selectable voxels found for the requested scope.")

    # Reuse the same UnstructuredGrid assembly pattern used in util/vtk_exporter.py.
    points = []
    cells = []
    voxel_ids: list[int] = []
    offset = 0

    for voxel_id in selectable_ids.tolist():
        body = bodies[int(voxel_id)]
        corners = np.asarray(body.get_corners(), dtype=float)
        points.append(corners)
        cell_ids = np.arange(offset, offset + 8, dtype=np.int64)
        cells.append(np.concatenate(([8], cell_ids)))
        voxel_ids.append(int(voxel_id))
        offset += 8

    points_arr = np.vstack(points)
    cells_arr = np.hstack(cells)
    n = len(voxel_ids)
    mesh = pv.UnstructuredGrid(
        cells_arr,
        np.array([pv.CellType.HEXAHEDRON] * n),
        points_arr,
    )
    mesh.cell_data["voxel_id"] = np.asarray(voxel_ids, dtype=np.int32)
    mesh.cell_data["state"] = np.zeros(n, dtype=np.uint8)

    id_to_cell = {voxel_id: idx for idx, voxel_id in enumerate(voxel_ids)}
    return mesh, id_to_cell


def main() -> int:
    args = parse_args()

    test_cfg = _load_test_config(args.test) if args.test else {}
    test_name = test_cfg.get("test_name")
    default_flood_fill = bool(_voxelizer_default("flood_fill", True))
    default_pad_voxels = int(_voxelizer_default("pad_voxels", 1))
    default_repair = bool(_voxelizer_default("repair", True))

    stl_value = args.stl if args.stl is not None else test_cfg.get("stl_path", DEFAULT_STL)
    resolution = int(args.resolution if args.resolution is not None else test_cfg.get("resolution", DEFAULT_RESOLUTION))
    max_ref_level = int(
        args.max_ref_level if args.max_ref_level is not None else test_cfg.get("max_ref_level", DEFAULT_MAX_REF_LEVEL)
    )
    flood_fill = bool(
        args.flood_fill if args.flood_fill is not None else test_cfg.get("flood_fill", default_flood_fill)
    )
    pad_voxels = int(
        args.pad_voxels if args.pad_voxels is not None else test_cfg.get("pad_voxels", default_pad_voxels)
    )
    repair = bool(args.repair if args.repair is not None else test_cfg.get("repair", default_repair))

    stl_path = Path(str(stl_value)).expanduser()
    if not stl_path.is_file():
        raise SystemExit(f"STL not found: {stl_path}")

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else Path.cwd() / f"{stl_path.stem}_voxel_selection.json"
    )

    (
        bodies,
        valid_mask,
        active_mask,
        levels,
        voxel_h,
        voxel_origin,
    ) = build_hierarchy_bodies(
        stl_path=stl_path,
        resolution=resolution,
        max_ref_level=max_ref_level,
        flood_fill=flood_fill,
        pad_voxels=pad_voxels,
        repair=repair,
    )

    selectable_mask = valid_mask & active_mask

    selectable_ids = np.flatnonzero(selectable_mask).astype(np.int64)
    mesh, id_to_cells = build_pick_mesh(bodies, selectable_ids)

    fixed_ids: set[int] = set()
    load_ids: set[int] = set()
    mode = {"name": "rotate"}  # fixed | load | rotate
    state_scalars = np.asarray(mesh.cell_data["state"])

    def _apply_state_colors() -> None:
        state_scalars.fill(0)
        for voxel_id in fixed_ids:
            cell_idx = id_to_cells.get(voxel_id)
            if cell_idx is not None:
                state_scalars[cell_idx] = 1
        for voxel_id in load_ids:
            cell_idx = id_to_cells.get(voxel_id)
            if cell_idx is not None:
                state_scalars[cell_idx] = 2
        try:
            mesh.GetCellData().Modified()
            mesh.Modified()
        except Exception:
            pass

    def _print_lists() -> None:
        fixed_list = sorted(fixed_ids)
        load_list = sorted(load_ids)
        print()
        if test_name:
            print(f"test = tests.{test_name}")
        print(f"stl = {stl_path}")
        print(
            f"resolution = {resolution} | max_ref_level = {max_ref_level} | "
            f"flood_fill = {flood_fill} | pad_voxels = {pad_voxels} | repair = {repair}"
        )
        print(f"mode = {mode['name']}")
        print(
            f"counts: total={len(bodies)} selectable={int(selectable_ids.size)} "
            f"fixed={len(fixed_list)} load={len(load_list)}"
        )
        print("Copy/paste into tests/L_bar.py:")
        print(f"FIXED_VOXEL_IDS = {fixed_list}")
        print(f"LOAD_VOXEL_IDS = {load_list}")
        print()

    def _save_json() -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "test_name": test_name,
            "stl_path": str(stl_path),
            "vox_resolution": int(resolution),
            "max_ref_level": int(max_ref_level),
            "scope": "highest",
            "scope_requested": args.scope,
            "flood_fill": bool(flood_fill),
            "pad_voxels": int(pad_voxels),
            "repair": bool(repair),
            "voxel_h": float(voxel_h),
            "origin": voxel_origin.tolist(),
            "counts": {
                "total_bodies": int(len(bodies)),
                "valid_bodies": int(valid_mask.sum()),
                "active_bodies": int(active_mask.sum()),
                "selectable_bodies": int(selectable_ids.size),
                "fixed": int(len(fixed_ids)),
                "load": int(len(load_ids)),
            },
            "fixed_ids": sorted(fixed_ids),
            "load_ids": sorted(load_ids),
            "selected_centers": {
                "fixed": {
                    str(idx): np.asarray(bodies[idx].get_center(), dtype=float).tolist()
                    for idx in sorted(fixed_ids)
                },
                "load": {
                    str(idx): np.asarray(bodies[idx].get_center(), dtype=float).tolist()
                    for idx in sorted(load_ids)
                },
            },
            "selected_levels": {
                "fixed": {str(idx): int(levels[idx]) for idx in sorted(fixed_ids)},
                "load": {str(idx): int(levels[idx]) for idx in sorted(load_ids)},
            },
        }
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _set_mode(name: str) -> bool:
        if mode["name"] == name:
            return False
        mode["name"] = name
        return True

    def _clear_all() -> None:
        fixed_ids.clear()
        load_ids.clear()
        _apply_state_colors()
        try:
            plotter.render()
        except Exception:
            pass

    def _extract_voxel_ids(picked_cells) -> list[int]:
        if picked_cells is None:
            return []
        cell_data = getattr(picked_cells, "cell_data", None)
        if cell_data is None:
            return []
        if "voxel_id" not in cell_data:
            return []
        raw = np.asarray(cell_data["voxel_id"], dtype=np.int64).reshape(-1)
        if raw.size == 0:
            return []
        return sorted(set(int(v) for v in raw.tolist()))

    def _on_pick(picked_cells) -> None:
        if mode["name"] == "rotate":
            return
        selected = _extract_voxel_ids(picked_cells)
        if not selected:
            return

        if mode["name"] == "fixed":
            primary, secondary = fixed_ids, load_ids
        else:
            primary, secondary = load_ids, fixed_ids

        for voxel_id in selected:
            if voxel_id in primary:
                primary.remove(voxel_id)
            else:
                primary.add(voxel_id)
                secondary.discard(voxel_id)

        _apply_state_colors()
        try:
            plotter.render()
        except Exception:
            pass

    def _snap_x() -> None:
        try:
            plotter.view_yz()
        except Exception:
            pass

    def _snap_y() -> None:
        try:
            plotter.view_xz()
        except Exception:
            pass

    def _snap_z() -> None:
        try:
            plotter.view_xy()
        except Exception:
            pass

    def _quit() -> None:
        try:
            plotter.close()
        except Exception:
            pass
        # Hard exit to guarantee no lingering process after closing with q.
        os._exit(0)

    _apply_state_colors()

    pv.set_plot_theme("document")
    plotter = pv.Plotter(window_size=(1400, 900))
    plotter.set_background("white")
    try:
        # Orthographic camera (no perspective distortion).
        plotter.enable_parallel_projection()
    except Exception:
        pass

    scalar_bar_args = {
        "title": "State (0=none, 1=fixed, 2=load)",
        "n_labels": 3,
    }
    plotter.add_mesh(
        mesh,
        scalars="state",
        cmap=["#d0d0d0", "#1f77b4", "#d62728"],
        clim=(0, 2),
        show_edges=True,
        edge_color="#1a1a1a",
        opacity=1.0,
        scalar_bar_args=scalar_bar_args,
        copy_mesh=False,
    )
    plotter.add_axes()

    try:
        plotter.add_legend(
            [
                ["unassigned", "#d0d0d0"],
                ["fixed", "#1f77b4"],
                ["load", "#d62728"],
            ]
        )
    except Exception:
        pass

    help_text = (
        "Voxel Picker Controls\n"
        "f/l/r: mode toggles (fixed/load/rotate)\n"
        "c:clear all | x/y/z: snap views\n"
        "starts in rotate mode | picker active only in fixed/load modes\n"
        "p:print IDs | s:save JSON | q:quit"
    )
    plotter.add_text(help_text, position="upper_left", font_size=8)

    picker_state = {"enabled": False}

    def _enable_picker() -> None:
        if picker_state["enabled"]:
            return
        try:
            plotter.enable_cell_picking(
                callback=_on_pick,
                through=True,
                show=False,
                show_message=False,
                style="surface",
                color="yellow",
                line_width=4,
                start=True,
            )
        except TypeError:
            # Backward compatibility for older PyVista signatures.
            try:
                plotter.enable_cell_picking(
                    callback=_on_pick,
                    through=True,
                    show=False,
                    show_message=False,
                    start=True,
                )
            except TypeError:
                plotter.enable_cell_picking(
                    callback=_on_pick,
                    through=True,
                    show=False,
                    show_message=False,
                )
        picker_state["enabled"] = True

    def _disable_picker() -> None:
        if not picker_state["enabled"]:
            return
        disable_fn = getattr(plotter, "disable_picking", None)
        if callable(disable_fn):
            try:
                disable_fn()
            except Exception:
                pass
        picker_state["enabled"] = False

    def _enter_fixed_mode() -> None:
        changed = _set_mode("fixed")
        if not changed:
            return
        _enable_picker()

    def _enter_load_mode() -> None:
        changed = _set_mode("load")
        if not changed:
            return
        _enable_picker()

    def _enter_rotate_mode() -> None:
        changed = _set_mode("rotate")
        if not changed:
            return
        _disable_picker()

    # Explicit startup state: rotate mode with picker disabled.
    _disable_picker()

    plotter.add_key_event("f", _enter_fixed_mode)
    plotter.add_key_event("F", _enter_fixed_mode)
    plotter.add_key_event("l", _enter_load_mode)
    plotter.add_key_event("L", _enter_load_mode)
    plotter.add_key_event("r", _enter_rotate_mode)
    plotter.add_key_event("R", _enter_rotate_mode)
    plotter.add_key_event("x", _snap_x)
    plotter.add_key_event("X", _snap_x)
    plotter.add_key_event("y", _snap_y)
    plotter.add_key_event("Y", _snap_y)
    plotter.add_key_event("z", _snap_z)
    plotter.add_key_event("Z", _snap_z)
    plotter.add_key_event("c", _clear_all)
    plotter.add_key_event("C", _clear_all)
    plotter.add_key_event("p", _print_lists)
    plotter.add_key_event("P", _print_lists)
    plotter.add_key_event("s", _save_json)
    plotter.add_key_event("S", _save_json)
    plotter.add_key_event("q", _quit)
    plotter.add_key_event("Q", _quit)

    try:
        plotter.show(title="Voxel ID Picker", auto_close=True)
    finally:
        # Ensure VTK resources are released even if the window is closed abruptly.
        try:
            plotter.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
