from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np


FRAME_MAGIC = b"AVB2"
BOND_META_MAGIC_V1 = b"ABM1"
BOND_META_MAGIC_V2 = b"ABM2"

RAW_BODY_DTYPE = np.dtype(
    [
        ("pos", "3f4"),
        ("quat", "4f4"),
        ("size", "3f4"),
        ("body_id", "i4"),
        ("assembly_id", "i4"),
        ("stress", "6f4"),
    ]
)
RAW_BOND_DTYPE = np.dtype(
    [
        ("bond_id", "i4"),
        ("bodyA_id", "i4"),
        ("bodyB_id", "i4"),
        ("C", "3f4"),
        ("rest", "3f4"),
        ("penalty_k", "3f4"),
        ("damage", "f4"),
        ("is_broken", "u1"),
        ("is_cohesive", "u1"),
        ("_pad", "u2"),
    ]
)
BOND_META_DTYPE_V1 = np.dtype(
    [
        ("bond_id", "i4"),
        ("bodyA_id", "i4"),
        ("bodyB_id", "i4"),
        ("area", "f4"),
    ]
)
BOND_META_DTYPE_V2 = np.dtype(
    [
        ("bond_id", "i4"),
        ("bodyA_id", "i4"),
        ("bodyB_id", "i4"),
        ("area", "f4"),
        ("f_min", "3f4"),
        ("f_max", "3f4"),
    ]
)


def _frame_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _maybe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_data_dir(run_dir: Path) -> tuple[Path, str]:
    raw_dir = run_dir / "raw"
    vtk_dir = run_dir / "vtk"

    if raw_dir.exists() and any(raw_dir.glob("frame_*.bin")):
        return raw_dir, "raw"
    if vtk_dir.exists() and any(vtk_dir.glob("voxels_*.vtu")):
        return vtk_dir, "vtk"
    raise FileNotFoundError(f"No supported frame exports found under {run_dir}")


def _read_energy_rows(data_dir: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for path in sorted(data_dir.glob("energy_*.csv")):
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = _safe_int(row.get("frame"), _frame_index(path))
                parsed = {"frame": frame}
                for key, value in row.items():
                    if key == "frame":
                        continue
                    parsed[key] = _safe_float(value)
                rows[frame] = parsed
    return rows


def _read_step_metric_rows(data_dir: Path) -> dict[int, dict[str, Any]]:
    path = data_dir / "step_metrics.csv"
    if not path.exists():
        return {}

    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = _safe_int(row.get("frame"))
            rows[frame] = {
                "frame": frame,
                "step": _safe_int(row.get("step")),
                "time": _safe_float(row.get("time")),
                "iters_used": _safe_int(row.get("iters_used")),
                "max_violation": _safe_float(row.get("max_violation")),
                "active_body_count": _safe_int(row.get("active_body_count")),
                "active_bond_count": _safe_int(row.get("active_bond_count")),
                "exported_body_count": _safe_int(row.get("exported_body_count")),
                "exported_bond_count": _safe_int(row.get("exported_bond_count")),
                "contact_count": _safe_int(row.get("contact_count")),
            }
    return rows


def _read_bond_meta(data_dir: Path) -> tuple[dict[int, dict[str, Any]], bool]:
    path = data_dir / "bond_meta.bin"
    if not path.exists():
        return {}, False

    with path.open("rb") as f:
        magic = f.read(4)
        count = struct.unpack("i", f.read(4))[0]
        if magic == BOND_META_MAGIC_V2:
            raw = np.frombuffer(f.read(count * BOND_META_DTYPE_V2.itemsize), dtype=BOND_META_DTYPE_V2)
            out = {
                int(row["bond_id"]): {
                    "bodyA_id": int(row["bodyA_id"]),
                    "bodyB_id": int(row["bodyB_id"]),
                    "area": float(row["area"]),
                    "f_min": np.asarray(row["f_min"], dtype=float),
                    "f_max": np.asarray(row["f_max"], dtype=float),
                }
                for row in raw
            }
            return out, True

        if magic == BOND_META_MAGIC_V1:
            raw = np.frombuffer(f.read(count * BOND_META_DTYPE_V1.itemsize), dtype=BOND_META_DTYPE_V1)
            out = {
                int(row["bond_id"]): {
                    "bodyA_id": int(row["bodyA_id"]),
                    "bodyB_id": int(row["bodyB_id"]),
                    "area": float(row["area"]),
                    "f_min": None,
                    "f_max": None,
                }
                for row in raw
            }
            return out, False

    raise ValueError(f"Unsupported bond metadata format in {path}")


def _read_raw_frame(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != FRAME_MAGIC:
            raise ValueError(f"Unsupported frame format in {path}")
        n_bodies, n_bonds, dt = struct.unpack("iif", f.read(12))
        raw_bodies = np.frombuffer(f.read(n_bodies * RAW_BODY_DTYPE.itemsize), dtype=RAW_BODY_DTYPE)
        raw_bonds = (
            np.frombuffer(f.read(n_bonds * RAW_BOND_DTYPE.itemsize), dtype=RAW_BOND_DTYPE)
            if n_bonds > 0
            else np.zeros((0,), dtype=RAW_BOND_DTYPE)
        )

    return {
        "dt": float(dt),
        "body_ids": raw_bodies["body_id"].astype(np.int32),
        "body_pos": np.asarray(raw_bodies["pos"], dtype=float),
        "stress6": np.asarray(raw_bodies["stress"], dtype=float),
        "bond_ids": raw_bonds["bond_id"].astype(np.int32),
        "bodyA_ids": raw_bonds["bodyA_id"].astype(np.int32),
        "bodyB_ids": raw_bonds["bodyB_id"].astype(np.int32),
        "C": np.asarray(raw_bonds["C"], dtype=float),
        "rest": np.asarray(raw_bonds["rest"], dtype=float),
        "penalty_k": np.asarray(raw_bonds["penalty_k"], dtype=float),
        "damage": np.asarray(raw_bonds["damage"], dtype=float),
        "is_broken": np.asarray(raw_bonds["is_broken"], dtype=np.uint8),
        "is_cohesive": np.asarray(raw_bonds["is_cohesive"], dtype=np.uint8),
    }


def _read_vtk_frame(voxel_path: Path, bond_path: Path | None) -> dict[str, Any]:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "PyVista is required to post-process runs that only retain VTK outputs."
        ) from exc

    grid = pv.read(voxel_path)
    centers = np.asarray(grid.cell_centers().points, dtype=float)
    body_ids = np.asarray(grid.cell_data.get("Body_ID", np.arange(grid.n_cells)), dtype=np.int32)
    stress9 = np.asarray(grid.cell_data.get("Stress_Tensor", np.zeros((grid.n_cells, 9))), dtype=float)
    stress6 = np.column_stack(
        (
            stress9[:, 0],
            stress9[:, 4],
            stress9[:, 8],
            stress9[:, 1],
            stress9[:, 5],
            stress9[:, 6],
        )
    )

    if bond_path is not None and bond_path.exists():
        poly = pv.read(bond_path)
        n_bonds = int(poly.n_cells)
        bond_ids = np.asarray(poly.cell_data.get("Bond_ID", np.zeros(n_bonds)), dtype=np.int32)
        bodyA_ids = np.asarray(poly.cell_data.get("BodyA_ID", np.zeros(n_bonds)), dtype=np.int32)
        bodyB_ids = np.asarray(poly.cell_data.get("BodyB_ID", np.zeros(n_bonds)), dtype=np.int32)
        C = np.asarray(poly.cell_data.get("C_Local", np.zeros((n_bonds, 3))), dtype=float)
        rest = np.asarray(poly.cell_data.get("Rest_Local", np.zeros((n_bonds, 3))), dtype=float)
        penalty_k = np.asarray(poly.cell_data.get("Penalty_K", np.zeros((n_bonds, 3))), dtype=float)
        damage = np.asarray(poly.cell_data.get("Damage", np.zeros(n_bonds)), dtype=float)
        is_broken = np.asarray(poly.cell_data.get("Is_Broken", np.zeros(n_bonds)), dtype=np.uint8)
        is_cohesive = np.asarray(poly.cell_data.get("Is_Cohesive", np.zeros(n_bonds)), dtype=np.uint8)
    else:
        bond_ids = np.zeros((0,), dtype=np.int32)
        bodyA_ids = np.zeros((0,), dtype=np.int32)
        bodyB_ids = np.zeros((0,), dtype=np.int32)
        C = np.zeros((0, 3), dtype=float)
        rest = np.zeros((0, 3), dtype=float)
        penalty_k = np.zeros((0, 3), dtype=float)
        damage = np.zeros((0,), dtype=float)
        is_broken = np.zeros((0,), dtype=np.uint8)
        is_cohesive = np.zeros((0,), dtype=np.uint8)

    return {
        "dt": math.nan,
        "body_ids": body_ids,
        "body_pos": centers,
        "stress6": stress6,
        "bond_ids": bond_ids,
        "bodyA_ids": bodyA_ids,
        "bodyB_ids": bodyB_ids,
        "C": C,
        "rest": rest,
        "penalty_k": penalty_k,
        "damage": damage,
        "is_broken": is_broken,
        "is_cohesive": is_cohesive,
    }


def _iter_frames(data_dir: Path, mode: str):
    if mode == "raw":
        for path in sorted(data_dir.glob("frame_*.bin")):
            yield _frame_index(path), _read_raw_frame(path)
        return

    voxel_files = sorted(data_dir.glob("voxels_*.vtu"))
    for voxel_path in voxel_files:
        frame = _frame_index(voxel_path)
        bond_path = data_dir / f"bonds_{frame:04d}.vtp"
        yield frame, _read_vtk_frame(voxel_path, bond_path if bond_path.exists() else None)


def _sym6_to_mats(stress6: np.ndarray) -> np.ndarray:
    n = int(stress6.shape[0])
    mats = np.zeros((n, 3, 3), dtype=float)
    if n == 0:
        return mats
    mats[:, 0, 0] = stress6[:, 0]
    mats[:, 1, 1] = stress6[:, 1]
    mats[:, 2, 2] = stress6[:, 2]
    mats[:, 0, 1] = mats[:, 1, 0] = stress6[:, 3]
    mats[:, 1, 2] = mats[:, 2, 1] = stress6[:, 4]
    mats[:, 0, 2] = mats[:, 2, 0] = stress6[:, 5]
    return mats


def _max_principal_proxy(stress6: np.ndarray) -> np.ndarray:
    if stress6.size == 0:
        return np.zeros((0,), dtype=float)
    return np.linalg.eigvalsh(_sym6_to_mats(stress6))[:, -1]


def _normalized_benchmark_token(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _is_l_panel_benchmark(benchmark_name: str | None, metadata: dict[str, Any]) -> bool:
    for value in (benchmark_name, metadata.get("test_name"), metadata.get("stl_path")):
        token = _normalized_benchmark_token(value)
        if "lbar" in token or "lpanel" in token:
            return True
    return False


def _load_direction(metadata: dict[str, Any], benchmark_name: str | None = None) -> np.ndarray | None:
    for key in ("loading_velocity", "load_velocity", "impact_velocity_vector_m_per_s"):
        value = metadata.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 0:
            continue
        norm = np.linalg.norm(arr)
        if norm > 0.0:
            return arr / norm
    if _is_l_panel_benchmark(benchmark_name, metadata):
        # The L-panel fixture uses a prescribed upward loading patch.
        return np.asarray([0.0, 1.0, 0.0], dtype=float)
    return None


def _extract_material_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    material_keys = [
        key
        for key in metadata
        if any(
            token in key
            for token in (
                "density",
                "penalty_gain",
                "tensile_strength",
                "fracture_toughness",
                "wall_E",
                "projectile_E",
                "wall_nu",
                "projectile_nu",
                "E",
                "nu",
            )
        )
    ]
    return {key: metadata[key] for key in sorted(material_keys)}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _gather_static_bond_values(
    bond_ids: np.ndarray,
    bond_meta: dict[int, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    n = int(bond_ids.shape[0])
    area = np.full((n,), np.nan, dtype=float)
    f_min = np.full((n, 3), np.nan, dtype=float)
    f_max = np.full((n, 3), np.nan, dtype=float)
    all_caps_available = True

    for i, bond_id in enumerate(bond_ids):
        rec = bond_meta.get(int(bond_id))
        if rec is None:
            all_caps_available = False
            continue
        area[i] = float(rec["area"])
        if rec["f_min"] is None or rec["f_max"] is None:
            all_caps_available = False
            continue
        f_min[i, :] = np.asarray(rec["f_min"], dtype=float)
        f_max[i, :] = np.asarray(rec["f_max"], dtype=float)

    return area, f_min, f_max, all_caps_available


def _finite_percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.percentile(finite, q))


def _finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.max(finite))


def _bond_dump_rows(
    *,
    frame: int,
    step: int,
    time_value: float,
    bond_ids: np.ndarray,
    bodyA_ids: np.ndarray,
    bodyB_ids: np.ndarray,
    damage: np.ndarray,
    is_broken: np.ndarray,
    is_cohesive: np.ndarray,
    area: np.ndarray,
    L0: np.ndarray,
    eps_n: np.ndarray,
    gamma_t1: np.ndarray,
    gamma_t2: np.ndarray,
    gamma_eq: np.ndarray,
    row_force: np.ndarray,
    sigma_n: np.ndarray,
    tau_t1: np.ndarray,
    tau_t2: np.ndarray,
    tau_eq: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(int(bond_ids.shape[0])):
        rows.append(
            {
                "frame": int(frame),
                "step": int(step),
                "time": float(time_value),
                "bond_id": int(bond_ids[i]),
                "bodyA_id": int(bodyA_ids[i]),
                "bodyB_id": int(bodyB_ids[i]),
                "damage": float(damage[i]),
                "is_broken": int(is_broken[i]),
                "is_cohesive": int(is_cohesive[i]),
                "area": float(area[i]),
                "L0": float(L0[i]),
                "eps_n": float(eps_n[i]),
                "gamma_t1": float(gamma_t1[i]),
                "gamma_t2": float(gamma_t2[i]),
                "gamma_eq": float(gamma_eq[i]),
                "f_n": float(row_force[i, 0]),
                "f_t1": float(row_force[i, 1]),
                "f_t2": float(row_force[i, 2]),
                "sigma_n": float(sigma_n[i]),
                "tau_t1": float(tau_t1[i]),
                "tau_t2": float(tau_t2[i]),
                "tau_eq": float(tau_eq[i]),
            }
        )
    return rows


def _write_thesis_guide(
    path: Path,
    *,
    load_displacement_available: bool,
    exact_force_caps_available: bool,
    bond_strain_length_note: str | None,
) -> None:
    lines = [
        "# Thesis Quantity Guide",
        "",
        "| Quantity | Meaning | Units | Allowed wording | Forbidden wording |",
        "| --- | --- | --- | --- | --- |",
        "| `peak_stress_proxy` | Maximum principal tensile stress proxy from exported `Stress_Tensor` | stress units of the solver setup | `max principal tensile stress proxy`, `tensile localization indicator` | `true Cauchy stress`, `physical tensile stress in the specimen` |",
        "| `eps_n` | Bond-normal separation normalized by a per-bond characteristic length | dimensionless | `cohesive-bond normal separation ratio`, `bond-normal strain-like quantity` | `continuum normal strain field` |",
        "| `gamma_t1`, `gamma_t2`, `gamma_eq` | Bond-tangential separation ratios normalized by a per-bond characteristic length | dimensionless | `cohesive-bond shear separation`, `bond-local tangential deformation` | `continuum shear strain field` |",
        "| `sigma_n`, `tau_t1`, `tau_t2`, `tau_eq` | Cohesive-bond tractions from capped row forces divided by exported bond area | traction units of the solver setup | `cohesive-bond traction`, `interface traction` | `continuum stress` |",
        "| `crack_area_proxy` | Sum of areas of broken bonds | area | `crack area proxy`, `fractured interface area proxy` | `exact crack surface area` |",
        "| `process_zone_area_proxy` | Sum of areas of damaged but unbroken bonds | area | `process-zone area proxy` | `exact damage-zone area` |",
        "| `damage_positive_bond_count` | Number of bonds with positive exported damage, including broken bonds | count | `damage-positive bond count` | `damaged-but-unbroken bond count`, `process-zone bond count` |",
        "| `broken_bond_count` | Number of broken bonds in the saved frame | count | `broken bond count` | `number of cracks` |",
        "| `damaged_bond_count` | Number of damaged but not yet broken bonds in the saved frame | count | `damaged cohesive-bond count` | `crack count` |",
        "| `load_displacement_along_loading_axis`, `load_displacement_magnitude` | Reconstructed prescribed/loading-point displacement from tracked load-group body motion | displacement in exported coordinate units | `prescribed displacement`, `loading-point displacement`, `kinematic displacement history` | `force-displacement response`, `structural load-displacement curve` |",
        "",
        "## Unsupported or Conditionally Supported Quantities",
        "",
        "- True reaction force is not reconstructable from the current exports.",
        "- Specimen engineering stress is not reconstructable from the current exports.",
        "- A true continuum stress field is not available; the exported tensor must be treated as a stress proxy only.",
        "- Contact stress is not reconstructable from the current exports.",
    ]

    if exact_force_caps_available:
        lines.append("- Exact capped cohesive-bond row forces are reconstructable for this run from the exported static bond metadata.")
    else:
        lines.append("- Exact capped cohesive-bond row forces are not reconstructable for this run because bond force bounds were not exported.")

    if load_displacement_available:
        lines.append("- Load displacement was reconstructed only as a body-motion kinematic quantity, not as a load-force response.")
    else:
        lines.append("- Load displacement was not reconstructed because the required boundary-group tracking data was not stable across saved frames.")

    if bond_strain_length_note:
        lines.append(f"- {bond_strain_length_note}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_run(
    run_dir: Path,
    *,
    damage_threshold: float,
    stress_threshold: float | None,
    bond_dump_mode: str,
) -> dict[str, Any]:
    data_dir, mode = _find_data_dir(run_dir)
    metadata = _maybe_json(run_dir / "meta_data.json")
    manifest = _maybe_json(run_dir / "manifest.json")
    bond_meta, exact_force_caps_available = _read_bond_meta(data_dir)
    energy_rows = _read_energy_rows(data_dir)
    step_metric_rows = _read_step_metric_rows(data_dir)

    dt_value = _safe_float(metadata.get("dt"), _safe_float(metadata.get("dt_physics")))
    if not math.isfinite(dt_value):
        dt_value = _safe_float(manifest.get("dt"), 0.0)

    benchmark_name = (
        metadata.get("test_name")
        or manifest.get("test")
        or run_dir.parent.name
    )
    l_panel_mode = _is_l_panel_benchmark(benchmark_name, metadata)
    bond_strain_length_note: str | None = None

    if stress_threshold is None:
        stress_threshold = _safe_float(metadata.get("refine_stress_threshold"))
        if not math.isfinite(stress_threshold):
            stress_threshold = None

    load_ids = metadata.get("load_voxel_ids") or []
    load_ids = [int(v) for v in load_ids]
    load_dir = _load_direction(metadata, benchmark_name)
    initial_load_positions: dict[int, np.ndarray] | None = None
    load_displacement_available = bool(load_ids)

    time_history_rows: list[dict[str, Any]] = []
    bond_summary_rows: list[dict[str, Any]] = []
    final_bond_dump_rows: list[dict[str, Any]] = []
    all_bond_dump_rows: list[dict[str, Any]] = []

    first_cohesive_frame: int | None = None
    first_cohesive_time: float | None = None
    first_broken_frame: int | None = None
    first_broken_time: float | None = None

    peak_stress_proxy = -math.inf
    peak_stress_frame: int | None = None
    peak_stress_time: float | None = None
    peak_stress_body_id: int | None = None
    peak_stress_pos = np.full((3,), math.nan, dtype=float)

    for frame, frame_data in _iter_frames(data_dir, mode):
        energy_row = energy_rows.get(frame, {})
        metric_row = step_metric_rows.get(frame, {})

        step_value = _safe_int(metric_row.get("step"), frame)
        time_value = _safe_float(metric_row.get("time"))
        if not math.isfinite(time_value):
            time_value = _safe_float(energy_row.get("time"))
        if not math.isfinite(time_value):
            time_value = step_value * dt_value

        body_ids = frame_data["body_ids"]
        body_pos = frame_data["body_pos"]
        stress6 = frame_data["stress6"]

        bond_ids = frame_data["bond_ids"]
        bodyA_ids = frame_data["bodyA_ids"]
        bodyB_ids = frame_data["bodyB_ids"]
        C = frame_data["C"]
        rest = frame_data["rest"]
        penalty_k = frame_data["penalty_k"]
        damage = frame_data["damage"]
        is_broken = frame_data["is_broken"].astype(bool)
        is_cohesive = frame_data["is_cohesive"].astype(bool)

        principal = _max_principal_proxy(stress6)
        peak_frame_proxy = float(np.max(principal)) if principal.size else math.nan
        peak_body_idx = int(np.argmax(principal)) if principal.size else -1
        peak_body = int(body_ids[peak_body_idx]) if peak_body_idx >= 0 else -1
        peak_body_position = body_pos[peak_body_idx] if peak_body_idx >= 0 else np.full((3,), math.nan)

        if principal.size and peak_frame_proxy > peak_stress_proxy:
            peak_stress_proxy = peak_frame_proxy
            peak_stress_frame = frame
            peak_stress_time = time_value
            peak_stress_body_id = peak_body
            peak_stress_pos = np.asarray(peak_body_position, dtype=float)

        if stress_threshold is not None and math.isfinite(stress_threshold) and principal.size:
            exceed_mask = principal >= stress_threshold
            stress_exceedance_count = int(np.count_nonzero(exceed_mask))
            stress_exceedance_fraction = float(stress_exceedance_count / max(principal.size, 1))
        else:
            stress_exceedance_count = 0
            stress_exceedance_fraction = math.nan

        area, f_min, f_max, frame_caps_available = _gather_static_bond_values(bond_ids, bond_meta)
        exact_force_caps_available = exact_force_caps_available and frame_caps_available

        broken_mask = is_broken
        damage_positive_mask = damage > damage_threshold
        damaged_unbroken_mask = damage_positive_mask & (~broken_mask)

        if first_cohesive_frame is None and np.any(is_cohesive):
            first_cohesive_frame = frame
            first_cohesive_time = time_value

        if first_broken_frame is None and np.any(broken_mask):
            first_broken_frame = frame
            first_broken_time = time_value

        crack_area_proxy = float(np.nansum(area[broken_mask])) if area.size else 0.0
        process_zone_area_proxy = float(np.nansum(area[damaged_unbroken_mask])) if area.size else 0.0

        damaged_body_ids = set(
            np.asarray(
                np.concatenate((bodyA_ids[damaged_unbroken_mask], bodyB_ids[damaged_unbroken_mask])),
                dtype=int,
            ).tolist()
        )
        broken_body_ids = set(np.asarray(np.concatenate((bodyA_ids[broken_mask], bodyB_ids[broken_mask])), dtype=int).tolist())

        peak_adjacent_to_damaged = int(peak_body in damaged_body_ids) if peak_body >= 0 else 0
        peak_adjacent_to_broken = int(peak_body in broken_body_ids) if peak_body >= 0 else 0

        L0 = np.linalg.norm(rest, axis=1) if rest.size else np.zeros((0,), dtype=float)
        strain_length = L0
        use_area_strain_length = bool(l_panel_mode and L0.size and not np.any(L0 > 1.0e-9) and area.size)
        if use_area_strain_length:
            strain_length = np.sqrt(np.clip(area, a_min=0.0, a_max=None))
            if bond_strain_length_note is None and np.isfinite(strain_length).any():
                bond_strain_length_note = (
                    f"For `{benchmark_name}`, exported bond `rest` vectors are degenerate, "
                    "so `eps_n` and `gamma_*` were normalized by `sqrt(area)` from bond metadata."
                )
        safe_strain_length = np.where(strain_length > np.finfo(float).eps, strain_length, np.nan)
        if use_area_strain_length:
            delta_n = C[:, 0] if C.size else np.zeros((0,), dtype=float)
        else:
            normal_sign = (
                np.where(np.abs(rest[:, 0]) > np.finfo(float).eps, np.sign(rest[:, 0]), 1.0)
                if rest.size
                else np.zeros((0,), dtype=float)
            )
            delta_n = normal_sign * C[:, 0] if C.size else np.zeros((0,), dtype=float)
        eps_n = delta_n / safe_strain_length if C.size else np.zeros((0,), dtype=float)
        gamma_t1 = C[:, 1] / safe_strain_length if C.size else np.zeros((0,), dtype=float)
        gamma_t2 = C[:, 2] / safe_strain_length if C.size else np.zeros((0,), dtype=float)
        gamma_eq = np.sqrt(np.square(gamma_t1) + np.square(gamma_t2)) if C.size else np.zeros((0,), dtype=float)

        if bond_ids.size and frame_caps_available:
            row_force = np.clip(penalty_k * C, f_min, f_max)
            sigma_n = row_force[:, 0] / area
            tau_t1 = row_force[:, 1] / area
            tau_t2 = row_force[:, 2] / area
            tau_eq = np.sqrt(np.square(tau_t1) + np.square(tau_t2))
        else:
            row_force = np.full((bond_ids.shape[0], 3), math.nan, dtype=float)
            sigma_n = np.full((bond_ids.shape[0],), math.nan, dtype=float)
            tau_t1 = np.full((bond_ids.shape[0],), math.nan, dtype=float)
            tau_t2 = np.full((bond_ids.shape[0],), math.nan, dtype=float)
            tau_eq = np.full((bond_ids.shape[0],), math.nan, dtype=float)

        bond_summary_rows.append(
            {
                "frame": int(frame),
                "step": int(step_value),
                "time": float(time_value),
                "bond_force_caps_available": int(frame_caps_available),
                "num_exported_bonds": int(bond_ids.shape[0]),
                "max_eps_n": _finite_max(eps_n),
                "p95_eps_n": _finite_percentile(eps_n, 95.0),
                "max_gamma_eq": _finite_max(gamma_eq),
                "p95_gamma_eq": _finite_percentile(gamma_eq, 95.0),
                "max_sigma_n": _finite_max(sigma_n),
                "p95_sigma_n": _finite_percentile(sigma_n, 95.0),
                "max_tau_eq": _finite_max(tau_eq),
                "p95_tau_eq": _finite_percentile(tau_eq, 95.0),
                "high_damage_bond_count": int(np.count_nonzero(damage >= 0.9)),
            }
        )

        if load_ids:
            pos_by_id = {int(body_id): body_pos[i] for i, body_id in enumerate(body_ids)}
            if initial_load_positions is None:
                if all(load_id in pos_by_id for load_id in load_ids):
                    initial_load_positions = {load_id: pos_by_id[load_id].copy() for load_id in load_ids}
                else:
                    load_displacement_available = False
            if load_displacement_available and initial_load_positions is not None:
                if all(load_id in pos_by_id for load_id in load_ids):
                    disp = np.mean([pos_by_id[load_id] - initial_load_positions[load_id] for load_id in load_ids], axis=0)
                    load_disp_x, load_disp_y, load_disp_z = map(float, disp)
                    load_disp_mag = float(np.linalg.norm(disp))
                    load_disp_axis = float(np.dot(disp, load_dir)) if load_dir is not None else math.nan
                else:
                    load_displacement_available = False
                    load_disp_x = load_disp_y = load_disp_z = load_disp_mag = load_disp_axis = math.nan
            else:
                load_disp_x = load_disp_y = load_disp_z = load_disp_mag = load_disp_axis = math.nan
        else:
            load_disp_x = load_disp_y = load_disp_z = load_disp_mag = load_disp_axis = math.nan

        row = {
            "frame": int(frame),
            "step": int(step_value),
            "time": float(time_value),
            "kinetic": _safe_float(energy_row.get("kinetic")),
            "bond_potential": _safe_float(energy_row.get("bond_potential")),
            "contact_potential": _safe_float(energy_row.get("contact_potential")),
            "fracture_work": _safe_float(energy_row.get("fracture_work")),
            "viscous_work": _safe_float(energy_row.get("viscous_work")),
            "mech_energy": _safe_float(energy_row.get("mech_energy")),
            "accounted_energy": _safe_float(energy_row.get("accounted_energy")),
            "broken_bond_count": int(np.count_nonzero(broken_mask)),
            "damaged_bond_count": int(np.count_nonzero(damaged_unbroken_mask)),
            "damage_positive_bond_count": int(np.count_nonzero(damage_positive_mask)),
            "crack_area_proxy": float(crack_area_proxy),
            "process_zone_area_proxy": float(process_zone_area_proxy),
            "peak_stress_proxy": float(peak_frame_proxy),
            "peak_stress_proxy_body_id": int(peak_body),
            "peak_stress_proxy_x": float(peak_body_position[0]),
            "peak_stress_proxy_y": float(peak_body_position[1]),
            "peak_stress_proxy_z": float(peak_body_position[2]),
            "peak_hotspot_adjacent_to_damaged": int(peak_adjacent_to_damaged),
            "peak_hotspot_adjacent_to_broken": int(peak_adjacent_to_broken),
            "stress_threshold": float(stress_threshold) if stress_threshold is not None else math.nan,
            "stress_exceedance_count": int(stress_exceedance_count),
            "stress_exceedance_fraction": float(stress_exceedance_fraction),
            "iters_used": _safe_int(metric_row.get("iters_used")),
            "max_violation": _safe_float(metric_row.get("max_violation")),
            "active_body_count": _safe_int(metric_row.get("active_body_count"), int(body_ids.shape[0])),
            "active_bond_count": _safe_int(metric_row.get("active_bond_count"), int(bond_ids.shape[0])),
            "exported_body_count": _safe_int(metric_row.get("exported_body_count"), int(body_ids.shape[0])),
            "exported_bond_count": _safe_int(metric_row.get("exported_bond_count"), int(bond_ids.shape[0])),
            "contact_count": _safe_int(metric_row.get("contact_count")),
            "load_displacement_x": float(load_disp_x),
            "load_displacement_y": float(load_disp_y),
            "load_displacement_z": float(load_disp_z),
            "load_displacement_magnitude": float(load_disp_mag),
            "load_displacement_along_loading_axis": float(load_disp_axis),
        }
        time_history_rows.append(row)

        current_bond_dump_rows = _bond_dump_rows(
            frame=frame,
            step=step_value,
            time_value=time_value,
            bond_ids=bond_ids,
            bodyA_ids=bodyA_ids,
            bodyB_ids=bodyB_ids,
            damage=damage,
            is_broken=is_broken.astype(np.uint8),
            is_cohesive=is_cohesive.astype(np.uint8),
            area=area,
            L0=L0,
            eps_n=eps_n,
            gamma_t1=gamma_t1,
            gamma_t2=gamma_t2,
            gamma_eq=gamma_eq,
            row_force=row_force,
            sigma_n=sigma_n,
            tau_t1=tau_t1,
            tau_t2=tau_t2,
            tau_eq=tau_eq,
        )
        final_bond_dump_rows = current_bond_dump_rows
        if bond_dump_mode == "all":
            all_bond_dump_rows.extend(current_bond_dump_rows)

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(analysis_dir / "time_history.csv", time_history_rows)
    _write_csv(analysis_dir / "bond_summary.csv", bond_summary_rows)
    if bond_dump_mode == "final":
        _write_csv(analysis_dir / "bond_dump_final.csv", final_bond_dump_rows)
    elif bond_dump_mode == "all":
        _write_csv(analysis_dir / "bond_dump_all.csv", all_bond_dump_rows)

    final_row = time_history_rows[-1] if time_history_rows else {}
    material_metadata = _extract_material_metadata(metadata)
    summary_row = {
        "run_id": run_dir.name,
        "benchmark_name": benchmark_name,
        "dt": dt_value,
        "iterations": _safe_int(metadata.get("iterations"), _safe_int(manifest.get("iterations"))),
        "material_parameters_json": json.dumps(material_metadata, sort_keys=True),
        "first_cohesive_onset_frame": first_cohesive_frame if first_cohesive_frame is not None else "",
        "first_cohesive_onset_time": first_cohesive_time if first_cohesive_time is not None else "",
        "first_broken_bond_frame": first_broken_frame if first_broken_frame is not None else "",
        "first_broken_bond_time": first_broken_time if first_broken_time is not None else "",
        "final_broken_bond_count": final_row.get("broken_bond_count", ""),
        "final_crack_area_proxy": final_row.get("crack_area_proxy", ""),
        "peak_stress_proxy": peak_stress_proxy if math.isfinite(peak_stress_proxy) else "",
        "peak_stress_proxy_frame": peak_stress_frame if peak_stress_frame is not None else "",
        "peak_stress_proxy_time": peak_stress_time if peak_stress_time is not None else "",
        "peak_stress_proxy_body_id": peak_stress_body_id if peak_stress_body_id is not None else "",
        "peak_stress_proxy_x": float(peak_stress_pos[0]) if np.isfinite(peak_stress_pos[0]) else "",
        "peak_stress_proxy_y": float(peak_stress_pos[1]) if np.isfinite(peak_stress_pos[1]) else "",
        "peak_stress_proxy_z": float(peak_stress_pos[2]) if np.isfinite(peak_stress_pos[2]) else "",
        "final_fracture_work": final_row.get("fracture_work", ""),
        "final_bond_potential": final_row.get("bond_potential", ""),
        "final_kinetic_energy": final_row.get("kinetic", ""),
        "load_displacement_reconstructable": int(load_displacement_available),
        "exact_bond_force_caps_available": int(exact_force_caps_available),
        "reaction_force_reconstructable": 0,
        "engineering_stress_reconstructable": 0,
        "continuum_stress_field_reconstructable": 0,
        "contact_stress_reconstructable": 0,
    }
    _write_csv(analysis_dir / "run_summary.csv", [summary_row])

    _write_thesis_guide(
        analysis_dir / "thesis_quantity_guide.md",
        load_displacement_available=load_displacement_available,
        exact_force_caps_available=exact_force_caps_available,
        bond_strain_length_note=bond_strain_length_note,
    )

    return {
        "analysis_dir": analysis_dir,
        "summary_row": summary_row,
        "time_history_rows": time_history_rows,
        "bond_summary_rows": bond_summary_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process AVBD fracture exports into thesis-ready summaries.")
    parser.add_argument("run_dir", nargs="+", help="One or more run directories, e.g. output/projectile_impact/run_001")
    parser.add_argument("--damage-threshold", type=float, default=1.0e-12, help="Threshold for treating a bond as damaged.")
    parser.add_argument(
        "--stress-threshold",
        type=float,
        default=None,
        help="Optional stress-proxy threshold. Defaults to refine_stress_threshold from metadata when available.",
    )
    parser.add_argument(
        "--bond-dump",
        choices=["none", "final", "all"],
        default="none",
        help="Optional detailed bond dump mode. Default is no full dump.",
    )
    args = parser.parse_args()

    for run_dir_str in args.run_dir:
        run_dir = Path(run_dir_str).expanduser().resolve()
        result = analyze_run(
            run_dir,
            damage_threshold=float(args.damage_threshold),
            stress_threshold=args.stress_threshold,
            bond_dump_mode=args.bond_dump,
        )
        print(f"[postprocess] Wrote analysis outputs to {result['analysis_dir']}")


if __name__ == "__main__":
    main()
