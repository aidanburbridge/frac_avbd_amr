from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import geometry.voxelizer as vox


DEFAULT_BENCHMARKS = [
    "L_bar",
    "miehe_shear",
    "ISO_20753",
    "three_point_fixture",
    "projectile_impact",
]


@dataclass
class AuditRow:
    benchmark_module: str
    benchmark_name: str
    part_name: str
    stl_path: str
    body_count: int
    assembly_id: int | str
    raw_extent_x: float
    raw_extent_y: float
    raw_extent_z: float
    candidate_scale_x_to_m: float
    candidate_scale_y_to_m: float
    candidate_scale_z_to_m: float
    applied_scale_to_m: float
    actual_extent_x_m: float
    actual_extent_y_m: float
    actual_extent_z_m: float
    actual_extent_min_m: float
    actual_extent_mid_m: float
    actual_extent_max_m: float
    nominal_extent_x_m: float
    nominal_extent_y_m: float
    nominal_extent_z_m: float
    nominal_extent_min_m: float
    nominal_extent_mid_m: float
    nominal_extent_max_m: float
    compared_dimensions: str
    max_abs_error_m: float
    max_rel_error_pct: float
    note: str


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sorted_triplet(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    out = np.full((3,), math.nan, dtype=float)
    finite = np.sort(arr[np.isfinite(arr)])
    count = min(finite.size, 3)
    if count:
        out[:count] = finite[:count]
    return out


def _extents_from_bodies(bodies: list[object]) -> np.ndarray:
    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = -mins
    for body in bodies:
        aabb = body.get_aabb()
        mins[0] = min(mins[0], aabb.min_x)
        maxs[0] = max(maxs[0], aabb.max_x)
        mins[1] = min(mins[1], aabb.min_y)
        maxs[1] = max(maxs[1], aabb.max_y)
        mins[2] = min(mins[2], aabb.min_z)
        maxs[2] = max(maxs[2], aabb.max_z)
    return maxs - mins


def _group_bodies_by_assembly(bodies: list[object]) -> list[dict[str, Any]]:
    grouped: dict[int, list[object]] = {}
    for body in bodies:
        assembly_id = int(getattr(body, "assembly_id", -1))
        grouped.setdefault(assembly_id, []).append(body)

    out: list[dict[str, Any]] = []
    for assembly_id, group_bodies in grouped.items():
        extents = _extents_from_bodies(group_bodies)
        out.append(
            {
                "assembly_id": assembly_id,
                "bodies": group_bodies,
                "body_count": len(group_bodies),
                "extents": extents,
                "bbox_volume": float(np.prod(extents)),
            }
        )
    out.sort(key=lambda item: (-item["body_count"], -item["bbox_volume"], item["assembly_id"]))
    return out


def _load_mesh_extents(stl_path: str) -> np.ndarray:
    stlvox = vox.STLVoxelizer(stl_path, flood_fill=False)
    return np.asarray(stlvox.mesh.extents, dtype=float)


def _load_setup(module_name: str):
    module = importlib.import_module(f"tests.{module_name}")
    setup_fn = module.build_setup
    kwargs = {}
    if "sync_bodies" in inspect.signature(setup_fn).parameters:
        kwargs["sync_bodies"] = False
    setup = setup_fn(**kwargs)
    return module, setup


def _make_row(
    *,
    benchmark_module: str,
    benchmark_name: str,
    part_name: str,
    stl_path: str,
    body_count: int,
    assembly_id: int | str,
    raw_extents: np.ndarray,
    candidate_scales: np.ndarray,
    applied_scale: float,
    actual_extents: np.ndarray,
    nominal_extents_xyz: np.ndarray,
    error_pairs: list[tuple[float, float]],
    compared_dimensions: str,
    note: str,
) -> AuditRow:
    raw_extents = np.asarray(raw_extents, dtype=float).reshape(3)
    candidate_scales = np.asarray(candidate_scales, dtype=float).reshape(3)
    actual_extents = np.asarray(actual_extents, dtype=float).reshape(3)
    nominal_extents_xyz = np.asarray(nominal_extents_xyz, dtype=float).reshape(3)

    abs_errors: list[float] = []
    rel_errors: list[float] = []
    for actual_value, nominal_value in error_pairs:
        if math.isfinite(actual_value) and math.isfinite(nominal_value):
            abs_err = abs(actual_value - nominal_value)
            abs_errors.append(abs_err)
            if nominal_value != 0.0:
                rel_errors.append(100.0 * abs_err / abs(nominal_value))

    actual_sorted = _sorted_triplet(actual_extents)
    nominal_sorted = _sorted_triplet(nominal_extents_xyz)

    return AuditRow(
        benchmark_module=benchmark_module,
        benchmark_name=benchmark_name,
        part_name=part_name,
        stl_path=stl_path,
        body_count=int(body_count),
        assembly_id=assembly_id,
        raw_extent_x=float(raw_extents[0]),
        raw_extent_y=float(raw_extents[1]),
        raw_extent_z=float(raw_extents[2]),
        candidate_scale_x_to_m=float(candidate_scales[0]),
        candidate_scale_y_to_m=float(candidate_scales[1]),
        candidate_scale_z_to_m=float(candidate_scales[2]),
        applied_scale_to_m=float(applied_scale),
        actual_extent_x_m=float(actual_extents[0]),
        actual_extent_y_m=float(actual_extents[1]),
        actual_extent_z_m=float(actual_extents[2]),
        actual_extent_min_m=float(actual_sorted[0]),
        actual_extent_mid_m=float(actual_sorted[1]),
        actual_extent_max_m=float(actual_sorted[2]),
        nominal_extent_x_m=float(nominal_extents_xyz[0]),
        nominal_extent_y_m=float(nominal_extents_xyz[1]),
        nominal_extent_z_m=float(nominal_extents_xyz[2]),
        nominal_extent_min_m=float(nominal_sorted[0]),
        nominal_extent_mid_m=float(nominal_sorted[1]),
        nominal_extent_max_m=float(nominal_sorted[2]),
        compared_dimensions=compared_dimensions,
        max_abs_error_m=max(abs_errors) if abs_errors else math.nan,
        max_rel_error_pct=max(rel_errors) if rel_errors else math.nan,
        note=note,
    )


def _benchmark_name(setup, module_name: str) -> str:
    metadata = dict(getattr(setup, "metadata", None) or {})
    return str(metadata.get("benchmark_name") or module_name)


def _applied_scale(setup, default: float = math.nan) -> float:
    metadata = dict(getattr(setup, "metadata", None) or {})
    return _safe_float(metadata.get("raw_length_scale_to_m"), default)


def _audit_l_bar(module, setup) -> list[AuditRow]:
    raw_extents = _load_mesh_extents(module.STL_PATH)
    actual_extents = _extents_from_bodies(setup.bodies)
    candidate_scales = np.array(
        [
            module.OUTER_DIM / raw_extents[0] if raw_extents[0] > 0.0 else math.nan,
            math.nan,
            module.OUTER_DIM / raw_extents[2] if raw_extents[2] > 0.0 else math.nan,
        ],
        dtype=float,
    )
    nominal = np.array([module.OUTER_DIM, math.nan, module.OUTER_DIM], dtype=float)
    errors = [
        (float(actual_extents[0]), float(module.OUTER_DIM)),
        (float(actual_extents[2]), float(module.OUTER_DIM)),
    ]
    return [
        _make_row(
            benchmark_module="L_bar",
            benchmark_name=_benchmark_name(setup, "L_bar"),
            part_name="panel",
            stl_path=module.STL_PATH,
            body_count=len(setup.bodies),
            assembly_id="all",
            raw_extents=raw_extents,
            candidate_scales=candidate_scales,
            applied_scale=_applied_scale(setup, float(np.nanmedian(candidate_scales))),
            actual_extents=actual_extents,
            nominal_extents_xyz=nominal,
            error_pairs=errors,
            compared_dimensions="x,z",
            note="Outer panel x/z extents audited against the 500 mm target. Plate thickness is not explicitly calibrated by this benchmark script.",
        )
    ]


def _audit_miehe(module, setup) -> list[AuditRow]:
    raw_extents = _load_mesh_extents(module.STL_PATH)
    actual_extents = _extents_from_bodies(setup.bodies)
    candidate_scales = np.array(
        [
            module.SPECIMEN_WIDTH / raw_extents[0] if raw_extents[0] > 0.0 else math.nan,
            module.SPECIMEN_THICKNESS / raw_extents[1] if raw_extents[1] > 0.0 else math.nan,
            module.SPECIMEN_HEIGHT / raw_extents[2] if raw_extents[2] > 0.0 else math.nan,
        ],
        dtype=float,
    )
    nominal = np.array(
        [
            module.SPECIMEN_WIDTH,
            module.SPECIMEN_THICKNESS,
            module.SPECIMEN_HEIGHT,
        ],
        dtype=float,
    )
    errors = [(float(actual_extents[i]), float(nominal[i])) for i in range(3)]
    return [
        _make_row(
            benchmark_module="miehe_shear",
            benchmark_name=_benchmark_name(setup, "miehe_shear"),
            part_name="specimen",
            stl_path=module.STL_PATH,
            body_count=len(setup.bodies),
            assembly_id="all",
            raw_extents=raw_extents,
            candidate_scales=candidate_scales,
            applied_scale=_applied_scale(setup, float(np.nanmedian(candidate_scales))),
            actual_extents=actual_extents,
            nominal_extents_xyz=nominal,
            error_pairs=errors,
            compared_dimensions="x,y,z",
            note="Direct width/thickness/height audit against the nominal Miehe specimen dimensions.",
        )
    ]


def _audit_iso_like(module_name: str, part_name: str, stl_path: str, target_length: float, setup) -> list[AuditRow]:
    raw_extents = _load_mesh_extents(stl_path)
    actual_extents = _extents_from_bodies(setup.bodies)
    scale_factor = target_length / float(np.max(raw_extents))
    candidate_scales = np.array([math.nan, math.nan, scale_factor], dtype=float)
    nominal = np.array([math.nan, math.nan, target_length], dtype=float)
    errors = [(float(actual_extents[2]), float(target_length))]
    return [
        _make_row(
            benchmark_module=module_name,
            benchmark_name=_benchmark_name(setup, module_name),
            part_name=part_name,
            stl_path=stl_path,
            body_count=len(setup.bodies),
            assembly_id="all",
            raw_extents=raw_extents,
            candidate_scales=candidate_scales,
            applied_scale=_applied_scale(setup, scale_factor),
            actual_extents=actual_extents,
            nominal_extents_xyz=nominal,
            error_pairs=errors,
            compared_dimensions="z",
            note="Only the longest specimen dimension is explicitly calibrated here. The remaining specimen proportions are inherited from the STL.",
        )
    ]


def _audit_three_point(module, setup) -> list[AuditRow]:
    groups = _group_bodies_by_assembly(setup.bodies)
    beam_group = groups[0]
    roller_groups = groups[1:]
    roller_extents = np.median(np.vstack([group["extents"] for group in roller_groups]), axis=0)

    beam_raw_extents = _load_mesh_extents(module.BEAM_STL)
    roller_raw_extents = _load_mesh_extents(module.ROLLER_STL)

    beam_nominal = np.array([module.BEAM_LENGTH, module.BEAM_WIDTH, module.BEAM_HEIGHT], dtype=float)
    beam_errors = [
        (float(beam_group["extents"][0]), float(module.BEAM_LENGTH)),
        *list(
            zip(
                np.sort(np.asarray(beam_group["extents"][1:], dtype=float)),
                np.sort(np.asarray([module.BEAM_WIDTH, module.BEAM_HEIGHT], dtype=float)),
            )
        ),
    ]

    roller_nominal = np.array([module.ROLLER_DIAMETER, module.ROLLER_LENGTH, module.ROLLER_DIAMETER], dtype=float)
    roller_errors = [
        (float(roller_extents[1]), float(module.ROLLER_LENGTH)),
        *list(
            zip(
                np.sort(np.asarray([roller_extents[0], roller_extents[2]], dtype=float)),
                np.sort(np.asarray([module.ROLLER_DIAMETER, module.ROLLER_DIAMETER], dtype=float)),
            )
        ),
    ]

    return [
        _make_row(
            benchmark_module="three_point_fixture",
            benchmark_name=_benchmark_name(setup, "three_point_fixture"),
            part_name="beam",
            stl_path=module.BEAM_STL,
            body_count=int(beam_group["body_count"]),
            assembly_id=int(beam_group["assembly_id"]),
            raw_extents=beam_raw_extents,
            candidate_scales=np.array([math.nan, math.nan, module.BEAM_LENGTH / float(np.max(beam_raw_extents))], dtype=float),
            applied_scale=float(module.BEAM_LENGTH / float(np.max(beam_raw_extents))),
            actual_extents=np.asarray(beam_group["extents"], dtype=float),
            nominal_extents_xyz=beam_nominal,
            error_pairs=beam_errors,
            compared_dimensions="x + sorted(y,z)",
            note="Beam length is audited on x after alignment. The y/z cross-section is audited as an unordered width/height pair.",
        ),
        _make_row(
            benchmark_module="three_point_fixture",
            benchmark_name=_benchmark_name(setup, "three_point_fixture"),
            part_name="roller",
            stl_path=module.ROLLER_STL,
            body_count=int(sum(group["body_count"] for group in roller_groups)),
            assembly_id="median_of_nonbeam_groups",
            raw_extents=roller_raw_extents,
            candidate_scales=np.array([math.nan, math.nan, module.ROLLER_LENGTH / float(np.max(roller_raw_extents))], dtype=float),
            applied_scale=float(module.ROLLER_LENGTH / float(np.max(roller_raw_extents))),
            actual_extents=np.asarray(roller_extents, dtype=float),
            nominal_extents_xyz=roller_nominal,
            error_pairs=roller_errors,
            compared_dimensions="y + sorted(x,z)",
            note="Roller length is audited on y after alignment. The x/z cross-section is audited as an unordered diameter pair.",
        ),
    ]


def _audit_projectile(module, setup) -> list[AuditRow]:
    groups = _group_bodies_by_assembly(setup.bodies)
    wall_group = groups[0]
    projectile_group = groups[1]

    wall_raw_extents = _load_mesh_extents(module.WALL_STL_PATH)
    projectile_raw_extents = _load_mesh_extents(module.PROJECTILE_STL_PATH)

    wall_nominal_sorted = np.sort(np.asarray([module.WALL_WIDTH, module.WALL_HEIGHT, module.WALL_THICKNESS], dtype=float))
    wall_actual_sorted = np.sort(np.asarray(wall_group["extents"], dtype=float))
    wall_errors = list(zip(wall_actual_sorted, wall_nominal_sorted))

    projectile_nominal = np.asarray(
        [module.PROJECTILE_DIAMETER, module.PROJECTILE_DIAMETER, module.PROJECTILE_DIAMETER],
        dtype=float,
    )
    projectile_errors = list(
        zip(
            np.sort(np.asarray(projectile_group["extents"], dtype=float)),
            np.sort(projectile_nominal),
        )
    )

    return [
        _make_row(
            benchmark_module="projectile_impact",
            benchmark_name=_benchmark_name(setup, "projectile_impact"),
            part_name="wall",
            stl_path=module.WALL_STL_PATH,
            body_count=int(wall_group["body_count"]),
            assembly_id=int(wall_group["assembly_id"]),
            raw_extents=wall_raw_extents,
            candidate_scales=np.array(
                [
                    module.WALL_THICKNESS / float(np.min(wall_raw_extents)),
                    min(module.WALL_WIDTH, module.WALL_HEIGHT) / float(np.sort(wall_raw_extents)[1]),
                    max(module.WALL_WIDTH, module.WALL_HEIGHT) / float(np.max(wall_raw_extents)),
                ],
                dtype=float,
            ),
            applied_scale=_safe_float(getattr(setup, "metadata", {}).get("wall_scale")),
            actual_extents=np.asarray(wall_group["extents"], dtype=float),
            nominal_extents_xyz=np.asarray(wall_nominal_sorted, dtype=float),
            error_pairs=wall_errors,
            compared_dimensions="sorted(min,mid,max)",
            note="Wall extents are audited as an unordered triplet: thickness is the smallest span and the two larger spans are compared to the nominal in-plane dimensions.",
        ),
        _make_row(
            benchmark_module="projectile_impact",
            benchmark_name=_benchmark_name(setup, "projectile_impact"),
            part_name="projectile",
            stl_path=module.PROJECTILE_STL_PATH,
            body_count=int(projectile_group["body_count"]),
            assembly_id=int(projectile_group["assembly_id"]),
            raw_extents=projectile_raw_extents,
            candidate_scales=np.array(
                [module.PROJECTILE_DIAMETER / float(np.max(projectile_raw_extents))] * 3,
                dtype=float,
            ),
            applied_scale=_safe_float(getattr(setup, "metadata", {}).get("projectile_scale")),
            actual_extents=np.asarray(projectile_group["extents"], dtype=float),
            nominal_extents_xyz=projectile_nominal,
            error_pairs=projectile_errors,
            compared_dimensions="sorted(x,y,z)",
            note="Projectile extents are audited as an unordered triplet against the nominal sphere diameter.",
        ),
    ]


def _audit_rows_for(module_name: str) -> list[AuditRow]:
    module, setup = _load_setup(module_name)
    if module_name == "L_bar":
        return _audit_l_bar(module, setup)
    if module_name == "miehe_shear":
        return _audit_miehe(module, setup)
    if module_name == "ISO_20753":
        return _audit_iso_like("ISO_20753", "specimen", module.STL_PATH, module.LENGTH, setup)
    if module_name == "bond_26_test":
        return _audit_iso_like("bond_26_test", "specimen", module.STL_PATH, module.LENGTH, setup)
    if module_name == "three_point_fixture":
        return _audit_three_point(module, setup)
    if module_name == "projectile_impact":
        return _audit_projectile(module, setup)
    raise ValueError(f"Unsupported benchmark module '{module_name}'")


def _write_csv(path: Path, rows: list[AuditRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(AuditRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:.6g}"


def _print_rows(rows: list[AuditRow]) -> None:
    for row in rows:
        print(f"[{row.benchmark_name} / {row.part_name}]")
        print(
            "  raw STL extents: "
            f"x={_fmt(row.raw_extent_x)}, y={_fmt(row.raw_extent_y)}, z={_fmt(row.raw_extent_z)}"
        )
        print(
            "  final extents [m]: "
            f"x={_fmt(row.actual_extent_x_m)}, y={_fmt(row.actual_extent_y_m)}, z={_fmt(row.actual_extent_z_m)}"
        )
        print(
            "  applied scale to m: "
            f"{_fmt(row.applied_scale_to_m)}"
        )
        print(
            "  compared dimensions: "
            f"{row.compared_dimensions}"
        )
        print(
            "  max error: "
            f"{_fmt(row.max_abs_error_m)} m, {_fmt(row.max_rel_error_pct)} %"
        )
        print(f"  note: {row.note}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit benchmark STL scaling against nominal physical dimensions.")
    parser.add_argument(
        "benchmarks",
        nargs="*",
        default=DEFAULT_BENCHMARKS,
        help=f"Benchmark module names under tests/. Defaults to: {', '.join(DEFAULT_BENCHMARKS)}",
    )
    parser.add_argument(
        "--csv",
        default="output/benchmark_scale_audit.csv",
        help="CSV output path for Excel-friendly audit rows.",
    )
    args = parser.parse_args()

    rows: list[AuditRow] = []
    for module_name in args.benchmarks:
        rows.extend(_audit_rows_for(module_name))

    _print_rows(rows)
    csv_path = Path(args.csv).expanduser().resolve()
    _write_csv(csv_path, rows)
    print(f"[audit] Wrote CSV to {csv_path}")


if __name__ == "__main__":
    main()
