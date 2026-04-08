from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.plot_results import (
    _benchmark_slug,
    _canonical_benchmark_name,
    _displacement_unit_label,
    _read_metadata,
    _read_summary_row,
    _read_time_history,
    _select_progress_axis,
    _unit_label,
    generate_plots,
)
from util.postprocess_results import analyze_run


def _discover_run_dirs(root_dir: Path, analysis_dir_name: str, recursive: bool) -> list[Path]:
    run_dirs: set[Path] = set()

    direct_history = root_dir / analysis_dir_name / "time_history.csv"
    if direct_history.exists():
        run_dirs.add(root_dir.resolve())

    direct_raw = root_dir / "raw"
    if any(direct_raw.glob("frame_*.bin")):
        run_dirs.add(root_dir.resolve())

    direct_vtk = root_dir / "vtk"
    if any(direct_vtk.glob("voxels_*.vtu")):
        run_dirs.add(root_dir.resolve())

    if recursive:
        history_paths = root_dir.rglob(f"{analysis_dir_name}/time_history.csv")
        raw_paths = root_dir.rglob("raw/frame_*.bin")
        vtk_paths = root_dir.rglob("vtk/voxels_*.vtu")
    else:
        history_paths = root_dir.glob(f"*/{analysis_dir_name}/time_history.csv")
        raw_paths = root_dir.glob("*/raw/frame_*.bin")
        vtk_paths = root_dir.glob("*/vtk/voxels_*.vtu")

    run_dirs.update(path.parent.parent.resolve() for path in history_paths)
    run_dirs.update(path.parent.parent.resolve() for path in raw_paths)
    run_dirs.update(path.parent.parent.resolve() for path in vtk_paths)

    return sorted(run_dirs)


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    numeric = _safe_float(value)
    if math.isfinite(numeric):
        magnitude = abs(numeric)
        if magnitude >= 1.0e4 or (magnitude > 0.0 and magnitude < 1.0e-3):
            return f"{numeric:.3e}"
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.6g}"
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _curve_label(run_dir: Path, resolved_config: dict[str, Any]) -> str:
    swept = resolved_config.get("swept_parameters")
    if isinstance(swept, dict) and swept:
        return ", ".join(f"{key}={_format_value(value)}" for key, value in swept.items())

    params = resolved_config.get("parameters")
    if isinstance(params, dict) and params:
        preferred = [
            key
            for key in ("load_velocity", "dt_physics", "youngs_modulus", "target_displacement", "steps")
            if key in params
        ]
        if preferred:
            return ", ".join(f"{key}={_format_value(params[key])}" for key in preferred)

    return run_dir.name


def _collect_batch_records(
    run_dirs: list[Path],
    *,
    analysis_dir_name: str,
    x_axis: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        analysis_dir = run_dir / analysis_dir_name
        history_path = analysis_dir / "time_history.csv"
        if not history_path.exists():
            continue

        data = _read_time_history(history_path)
        if not data:
            continue

        metadata = _read_metadata(run_dir / "meta_data.json")
        summary = _read_summary_row(analysis_dir / "run_summary.csv")
        resolved_config = _read_json(run_dir / "resolved_config.json")

        benchmark_name = _canonical_benchmark_name(
            summary.get("benchmark_name") or metadata.get("benchmark_name"),
            run_dir,
        )
        displacement_unit_label = _displacement_unit_label(summary, metadata)
        x, x_label, title_suffix, axis_slug = _select_progress_axis(
            data,
            x_axis_mode=x_axis,
            displacement_unit_label=displacement_unit_label,
        )

        records.append(
            {
                "run_dir": run_dir,
                "data": data,
                "metadata": metadata,
                "summary": summary,
                "resolved_config": resolved_config,
                "label": _curve_label(run_dir, resolved_config),
                "benchmark_name": benchmark_name,
                "benchmark_slug": _benchmark_slug(benchmark_name),
                "x": x,
                "x_label": x_label,
                "title_suffix": title_suffix,
                "axis_slug": axis_slug,
                "displacement_unit_label": displacement_unit_label,
                "area_unit_label": _unit_label(
                    summary,
                    metadata,
                    summary_key="area_unit_label",
                    metadata_key="area_unit_label",
                    default="exported coordinate units^2",
                ),
                "stress_unit_label": _unit_label(
                    summary,
                    metadata,
                    summary_key="stress_unit_label",
                    metadata_key="stress_unit_label",
                    default="stress units of the solver setup",
                ),
            }
        )

    return records


def _save_plot(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.1)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.12)
    fig.clf()


def _plot_overlay(
    records: list[dict[str, Any]],
    *,
    y_key: str,
    y_label: str,
    output_path: Path,
    title: str,
) -> bool:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plotted = 0

    for record in records:
        y = record["data"].get(y_key)
        if y is None:
            continue
        finite_mask = np.isfinite(record["x"]) & np.isfinite(y)
        if not np.any(finite_mask):
            continue
        ax.plot(record["x"][finite_mask], y[finite_mask], linewidth=1.8, label=record["label"])
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False

    ax.set_xlabel(records[0]["x_label"])
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(True, alpha=0.3)
    if plotted <= 14:
        ax.legend(fontsize=8)
    _save_plot(fig, output_path)
    plt.close(fig)
    return True


def _plot_parameter_response(
    records: list[dict[str, Any]],
    *,
    parameter_key: str,
    y_series: list[tuple[str, list[float], str]],
    output_path: Path,
    title: str,
) -> bool:
    import matplotlib.pyplot as plt

    x_values = np.asarray(
        [_safe_float(record["resolved_config"].get("swept_parameters", {}).get(parameter_key)) for record in records],
        dtype=float,
    )
    finite_x = np.isfinite(x_values)
    if not np.any(finite_x):
        return False

    order = np.argsort(x_values[finite_x])
    x_sorted = x_values[finite_x][order]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plotted = 0
    for label, y_values_raw, _y_label in y_series:
        y_values = np.asarray(y_values_raw, dtype=float)[finite_x][order]
        finite_mask = np.isfinite(x_sorted) & np.isfinite(y_values)
        if not np.any(finite_mask):
            continue
        ax.plot(x_sorted[finite_mask], y_values[finite_mask], marker="o", linewidth=1.6, label=label)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False

    ax.set_xlabel(parameter_key)
    ax.set_ylabel(y_series[0][2] if len(y_series) == 1 else "Response")
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save_plot(fig, output_path)
    plt.close(fig)
    return True


def _generate_batch_comparison_plots(
    root_dir: Path,
    *,
    run_dirs: list[Path],
    analysis_dir_name: str,
    x_axis: str,
) -> Path | None:
    records = _collect_batch_records(run_dirs, analysis_dir_name=analysis_dir_name, x_axis=x_axis)
    if len(records) < 2:
        return None

    output_dir = root_dir / "batch_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_name = records[0]["benchmark_name"]
    benchmark_slug = records[0]["benchmark_slug"]
    axis_slug = records[0]["axis_slug"]
    title_suffix = records[0]["title_suffix"]

    _plot_overlay(
        records,
        y_key="crack_area_proxy",
        y_label=f"Crack area proxy [{records[0]['area_unit_label']}]",
        output_path=output_dir / f"{benchmark_slug}_batch_crack_area_proxy_vs_{axis_slug}.png",
        title=f"{benchmark_name}: Crack Area Proxy vs {title_suffix}",
    )
    _plot_overlay(
        records,
        y_key="broken_bond_count",
        y_label="Broken bond count",
        output_path=output_dir / f"{benchmark_slug}_batch_broken_bond_count_vs_{axis_slug}.png",
        title=f"{benchmark_name}: Broken Bond Count vs {title_suffix}",
    )
    _plot_overlay(
        records,
        y_key="peak_mixed_mode_traction_utilization",
        y_label="Peak bond traction-cap utilization",
        output_path=output_dir / f"{benchmark_slug}_batch_mixed_mode_traction_utilization_vs_{axis_slug}.png",
        title=f"{benchmark_name}: Peak Bond Traction-Cap Utilization vs {title_suffix}",
    )
    _plot_overlay(
        records,
        y_key="peak_stress_proxy",
        y_label=f"Max principal tensile stress proxy [{records[0]['stress_unit_label']}]",
        output_path=output_dir / f"{benchmark_slug}_batch_peak_stress_proxy_vs_{axis_slug}.png",
        title=f"{benchmark_name}: Peak Stress Proxy vs {title_suffix}",
    )
    _plot_overlay(
        records,
        y_key="fracture_work",
        y_label="Fracture work",
        output_path=output_dir / f"{benchmark_slug}_batch_fracture_work_vs_{axis_slug}.png",
        title=f"{benchmark_name}: Fracture Work vs {title_suffix}",
    )

    swept_key_sets = [set(record["resolved_config"].get("swept_parameters", {}).keys()) for record in records]
    swept_keys = sorted(set.union(*swept_key_sets)) if swept_key_sets else []
    if len(swept_keys) == 1 and all(len(keys) <= 1 for keys in swept_key_sets):
        parameter_key = swept_keys[0]
        displacement_unit = records[0]["displacement_unit_label"]
        _plot_parameter_response(
            records,
            parameter_key=parameter_key,
            y_series=[
                (
                    "First damage onset",
                    [_safe_float(record["summary"].get("first_damage_onset_displacement")) for record in records],
                    f"Displacement [{displacement_unit}]",
                ),
                (
                    "First broken bond",
                    [_safe_float(record["summary"].get("first_broken_bond_displacement")) for record in records],
                    f"Displacement [{displacement_unit}]",
                ),
                (
                    "Peak crack growth",
                    [_safe_float(record["summary"].get("peak_crack_growth_stage_displacement")) for record in records],
                    f"Displacement [{displacement_unit}]",
                ),
                (
                    "Final plateau",
                    [_safe_float(record["summary"].get("final_plateau_displacement")) for record in records],
                    f"Displacement [{displacement_unit}]",
                ),
            ],
            output_path=output_dir / f"{benchmark_slug}_batch_event_displacements_vs_{parameter_key}.png",
            title=f"{benchmark_name}: Event Displacements vs {parameter_key}",
        )
        _plot_parameter_response(
            records,
            parameter_key=parameter_key,
            y_series=[
                (
                    "Final crack area proxy",
                    [_safe_float(record["summary"].get("final_crack_area_proxy")) for record in records],
                    f"Crack area proxy [{records[0]['area_unit_label']}]",
                ),
            ],
            output_path=output_dir / f"{benchmark_slug}_batch_final_crack_area_vs_{parameter_key}.png",
            title=f"{benchmark_name}: Final Crack Area Proxy vs {parameter_key}",
        )
        _plot_parameter_response(
            records,
            parameter_key=parameter_key,
            y_series=[
                (
                    "Final broken bond count",
                    [_safe_float(record["summary"].get("final_broken_bond_count")) for record in records],
                    "Broken bond count",
                ),
            ],
            output_path=output_dir / f"{benchmark_slug}_batch_final_broken_bond_count_vs_{parameter_key}.png",
            title=f"{benchmark_name}: Final Broken Bond Count vs {parameter_key}",
        )
        _plot_parameter_response(
            records,
            parameter_key=parameter_key,
            y_series=[
                (
                    "Peak stress proxy",
                    [_safe_float(record["summary"].get("peak_stress_proxy")) for record in records],
                    f"Stress proxy [{records[0]['stress_unit_label']}]",
                ),
            ],
            output_path=output_dir / f"{benchmark_slug}_batch_peak_stress_proxy_vs_{parameter_key}.png",
            title=f"{benchmark_name}: Peak Stress Proxy vs {parameter_key}",
        )
        _plot_parameter_response(
            records,
            parameter_key=parameter_key,
            y_series=[
                (
                    "Solve time",
                    [_safe_float(record["metadata"].get("solve_time_s")) for record in records],
                    "Solve time [s]",
                ),
            ],
            output_path=output_dir / f"{benchmark_slug}_batch_solve_time_vs_{parameter_key}.png",
            title=f"{benchmark_name}: Solve Time vs {parameter_key}",
        )

    print(f"[batch-plot] Wrote batch comparison figures to {output_dir}")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch wrapper around util.plot_results.generate_plots().")
    parser.add_argument("root_dir", help="Root directory containing one or more run directories.")
    parser.add_argument("--analysis-dir-name", default="analysis", help="Analysis subdirectory name.")
    parser.add_argument("--recursive", action="store_true", help="Search nested folders recursively.")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue after a per-run plotting failure (default: true).",
    )
    parser.add_argument(
        "--comparisons",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate batch-level comparison plots when multiple runs are found (default: true).",
    )
    parser.add_argument(
        "--x-axis",
        default="frame",
        choices=("frame", "step", "displacement", "time"),
        help="X-axis passed through to util.plot_results and the batch comparison overlays.",
    )
    args = parser.parse_args(argv)

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.exists():
        raise SystemExit(f"Root directory not found: {root_dir}")

    run_dirs = _discover_run_dirs(root_dir, args.analysis_dir_name, args.recursive)
    if not run_dirs:
        print(f"[SKIP] No valid run directories found under {root_dir}")
        print("Summary:")
        print("  Found: 0")
        print("  Plotted: 0")
        print("  Skipped: 1")
        print("  Failed: 0")
        return 0

    plotted = 0
    failed: list[tuple[Path, str]] = []
    successful_run_dirs: list[Path] = []
    comparison_failed = False

    for run_dir in run_dirs:
        try:
            analyze_run(
                run_dir,
                damage_threshold=1.0e-12,
                stress_threshold=None,
                bond_dump_mode="none",
                analysis_dir_name=args.analysis_dir_name,
            )
            generate_plots(run_dir, analysis_dir_name=args.analysis_dir_name, x_axis=args.x_axis)
            print(f"[OK] {run_dir}")
            plotted += 1
            successful_run_dirs.append(run_dir)
        except Exception as exc:
            print(f"[FAIL] {run_dir}: {exc}")
            failed.append((run_dir, str(exc)))
            if not args.continue_on_error:
                break

    if args.comparisons and len(successful_run_dirs) >= 2:
        try:
            _generate_batch_comparison_plots(
                root_dir,
                run_dirs=successful_run_dirs,
                analysis_dir_name=args.analysis_dir_name,
                x_axis=args.x_axis,
            )
        except Exception as exc:
            print(f"[FAIL] Batch comparison plots: {exc}")
            comparison_failed = True

    skipped = len(run_dirs) - plotted - len(failed)

    print("Summary:")
    print(f"  Found: {len(run_dirs)}")
    print(f"  Plotted: {plotted}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {len(failed)}")
    if failed:
        for run_dir, message in failed:
            print(f"  - {run_dir}: {message}")
    if comparison_failed:
        print("  - Batch comparison plot generation failed")

    return 1 if comparison_failed or (run_dirs and plotted == 0 and failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
