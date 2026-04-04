from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from util.plot_results import generate_plots
from util.postprocess_results import analyze_run


REPO_ROOT = Path(__file__).resolve().parents[1]

METRIC_SPECS: tuple[tuple[str, str], ...] = (
    ("final_crack_area_proxy", "Final crack area proxy"),
    ("final_broken_bond_count", "Final broken bond count"),
    ("final_process_zone_area_proxy", "Final process-zone area proxy"),
    ("peak_mixed_mode_traction_utilization", "Peak mixed-mode traction utilization"),
    ("final_peak_mixed_mode_traction_utilization", "Final peak mixed-mode traction utilization"),
    ("peak_stress_proxy", "Peak stress proxy"),
    ("final_peak_stress_proxy", "Final peak stress proxy"),
    ("final_fracture_work", "Final fracture work"),
    ("final_bond_potential", "Final bond potential"),
    ("final_kinetic_energy", "Final kinetic energy"),
    ("final_mech_energy", "Final mechanical energy"),
    ("final_accounted_energy", "Final accounted energy"),
    ("first_broken_bond_time", "First broken-bond time"),
    ("peak_crack_growth_stage_time", "Peak crack-growth-stage time"),
    ("final_plateau_time", "Final plateau time"),
    ("first_broken_bond_displacement", "First broken-bond displacement"),
    ("peak_crack_growth_stage_displacement", "Peak crack-growth-stage displacement"),
    ("final_plateau_displacement", "Final plateau displacement"),
    ("solve_time_s", "Solve time [s]"),
)


def _maybe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_summary_row(path: Path) -> dict[str, str]:
    rows = _read_csv_rows(path)
    return dict(rows[0]) if rows else {}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _slugify(value: str) -> str:
    chars: list[str] = []
    for ch in str(value).strip().lower():
        if ch.isalnum():
            chars.append(ch)
        elif chars and chars[-1] != "_":
            chars.append("_")
    return "".join(chars).strip("_") or "batch"


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    preferred = [
        "run_id",
        "run_name",
        "test",
        "benchmark_name",
        "batch_status",
        "analysis_available",
        "analysis_status",
        "analysis_error",
        "run_plots_status",
        "run_plots_error",
        "run_dir",
    ]
    all_keys = []
    seen: set[str] = set()
    for key in preferred:
        if any(key in row for row in rows):
            all_keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in all_keys})


def _save_plot(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.1)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.12)
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        fig.clf()


def _human_label(token: str) -> str:
    text = str(token or "").replace("_", " ").replace("-", " ").strip()
    if not text:
        return "Value"
    words: list[str] = []
    for word in text.split():
        upper = word.upper()
        if upper in {"ISO", "AVBD"}:
            words.append(upper)
        elif word.isupper() and len(word) <= 4:
            words.append(word)
        else:
            words.append(word.capitalize())
    return " ".join(words)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4g}"
    if isinstance(value, np.generic):
        return _format_value(value.item())
    return str(value)


def _iter_run_dirs(batch_dir: Path) -> list[Path]:
    run_dirs = [
        path
        for path in batch_dir.iterdir()
        if path.is_dir() and "_run_" in path.name
    ]
    return sorted(run_dirs, key=lambda path: path.name)


def _batch_status_map(batch_dir: Path) -> dict[str, dict[str, str]]:
    status_map: dict[str, dict[str, str]] = {}
    for row in _read_csv_rows(batch_dir / "batch_summary.csv"):
        output_dir = str(row.get("output_dir", "")).replace("\\", "/").strip()
        run_name = Path(output_dir).name if output_dir else ""
        if run_name:
            status_map[run_name] = row
    return status_map


def _ensure_analysis(
    run_dir: Path,
    *,
    recompute: bool,
    damage_threshold: float,
    stress_threshold: float | None,
    bond_dump_mode: str,
) -> tuple[str, str]:
    summary_path = run_dir / "analysis" / "run_summary.csv"
    if summary_path.exists() and not recompute:
        return "existing", ""

    analyze_run(
        run_dir,
        damage_threshold=damage_threshold,
        stress_threshold=stress_threshold,
        bond_dump_mode=bond_dump_mode,
    )
    return ("recomputed" if summary_path.exists() else "generated"), ""


def _collect_record(run_dir: Path, status_row: dict[str, str] | None) -> dict[str, Any]:
    resolved = _maybe_json(run_dir / "resolved_config.json")
    metadata = _maybe_json(run_dir / "meta_data.json")
    summary = _read_summary_row(run_dir / "analysis" / "run_summary.csv")

    swept = resolved.get("swept_parameters", {})
    if not isinstance(swept, dict):
        swept = {}
    params = resolved.get("parameters", {})
    if not isinstance(params, dict):
        params = {}

    test_name = str(resolved.get("test") or metadata.get("test_name") or run_dir.name.split("_run_")[0])
    benchmark_name = str(summary.get("benchmark_name") or metadata.get("benchmark_name") or test_name)

    row: dict[str, Any] = {
        "run_id": str(resolved.get("run_id") or run_dir.name),
        "run_name": run_dir.name,
        "test": test_name,
        "benchmark_name": benchmark_name,
        "batch_status": str((status_row or {}).get("status", "")).strip(),
        "analysis_available": int(bool(summary)),
        "run_dir": _relative_to_repo(run_dir),
    }

    for key, value in swept.items():
        row[f"swept_{key}"] = _csv_value(value)
    for key, value in params.items():
        row[f"param_{key}"] = _csv_value(value)
    for key in ("solve_time_s", "voxel_count", "bond_count", "dt", "iterations", "gravity", "friction", "total_steps"):
        if key in metadata:
            row[f"meta_{key}"] = _csv_value(metadata.get(key))
    for key, value in summary.items():
        row[f"summary_{key}"] = _csv_value(value)

    return {
        "run_dir": run_dir,
        "test": test_name,
        "benchmark_name": benchmark_name,
        "swept": swept,
        "params": params,
        "metadata": metadata,
        "summary": summary,
        "row": row,
    }


def _varying_sweep_keys(records: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record["swept"]:
            if key not in seen:
                keys.append(key)
                seen.add(key)

    varying: list[str] = []
    for key in keys:
        values = {
            json.dumps(record["swept"].get(key), sort_keys=True)
            for record in records
            if key in record["swept"]
        }
        if len(values) > 1:
            varying.append(key)
    return varying


def _numeric_sweep_keys(records: list[dict[str, Any]]) -> list[str]:
    numeric: list[str] = []
    for key in _varying_sweep_keys(records):
        values = [_safe_float(record["swept"].get(key)) for record in records]
        finite = [value for value in values if math.isfinite(value)]
        if len(set(finite)) >= 2:
            numeric.append(key)
    return numeric


def _metric_value(record: dict[str, Any], metric_key: str) -> float:
    if metric_key == "solve_time_s":
        return _safe_float(record["metadata"].get("solve_time_s"))
    return _safe_float(record["summary"].get(metric_key))


def _other_sweep_label(record: dict[str, Any], x_key: str, varying_keys: list[str]) -> str:
    parts = []
    for key in varying_keys:
        if key == x_key:
            continue
        if key not in record["swept"]:
            continue
        parts.append(f"{key}={_format_value(record['swept'][key])}")
    return ", ".join(parts)


def _plot_metric_vs_sweep(
    test_name: str,
    benchmark_name: str,
    records: list[dict[str, Any]],
    *,
    x_key: str,
    metric_key: str,
    metric_label: str,
    plots_dir: Path,
) -> bool:
    points: list[tuple[float, float, str, str]] = []
    varying_keys = _varying_sweep_keys(records)

    for record in records:
        x_value = _safe_float(record["swept"].get(x_key))
        y_value = _metric_value(record, metric_key)
        if not (math.isfinite(x_value) and math.isfinite(y_value)):
            continue
        label = _other_sweep_label(record, x_key, varying_keys)
        points.append((x_value, y_value, str(record["row"].get("run_id", record["run_dir"].name)), label))

    if len(points) < 2 or len({x for x, *_ in points}) < 2:
        return False

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate batch plots.") from exc

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    grouped: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for x_value, y_value, run_id, label in points:
        grouped[label].append((x_value, y_value, run_id))

    nonempty_labels = [label for label in grouped if label]
    if not nonempty_labels:
        ordered = sorted(grouped[""], key=lambda item: item[0])
        ax.plot(
            [item[0] for item in ordered],
            [item[1] for item in ordered],
            marker="o",
            linewidth=1.8,
        )
    elif len(nonempty_labels) <= 8:
        for label in sorted(grouped):
            ordered = sorted(grouped[label], key=lambda item: item[0])
            ax.plot(
                [item[0] for item in ordered],
                [item[1] for item in ordered],
                marker="o",
                linewidth=1.5,
                label=label or "run",
            )
        ax.legend(
            title="Other swept params",
            fontsize=8,
            title_fontsize=8,
            framealpha=0.95,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
    else:
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        ax.scatter(xs, ys, s=36)
        if len(points) <= 15:
            for x_value, y_value, run_id, _label in points:
                ax.annotate(run_id, (x_value, y_value), textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax.set_xlabel(_human_label(x_key))
    ax.set_ylabel(metric_label)
    ax.set_title(f"{benchmark_name}:\n{metric_label} vs {_human_label(x_key)}", fontsize=13, pad=10)
    ax.grid(True, alpha=0.3)

    test_slug = _slugify(test_name)
    metric_slug = _slugify(metric_key)
    x_slug = _slugify(x_key)
    _save_plot(fig, plots_dir / f"{test_slug}_{metric_slug}_vs_{x_slug}.png")
    return True


def _generate_batch_plots(records: list[dict[str, Any]], plots_dir: Path) -> int:
    created = 0
    records_by_test: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["summary"]:
            records_by_test[record["test"]].append(record)

    for test_name, test_records in sorted(records_by_test.items()):
        if len(test_records) < 2:
            continue

        numeric_sweeps = _numeric_sweep_keys(test_records)
        if not numeric_sweeps:
            continue

        benchmark_name = test_records[0]["benchmark_name"]
        for x_key in numeric_sweeps:
            for metric_key, metric_label in METRIC_SPECS:
                if _plot_metric_vs_sweep(
                    test_name,
                    benchmark_name,
                    test_records,
                    x_key=x_key,
                    metric_key=metric_key,
                    metric_label=metric_label,
                    plots_dir=plots_dir,
                ):
                    created += 1

    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process all runs in a batch directory, generate per-run plots, and build aggregate sweep plots."
    )
    parser.add_argument("batch_dir", help="Batch directory, e.g. output/batches/batch_1")
    parser.add_argument(
        "--recompute-analysis",
        action="store_true",
        help="Rebuild per-run analysis CSVs even when analysis/run_summary.csv already exists.",
    )
    parser.add_argument(
        "--skip-run-plots",
        action="store_true",
        help="Skip regenerating the existing per-run figures under each run's analysis/plots directory.",
    )
    parser.add_argument(
        "--skip-batch-plots",
        action="store_true",
        help="Skip generating aggregate comparison plots across runs.",
    )
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
        help="Optional detailed bond dump mode passed through to postprocessing.",
    )
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir).expanduser().resolve()
    if not batch_dir.exists() or not batch_dir.is_dir():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

    run_dirs = _iter_run_dirs(batch_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {batch_dir}")

    batch_analysis_dir = batch_dir / "analysis"
    batch_analysis_dir.mkdir(parents=True, exist_ok=True)
    batch_plots_dir = batch_analysis_dir / "plots"
    status_map = _batch_status_map(batch_dir)

    records: list[dict[str, Any]] = []
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[batch-plot {index}/{len(run_dirs)}] Processing {_relative_to_repo(run_dir)}")

        summary_path = run_dir / "analysis" / "run_summary.csv"
        analysis_status = "existing" if summary_path.exists() and not args.recompute_analysis else "generated"
        analysis_error = ""
        try:
            analysis_status, _ = _ensure_analysis(
                run_dir,
                recompute=bool(args.recompute_analysis),
                damage_threshold=float(args.damage_threshold),
                stress_threshold=args.stress_threshold,
                bond_dump_mode=args.bond_dump,
            )
        except Exception as exc:
            analysis_status = "failed"
            analysis_error = str(exc)
            print(f"[batch-plot] Analysis failed for {_relative_to_repo(run_dir)}: {exc}")

        run_plots_status = "skipped" if args.skip_run_plots else "pending"
        run_plots_error = ""
        if not args.skip_run_plots and analysis_status != "failed" and (run_dir / "analysis" / "run_summary.csv").exists():
            try:
                generate_plots(run_dir)
                run_plots_status = "success"
            except Exception as exc:
                run_plots_status = "failed"
                run_plots_error = str(exc)
                print(f"[batch-plot] Run plots failed for {_relative_to_repo(run_dir)}: {exc}")

        record = _collect_record(run_dir, status_map.get(run_dir.name))
        record["row"]["analysis_status"] = analysis_status
        record["row"]["analysis_error"] = analysis_error
        record["row"]["run_plots_status"] = run_plots_status
        record["row"]["run_plots_error"] = run_plots_error
        records.append(record)

    combined_rows = [record["row"] for record in records]
    _write_csv(batch_analysis_dir / "aggregate_run_summary.csv", combined_rows)

    records_by_test: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_test[record["test"]].append(record)
    for test_name, test_records in sorted(records_by_test.items()):
        test_slug = _slugify(test_name)
        _write_csv(batch_analysis_dir / f"{test_slug}_aggregate_run_summary.csv", [record["row"] for record in test_records])

    plot_count = 0
    if not args.skip_batch_plots:
        plot_count = _generate_batch_plots(records, batch_plots_dir)

    print(f"[batch-plot] Wrote aggregate summaries to {_relative_to_repo(batch_analysis_dir)}")
    if args.skip_batch_plots:
        print("[batch-plot] Skipped aggregate batch plots.")
    else:
        print(f"[batch-plot] Wrote {plot_count} aggregate plot(s) to {_relative_to_repo(batch_plots_dir)}")


if __name__ == "__main__":
    main()
