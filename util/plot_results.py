from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _normalized_benchmark_token(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _humanize_benchmark_name(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "Benchmark"

    normalized = text.replace("_", " ").replace("-", " ")
    words: list[str] = []
    for word in normalized.split():
        upper = word.upper()
        if upper in {"ISO", "AVBD"}:
            words.append(upper)
        elif word.isupper() and len(word) <= 4:
            words.append(word)
        else:
            words.append(word.capitalize())

    label = " ".join(words)
    if label.lower().endswith((" benchmark", " test")):
        return label
    return f"{label} benchmark"


def _canonical_benchmark_name(value: object, run_dir: Path) -> str:
    candidates = (value, run_dir.parent.name, run_dir.name)
    for candidate in candidates:
        token = _normalized_benchmark_token(candidate)
        if not token:
            continue
        if "lbar" in token or "lpanel" in token:
            return "L-panel benchmark"
        if "miehe" in token and "shear" in token:
            return "Miehe shear test"
        if "projectile" in token and "impact" in token:
            return "Projectile impact benchmark"
        if "iso20753" in token:
            return "ISO 20753 test"
        if "threepoint" in token and ("fixture" in token or "bend" in token or "bending" in token):
            return "Three-point bending test"
    return _humanize_benchmark_name(value or run_dir.parent.name)


def _benchmark_slug(label: str) -> str:
    slug = []
    for ch in label.lower():
        if ch.isalnum():
            slug.append(ch)
        elif slug and slug[-1] != "_":
            slug.append("_")
    return "".join(slug).strip("_") or "benchmark"


def _read_time_history(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}

    columns: dict[str, list[float]] = {key: [] for key in rows[0].keys()}
    for row in rows:
        for key, value in row.items():
            try:
                columns[key].append(float(value))
            except (TypeError, ValueError):
                columns[key].append(math.nan)

    return {key: np.asarray(values, dtype=float) for key, values in columns.items()}


def _read_summary_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    return dict(rows[0])


def _read_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summary_float(summary: dict[str, str], key: str) -> float:
    try:
        return float(summary.get(key, ""))
    except (TypeError, ValueError):
        return math.nan


def _summary_str(summary: dict[str, str], key: str) -> str:
    return str(summary.get(key, "")).strip()


def _truthy_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _unit_label(
    summary: dict[str, str],
    metadata: dict[str, object],
    *,
    summary_key: str,
    metadata_key: str,
    default: str,
) -> str:
    summary_value = _summary_str(summary, summary_key)
    if summary_value:
        return summary_value
    metadata_value = str(metadata.get(metadata_key) or "").strip()
    if metadata_value:
        return metadata_value
    return default


def _geometry_scaled(summary: dict[str, str], metadata: dict[str, object]) -> bool:
    if _truthy_flag(summary.get("geometry_scaled_to_physical_units")):
        return True
    return _truthy_flag(metadata.get("geometry_scaled_to_physical_units"))


def _displacement_unit_label(summary: dict[str, str], metadata: dict[str, object]) -> str:
    explicit = _unit_label(
        summary,
        metadata,
        summary_key="displacement_unit_label",
        metadata_key="displacement_unit_label",
        default="",
    )
    if explicit:
        return explicit

    if _geometry_scaled(summary, metadata):
        return "m"
    return "exported coordinate units"


def _require_finite_column(data: dict[str, np.ndarray], key: str, *, context: str) -> np.ndarray:
    values = data.get(key)
    if values is None:
        raise RuntimeError(f"{context}: '{key}' is missing from time_history.csv.")
    if not np.isfinite(values).any():
        raise RuntimeError(f"{context}: '{key}' in time_history.csv has no finite values.")
    return values


def _series_has_finite_values(values: np.ndarray | None) -> bool:
    return values is not None and np.isfinite(values).any()


def _series_has_positive_values(values: np.ndarray | None) -> bool:
    if values is None:
        return False
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return float(np.max(finite)) > 0.0


def _select_displacement_axis(data: dict[str, np.ndarray]) -> np.ndarray:
    for key in ("load_displacement_along_loading_axis", "load_displacement_magnitude"):
        values = data.get(key)
        if _series_has_finite_values(values):
            return values
    raise RuntimeError(
        "Requested x-axis 'displacement' but neither 'load_displacement_along_loading_axis' nor "
        "'load_displacement_magnitude' has finite values in time_history.csv."
    )


def _select_progress_axis(
    data: dict[str, np.ndarray],
    *,
    x_axis_mode: str,
    displacement_unit_label: str,
) -> tuple[np.ndarray, str, str, str]:
    if x_axis_mode == "frame":
        frame_values = data.get("frame")
        if frame_values is None:
            return (
                np.arange(len(next(iter(data.values()))), dtype=float),
                "Frame",
                "Frame",
                "frame",
            )
        if not np.isfinite(frame_values).any():
            raise RuntimeError("Requested x-axis 'frame' but 'frame' in time_history.csv has no finite values.")
        return frame_values, "Frame", "Frame", "frame"

    if x_axis_mode == "step":
        return _require_finite_column(data, "step", context=f"Requested x-axis '{x_axis_mode}'"), "Solver step", "Solver Step", "step"

    if x_axis_mode == "displacement":
        return (
            _select_displacement_axis(data),
            f"Prescribed displacement [{displacement_unit_label}]",
            "Prescribed Displacement",
            "displacement",
        )

    if x_axis_mode == "time":
        return _require_finite_column(data, "time", context=f"Requested x-axis '{x_axis_mode}'"), "Time [s]", "Time", "time"

    raise ValueError(f"Unsupported x-axis mode: {x_axis_mode}")


def _event_markers(summary: dict[str, str], data: dict[str, np.ndarray], x_axis_mode: str) -> list[tuple[str, float]]:
    key_map = (
        ("first break", "first_broken_bond"),
        ("peak growth stage", "peak_crack_growth_stage"),
        ("final plateau", "final_plateau"),
        ("peak traction-cap util", "peak_mixed_mode_traction_utilization"),
    )

    if x_axis_mode == "frame":
        suffix = "frame"
    elif x_axis_mode == "time":
        suffix = "time"
    elif x_axis_mode == "displacement":
        suffix = "displacement"
    elif x_axis_mode == "step":
        frame_values = data.get("frame")
        if frame_values is None:
            raise RuntimeError(
                "Requested x-axis 'step' but event markers require the 'frame' column in time_history.csv "
                "because run_summary.csv stores these event locations by frame."
            )
        step_values = data.get("step")
        if step_values is None:
            raise RuntimeError("Requested x-axis 'step' but 'step' is missing from time_history.csv.")
        finite_mask = np.isfinite(frame_values) & np.isfinite(step_values)
        if not np.any(finite_mask):
            raise RuntimeError(
                "Requested x-axis 'step' but 'frame' and 'step' in time_history.csv do not contain "
                "overlapping finite values for event mapping."
            )
        frame_finite = frame_values[finite_mask]
        step_finite = step_values[finite_mask]
        events: list[tuple[str, float]] = []
        for label, base_key in key_map:
            frame_value = _summary_float(summary, f"{base_key}_frame")
            if math.isfinite(frame_value):
                nearest_idx = int(np.argmin(np.abs(frame_finite - frame_value)))
                events.append((label, float(step_finite[nearest_idx])))
        return events
    else:
        raise ValueError(f"Unsupported x-axis mode: {x_axis_mode}")

    events: list[tuple[str, float]] = []
    for label, base_key in key_map:
        value = _summary_float(summary, f"{base_key}_{suffix}")
        if math.isfinite(value):
            events.append((label, float(value)))
    return events


def _format_plot_title(benchmark_name: str, quantity_label: str, title_suffix: str) -> str:
    return f"{benchmark_name}:\n{quantity_label} vs {title_suffix}"


def _annotate_events(ax, x: np.ndarray, y: np.ndarray, events: list[tuple[str, float]]) -> None:
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite_mask):
        return

    x_finite = x[finite_mask]
    y_finite = y[finite_mask]
    for idx, (label, x_value) in enumerate(events):
        marker_color = f"C{idx + 1}"
        nearest_idx = int(np.argmin(np.abs(x_finite - x_value)))
        x_mark = float(x_finite[nearest_idx])
        y_mark = float(y_finite[nearest_idx])
        ax.axvline(x_mark, color=marker_color, linestyle="--", linewidth=0.9, alpha=0.35)
        ax.scatter([x_mark], [y_mark], color=marker_color, s=22, zorder=4)


def _add_event_key(ax, events: list[tuple[str, float]]) -> None:
    if not events:
        return

    from matplotlib.lines import Line2D

    handles = []
    for idx, (label, _x_value) in enumerate(events):
        marker_color = f"C{idx + 1}"
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                markersize=6,
                markerfacecolor=marker_color,
                markeredgecolor=marker_color,
                label=label,
            )
        )

    ax.legend(
        handles=handles,
        title="Key events",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        ncol=1,
        framealpha=0.95,
        fontsize=8,
        title_fontsize=8,
        borderaxespad=0.0,
    )


def _save_plot(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.1)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.12)
    fig.clf()
    plt.close(fig)


def _clear_existing_plots(plots_dir: Path, benchmark_slug: str) -> None:
    if not plots_dir.exists():
        return
    for path in plots_dir.glob(f"{benchmark_slug}_*.png"):
        path.unlink(missing_ok=True)


def generate_plots(run_dir: str | Path, analysis_dir_name: str = "analysis", x_axis: str = "frame") -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    analysis_dir = run_dir / analysis_dir_name
    history_path = analysis_dir / "time_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing {history_path}. Run util/postprocess_results.py first.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate plots.") from exc

    data = _read_time_history(history_path)
    if not data:
        raise RuntimeError(f"No rows found in {history_path}")

    metadata = _read_metadata(run_dir / "meta_data.json")
    summary = _read_summary_row(analysis_dir / "run_summary.csv")
    benchmark_name = _canonical_benchmark_name(summary.get("benchmark_name") or metadata.get("benchmark_name"), run_dir)
    benchmark_slug = _benchmark_slug(benchmark_name)
    displacement_unit_label = _displacement_unit_label(summary, metadata)
    area_unit_label = _unit_label(
        summary,
        metadata,
        summary_key="area_unit_label",
        metadata_key="area_unit_label",
        default="exported coordinate units^2",
    )
    stress_unit_label = _unit_label(
        summary,
        metadata,
        summary_key="stress_unit_label",
        metadata_key="stress_unit_label",
        default="stress units of the solver setup",
    )
    energy_unit_label = _unit_label(
        summary,
        metadata,
        summary_key="energy_unit_label",
        metadata_key="energy_unit_label",
        default="energy units of the solver setup",
    )
    x, x_label, title_suffix, axis_slug = _select_progress_axis(
        data,
        x_axis_mode=x_axis,
        displacement_unit_label=displacement_unit_label,
    )
    events = _event_markers(summary, data, x_axis)
    plots_dir = analysis_dir / "plots"
    _clear_existing_plots(plots_dir, benchmark_slug)

    crack_area = _require_finite_column(data, "crack_area_proxy", context="Missing required plot series")
    broken_bond_count = _require_finite_column(data, "broken_bond_count", context="Missing required plot series")
    peak_stress_proxy = _require_finite_column(data, "peak_stress_proxy", context="Missing required plot series")
    traction_cap_utilization = data.get("peak_mixed_mode_traction_utilization")
    stress_exceedance_fraction = data.get("stress_exceedance_fraction")
    process_zone_area_proxy = data.get("process_zone_area_proxy")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, crack_area, linewidth=2.0)
    _annotate_events(ax, x, crack_area, events)
    _add_event_key(ax, events)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Crack area proxy [{area_unit_label}]")
    ax.set_title(_format_plot_title(benchmark_name, "Crack Area Proxy", title_suffix), fontsize=13, pad=10)
    ax.grid(True, alpha=0.3)
    _save_plot(fig, plots_dir / f"{benchmark_slug}_crack_area_proxy_vs_{axis_slug}.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, broken_bond_count, linewidth=1.8)
    _annotate_events(ax, x, broken_bond_count, events)
    _add_event_key(ax, events)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Broken bond count")
    ax.set_title(_format_plot_title(benchmark_name, "Broken Bond Count", title_suffix), fontsize=13, pad=10)
    ax.grid(True, alpha=0.3)
    _save_plot(fig, plots_dir / f"{benchmark_slug}_broken_bond_count_vs_{axis_slug}.png")

    if _series_has_finite_values(traction_cap_utilization):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, traction_cap_utilization, linewidth=2.0)
        _annotate_events(ax, x, traction_cap_utilization, events)
        _add_event_key(ax, events)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Peak bond traction-cap utilization")
        ax.set_title(
            _format_plot_title(benchmark_name, "Peak Bond Traction-Cap Utilization (Diagnostic)", title_suffix),
            fontsize=13,
            pad=10,
        )
        ax.grid(True, alpha=0.3)
        _save_plot(fig, plots_dir / f"{benchmark_slug}_mixed_mode_traction_utilization_vs_{axis_slug}.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, peak_stress_proxy)
    _annotate_events(ax, x, peak_stress_proxy, events)
    _add_event_key(ax, events)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Max principal tensile stress proxy [{stress_unit_label}]")
    ax.set_title(_format_plot_title(benchmark_name, "Peak Stress Proxy", title_suffix), fontsize=13, pad=10)
    ax.grid(True, alpha=0.3)
    _save_plot(fig, plots_dir / f"{benchmark_slug}_peak_stress_proxy_vs_{axis_slug}.png")

    if _series_has_finite_values(stress_exceedance_fraction):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, stress_exceedance_fraction)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Fraction above stress-proxy threshold")
        ax.set_title(_format_plot_title(benchmark_name, "Stress-Proxy Threshold Exceedance", title_suffix), fontsize=13, pad=10)
        ax.grid(True, alpha=0.3)
        _save_plot(fig, plots_dir / f"{benchmark_slug}_stress_proxy_threshold_exceedance_vs_{axis_slug}.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    plotted_energy = False
    for key, label in (
        ("fracture_work", "Fracture work"),
        ("bond_potential", "Bond potential"),
        ("kinetic", "Kinetic"),
        ("contact_potential", "Contact potential"),
    ):
        if key in data and np.isfinite(data[key]).any():
            ax.plot(x, data[key], label=label)
            plotted_energy = True
    if plotted_energy:
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"Energy [{energy_unit_label}]")
        ax.set_title(_format_plot_title(benchmark_name, "Energy Diagnostics", title_suffix), fontsize=13, pad=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _save_plot(fig, plots_dir / f"{benchmark_slug}_energy_diagnostics_vs_{axis_slug}.png")
    else:
        plt.close(fig)

    if _series_has_positive_values(process_zone_area_proxy):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, process_zone_area_proxy)
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"Process-zone area proxy [{area_unit_label}]")
        ax.set_title(_format_plot_title(benchmark_name, "Process-Zone Area Proxy", title_suffix), fontsize=13, pad=10)
        ax.grid(True, alpha=0.3)
        _save_plot(fig, plots_dir / f"{benchmark_slug}_process_zone_area_proxy_vs_{axis_slug}.png")

    print(f"[plot] Wrote figures to {plots_dir}")
    return plots_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lightweight result figures from analysis CSVs.")
    parser.add_argument("run_dir", help="Run directory, e.g. output/projectile_impact/run_001")
    parser.add_argument("--analysis-dir", default="analysis", help="Analysis subdirectory name.")
    parser.add_argument(
        "--x-axis",
        default="frame",
        choices=("frame", "step", "displacement", "time"),
        help="X-axis for generated plots. Defaults to exported frame.",
    )
    args = parser.parse_args()

    generate_plots(args.run_dir, analysis_dir_name=args.analysis_dir, x_axis=args.x_axis)


if __name__ == "__main__":
    main()
