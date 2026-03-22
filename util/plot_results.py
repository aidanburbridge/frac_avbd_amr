from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


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


def _save_plot(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.clf()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lightweight result figures from analysis CSVs.")
    parser.add_argument("run_dir", help="Run directory, e.g. output/projectile_impact/run_001")
    parser.add_argument("--analysis-dir", default="analysis", help="Analysis subdirectory name.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    analysis_dir = run_dir / args.analysis_dir
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

    t = data.get("time", np.arange(len(next(iter(data.values()))), dtype=float))
    plots_dir = analysis_dir / "plots"

    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ("kinetic", "bond_potential", "contact_potential", "fracture_work", "mech_energy"):
        if key in data and np.isfinite(data[key]).any():
            ax.plot(t, data[key], label=key)
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_title("Energy vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_plot(fig, plots_dir / "energy_vs_time.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, data.get("broken_bond_count", np.zeros_like(t)))
    ax.set_xlabel("Time")
    ax.set_ylabel("Broken bond count")
    ax.set_title("Broken Bonds vs Time")
    ax.grid(True, alpha=0.3)
    _save_plot(fig, plots_dir / "broken_bonds_vs_time.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    if "crack_area_proxy" in data:
        ax.plot(t, data["crack_area_proxy"], label="crack_area_proxy")
    if "process_zone_area_proxy" in data and np.isfinite(data["process_zone_area_proxy"]).any():
        ax.plot(t, data["process_zone_area_proxy"], label="process_zone_area_proxy")
    ax.set_xlabel("Time")
    ax.set_ylabel("Area proxy")
    ax.set_title("Crack / Process Zone Area vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_plot(fig, plots_dir / "crack_area_vs_time.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, data.get("peak_stress_proxy", np.full_like(t, math.nan)))
    ax.set_xlabel("Time")
    ax.set_ylabel("Peak stress proxy")
    ax.set_title("Peak Stress Proxy vs Time")
    ax.grid(True, alpha=0.3)
    _save_plot(fig, plots_dir / "peak_stress_proxy_vs_time.png")

    if "stress_exceedance_fraction" in data and np.isfinite(data["stress_exceedance_fraction"]).any():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, data["stress_exceedance_fraction"])
        ax.set_xlabel("Time")
        ax.set_ylabel("Threshold exceedance fraction")
        ax.set_title("Stress-Proxy Threshold Exceedance vs Time")
        ax.grid(True, alpha=0.3)
        _save_plot(fig, plots_dir / "stress_threshold_exceedance_vs_time.png")

    load_key = None
    for candidate in ("load_displacement_along_loading_axis", "load_displacement_magnitude"):
        if candidate in data and np.isfinite(data[candidate]).any():
            load_key = candidate
            break
    if load_key is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, data[load_key])
        ax.set_xlabel("Time")
        ax.set_ylabel(load_key)
        ax.set_title("Load Displacement vs Time")
        ax.grid(True, alpha=0.3)
        _save_plot(fig, plots_dir / "load_displacement_vs_time.png")

    print(f"[plot] Wrote figures to {plots_dir}")


if __name__ == "__main__":
    main()
