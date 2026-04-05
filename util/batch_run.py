from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import itertools
import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.engine import SimulationSetup, build_solver, run_headless
from util.simulate import _save_metadata, convert_results

OUTPUT_ROOT = REPO_ROOT / "output" / "batches"


class ManifestError(ValueError):
    pass


def _json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _slugify(value: str) -> str:
    cleaned = []
    for char in value.strip():
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return slug or "batch"


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ManifestError("Manifest root must be a JSON object.")
    return data


def _get_test_spec(test_name: str) -> dict[str, Any]:
    if test_name in {"ISO_20753", "iso_20753_tensile"}:
        return {
            "module_name": "tests.ISO_20753",
            "canonical_name": "ISO_20753",
            "defaults_fn": _iso_20753_defaults,
            "overrides_fn": _iso_20753_overrides,
        }
    if test_name == "L_bar":
        return {
            "module_name": "tests.L_bar",
            "canonical_name": "L_bar",
            "defaults_fn": _l_bar_defaults,
            "overrides_fn": _l_bar_overrides,
        }
    if test_name == "miehe_shear":
        return {
            "module_name": "tests.miehe_shear",
            "canonical_name": "miehe_shear",
            "defaults_fn": _miehe_shear_defaults,
            "overrides_fn": _miehe_shear_overrides,
        }
    if test_name in {"three_point_bending", "three_point_fixture"}:
        return {
            "module_name": "tests.three_point_fixture",
            "canonical_name": "three_point_bending",
            "defaults_fn": _three_point_defaults,
            "overrides_fn": _three_point_overrides,
        }
    if test_name == "projectile_impact":
        return {
            "module_name": "tests.projectile_impact",
            "canonical_name": "projectile_impact",
            "defaults_fn": _projectile_impact_defaults,
            "overrides_fn": _projectile_impact_overrides,
        }
    raise ManifestError(
        "Unsupported test "
        f"'{test_name}'. Supported tests: ISO_20753, L_bar, miehe_shear, three_point_bending, projectile_impact."
    )


def _iso_20753_defaults(module) -> dict[str, float]:
    return {
        "load_velocity": float(module.PULL_RATE),
        "dt_physics": float(module.DT_PHYSICS),
        "youngs_modulus": float(module.E_MODULUS),
        "steps": int(module.STEPS),
    }


def _iso_20753_overrides(module, resolved: dict[str, Any]) -> dict[str, Any]:
    dt_physics = float(resolved["dt_physics"])
    return {
        "PULL_RATE": float(resolved["load_velocity"]),
        "DT_PHYSICS": dt_physics,
        "E_MODULUS": float(resolved["youngs_modulus"]),
        "STEPS": int(resolved["steps"]),
        "STEPS_PER": max(1, int(module.DT_RENDER / dt_physics)),
    }


def _l_bar_defaults(module) -> dict[str, float]:
    load_velocity = np.asarray(module.LOAD_VELOCITY, dtype=float)
    return {
        "load_velocity": float(load_velocity[1]),
        "dt_physics": float(module.DT_PHYSICS),
        "youngs_modulus": float(module.E_MODULUS),
        "steps": int(module.STEPS),
    }


def _l_bar_overrides(module, resolved: dict[str, Any]) -> dict[str, Any]:
    dt_physics = float(resolved["dt_physics"])
    return {
        "LOAD_VELOCITY": np.array([0.0, float(resolved["load_velocity"]), 0.0], dtype=float),
        "DT_PHYSICS": dt_physics,
        "E_MODULUS": float(resolved["youngs_modulus"]),
        "STEPS": int(resolved["steps"]),
        "STEPS_PER_EXPORT": max(1, int(module.DT_RENDER / dt_physics)),
    }


def _three_point_defaults(module) -> dict[str, float]:
    return {
        "load_velocity": float(module.PULL_RATE),
        "dt_physics": float(module.DT_PHYSICS),
        "youngs_modulus": float(module.E_MODULUS),
        "steps": int(module.STEPS),
    }


def _three_point_overrides(module, resolved: dict[str, Any]) -> dict[str, Any]:
    dt_physics = float(resolved["dt_physics"])
    return {
        "PULL_RATE": abs(float(resolved["load_velocity"])),
        "DT_PHYSICS": dt_physics,
        "E_MODULUS": float(resolved["youngs_modulus"]),
        "STEPS": int(resolved["steps"]),
        "STEPS_PER": max(1, int(module.DT_RENDER / dt_physics)),
    }


def _miehe_shear_defaults(module) -> dict[str, float]:
    load_velocity = np.asarray(module.LOAD_VELOCITY, dtype=float)
    return {
        "load_velocity": float(load_velocity[0]),
        "dt_physics": float(module.DT_PHYSICS),
        "youngs_modulus": float(module.E_MODULUS),
        "steps": int(module.STEPS),
    }


def _miehe_shear_overrides(module, resolved: dict[str, Any]) -> dict[str, Any]:
    dt_physics = float(resolved["dt_physics"])
    youngs_modulus = float(resolved["youngs_modulus"])
    return {
        "LOAD_VELOCITY": np.array([float(resolved["load_velocity"]), 0.0, 0.0], dtype=float),
        "DT_PHYSICS": dt_physics,
        "E_MODULUS": youngs_modulus,
        "FRACTURE_TOUGHNESS": np.sqrt(youngs_modulus * float(module.GC_TARGET) / (1.0 - float(module.NU) ** 2)),
        "STEPS": int(resolved["steps"]),
        "STEPS_PER_EXPORT": max(1, int(module.DT_RENDER / dt_physics)),
    }


def _projectile_impact_defaults(module) -> dict[str, float]:
    return {
        "impact_velocity": float(module.IMPACT_VELOCITY),
        "dt_physics": float(module.DT_PHYSICS),
        "steps": int(module.STEPS),
    }


def _projectile_impact_overrides(module, resolved: dict[str, Any]) -> dict[str, Any]:
    dt_physics = float(resolved["dt_physics"])
    return {
        "IMPACT_VELOCITY": float(resolved["impact_velocity"]),
        "DT_PHYSICS": dt_physics,
        "STEPS": int(resolved["steps"]),
        "STEPS_PER_EXPORT": max(1, int(module.DT_RENDER / dt_physics)),
    }


def _validate_run_spec(run_spec: Any, index: int) -> dict[str, Any]:
    if not isinstance(run_spec, dict):
        raise ManifestError(f"runs[{index}] must be an object.")

    test_name = run_spec.get("test")
    mode = run_spec.get("mode")
    vary = run_spec.get("vary", {})
    fixed = run_spec.get("fixed", {})

    if not isinstance(test_name, str) or not test_name:
        raise ManifestError(f"runs[{index}].test must be a non-empty string.")
    if mode not in {"product", "zip"}:
        raise ManifestError(f"runs[{index}].mode must be 'product' or 'zip'.")
    if not isinstance(vary, dict):
        raise ManifestError(f"runs[{index}].vary must be an object.")
    if not isinstance(fixed, dict):
        raise ManifestError(f"runs[{index}].fixed must be an object when provided.")

    spec = _get_test_spec(test_name)
    module = importlib.import_module(spec["module_name"])
    defaults = spec["defaults_fn"](module)
    allowed_keys = set(defaults)

    overlap = set(vary).intersection(fixed)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ManifestError(f"runs[{index}] has keys in both 'vary' and 'fixed': {names}")

    unknown_keys = (set(vary) | set(fixed)) - allowed_keys
    if unknown_keys:
        names = ", ".join(sorted(unknown_keys))
        supported = ", ".join(sorted(allowed_keys))
        raise ManifestError(
            f"runs[{index}] has unsupported parameters: {names}. Supported parameters for {test_name}: {supported}"
        )

    for key, values in vary.items():
        if not isinstance(values, list) or not values:
            raise ManifestError(f"runs[{index}].vary.{key} must be a non-empty list.")

    if mode == "zip" and vary:
        lengths = {len(values) for values in vary.values()}
        if len(lengths) != 1:
            raise ManifestError(f"runs[{index}] uses mode='zip' but vary lists have different lengths.")

    return {
        "test": test_name,
        "mode": mode,
        "vary": vary,
        "fixed": fixed,
        "spec": spec,
        "defaults": defaults,
    }


def _expand_parameter_sets(mode: str, vary: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not vary:
        return [{}]

    keys = list(vary)
    values = [vary[key] for key in keys]

    if mode == "product":
        combos = itertools.product(*values)
    else:
        combos = zip(*values, strict=True)

    return [dict(zip(keys, combo)) for combo in combos]


def _expand_manifest(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], str]:
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ManifestError("Manifest must contain a non-empty 'runs' list.")

    batch_name = manifest.get("batch_name")
    if batch_name is not None and (not isinstance(batch_name, str) or not batch_name.strip()):
        raise ManifestError("batch_name must be a non-empty string when provided.")

    expanded_runs: list[dict[str, Any]] = []
    sweep_keys: list[str] = []

    for index, run_spec in enumerate(runs):
        validated = _validate_run_spec(run_spec, index)
        parameter_sets = _expand_parameter_sets(validated["mode"], validated["vary"])

        for key in validated["vary"]:
            if key not in sweep_keys:
                sweep_keys.append(key)

        for swept_values in parameter_sets:
            resolved = dict(validated["defaults"])
            resolved.update(validated["fixed"])
            resolved.update(swept_values)
            expanded_runs.append(
                {
                    "test": validated["test"],
                    "canonical_name": validated["spec"]["canonical_name"],
                    "module_name": validated["spec"]["module_name"],
                    "resolved": resolved,
                    "swept": swept_values,
                    "overrides_fn": validated["spec"]["overrides_fn"],
                }
            )

    return expanded_runs, sweep_keys, batch_name or ""


def _create_batch_dir(batch_name: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base_name = _slugify(batch_name)
    candidate = OUTPUT_ROOT / base_name
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    for index in itertools.count(2):
        candidate = OUTPUT_ROOT / f"{base_name}_{index:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate

    raise RuntimeError("Unable to allocate a batch output directory.")


@contextmanager
def _temporary_overrides(module, overrides: dict[str, Any]):
    originals = {name: getattr(module, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


def _build_setup(module) -> SimulationSetup:
    setup_fn = getattr(module, "build_setup", None)
    if not callable(setup_fn):
        raise RuntimeError(f"{module.__name__} must expose build_setup().")

    kwargs = {}
    if "sync_bodies" in inspect.signature(setup_fn).parameters:
        kwargs["sync_bodies"] = False

    setup = setup_fn(**kwargs)
    if not isinstance(setup, SimulationSetup):
        raise RuntimeError(f"{module.__name__}.build_setup() must return SimulationSetup.")

    setup.sync_bodies = False
    if setup.headless_kwargs is None:
        setup.headless_kwargs = {}
    return setup


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _write_summary_csv(path: Path, rows: list[dict[str, Any]], sweep_keys: list[str]) -> None:
    fieldnames = ["run_id", "test", "status", "output_dir", *sweep_keys]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _run_single_case(
    run_index: int,
    total_runs: int,
    batch_dir: Path,
    run_spec: dict[str, Any],
    solver_type: str,
) -> dict[str, Any]:
    run_id = f"run_{run_index:03d}"
    test_name = run_spec["canonical_name"]
    run_dir = batch_dir / f"{test_name}_run_{run_index:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)

    resolved_config = {
        "run_id": run_id,
        "test": test_name,
        "solver": solver_type,
        "parameters": run_spec["resolved"],
        "swept_parameters": run_spec["swept"],
        "output_dir": _relative_to_repo(run_dir),
    }
    _write_json(run_dir / "resolved_config.json", resolved_config)

    summary_row = {
        "run_id": run_id,
        "test": test_name,
        "status": "failed",
        "output_dir": _relative_to_repo(run_dir),
    }
    summary_row.update(run_spec["swept"])

    print(f"[{run_index}/{total_runs}] Running {test_name} -> {_relative_to_repo(run_dir)}")

    module = importlib.import_module(run_spec["module_name"])
    overrides = run_spec["overrides_fn"](module, run_spec["resolved"])

    try:
        with _temporary_overrides(module, overrides):
            setup = _build_setup(module)
            raw_dir = run_dir / "raw"
            steps = int(setup.headless_steps or 1000)
            headless_kwargs = dict(setup.headless_kwargs or {})
            solver = build_solver(setup, solver_type=solver_type)
            solve_start = time.perf_counter()
            run_headless(solver, num_steps=steps, export_dir=str(raw_dir), **headless_kwargs)
            solve_time_s = time.perf_counter() - solve_start

            metadata_args = SimpleNamespace(test=test_name, solver=solver_type)
            _save_metadata(run_dir, setup, metadata_args, total_steps=steps, solve_time_s=solve_time_s)
            convert_results(run_dir)

        summary_row["status"] = "success"
        print(f"[{run_index}/{total_runs}] Success")
        return summary_row
    except Exception as exc:
        failure = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(run_dir / "error.json", failure)
        print(f"[{run_index}/{total_runs}] Failed: {exc}")
        return summary_row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a simple batch of existing benchmark setups.")
    parser.add_argument("manifest", help="Path to the batch manifest JSON file.")
    parser.add_argument(
        "--solver",
        default="hybrid",
        choices=["hybrid", "python"],
        help="Solver backend to use for all runs.",
    )
    parser.add_argument(
        "--batch-name",
        help="Optional output batch name. Defaults to manifest batch_name or manifest filename stem.",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    manifest = _load_manifest(manifest_path)
    expanded_runs, sweep_keys, manifest_batch_name = _expand_manifest(manifest)

    batch_name = args.batch_name or manifest_batch_name or manifest_path.stem
    batch_dir = _create_batch_dir(batch_name)
    summary_csv_path = batch_dir / "batch_summary.csv"

    _write_json(batch_dir / "manifest.json", manifest)

    print(f"Batch output: {_relative_to_repo(batch_dir)}")
    print(f"Expanded {len(expanded_runs)} runs from {manifest_path.name}")

    summary_rows: list[dict[str, Any]] = []
    for run_index, run_spec in enumerate(expanded_runs, start=1):
        summary_rows.append(
            _run_single_case(
                run_index=run_index,
                total_runs=len(expanded_runs),
                batch_dir=batch_dir,
                run_spec=run_spec,
                solver_type=args.solver,
            )
        )
        _write_summary_csv(summary_csv_path, summary_rows, sweep_keys)

    success_count = sum(1 for row in summary_rows if row["status"] == "success")
    failed_count = len(summary_rows) - success_count
    print(f"Batch complete: {success_count} succeeded, {failed_count} failed")
    print(f"Summary CSV: {_relative_to_repo(summary_csv_path)}")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ManifestError as exc:
        print(f"Manifest error: {exc}")
        raise SystemExit(2)
