"""
CLI helper to run simulation setups from ``tests`` with a chosen solver backend.

Usage:
    python util/run_test.py double_beam_fall hybrid

- First argument: test module name under ``tests`` (without ``.py``).
- Second argument: solver backend (``hybrid`` by default, or ``python``).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
import json
from pathlib import Path

from util.pyvista_visualizer import SimulationSetup, run_simulation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _get_save_dir(experiment_name: str) -> Path:
    base_dir = REPO_ROOT / "output" / experiment_name
    base_dir.mkdir(parents=True, exist_ok=True)

    existing = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]

    if not existing:
        next_id = 1
    else:
        ids = []
        for p in existing:
            try:
                ids.append(int(p.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
        next_id = max(ids) + 1 if ids else 1

    run_dir = base_dir / f"run_{next_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _load_setup(module_name: str, *, sync_bodies: bool | None) -> SimulationSetup:
    if module_name.endswith(".py"):
        module_name = module_name[:-3]
    module_path = f"tests.{module_name}"

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Test '{module_name}' not found under tests/") from exc

    setup_fn = getattr(module, "build_setup", None)
    if not callable(setup_fn):
        raise SystemExit(f"Test '{module_name}' must expose a callable build_setup()")

    kwargs = {}
    sig = inspect.signature(setup_fn)
    if sync_bodies is not None and "sync_bodies" in sig.parameters:
        kwargs["sync_bodies"] = sync_bodies

    setup = setup_fn(**kwargs)
    if not isinstance(setup, SimulationSetup):
        raise SystemExit(f"build_setup() in '{module_name}' must return SimulationSetup")

    if sync_bodies is not None:
        setup.sync_bodies = sync_bodies

    return setup


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run a test scene with a chosen solver backend.")
    parser.add_argument(
        "--test",
        help="Test module under tests/ (without .py), e.g., 'double_beam_fall'",
    )
    parser.add_argument(
        "--solver",
        nargs="?",
        default="hybrid",
        choices=["hybrid", "python"],
        help="Solver backend to use (hybrid Julia solver by default).",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Hybrid only: disable copying poses back to Python each step.",
    )
    args = parser.parse_args(argv)

    test_name = args.test
    setup = _load_setup(test_name, sync_bodies=not args.no_sync)
    
    run_dir = _get_save_dir(args.test)
    print(f"\n[Run Manager] Output Directory: {run_dir}")

    if setup.headless_kwargs is None:
        setup.headless_kwargs = {}
    
    setup.headless_kwargs["export_binary"] = True
    setup.headless_kwargs["export_dir"] = str(run_dir / "raw")

    manifest = {
        "test": args.test,
        "solver": args.solver,
        "dt": setup.dt,
        "iterations": setup.iterations,
        "gravity": setup.gravity,
        "friction": setup.friction,
        # Add python_solver_params if needed
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # 4. Run
    result = run_simulation(setup, solver_type=args.solver)

    samples = result.get("timing_samples", [])
    print(samples[:3]) 
    
    # 5. Suggest Next Step
    print(f"\n[Run Manager] Done. To visualize:")
    print(f"python util/process_data.py {run_dir}")

if __name__ == "__main__":
    main()
