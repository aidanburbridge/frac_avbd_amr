from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.plot_results import generate_plots
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
        "--x-axis",
        default="frame",
        choices=("frame", "step", "displacement", "time"),
        help="X-axis passed through to util.plot_results.",
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
        except Exception as exc:
            print(f"[FAIL] {run_dir}: {exc}")
            failed.append((run_dir, str(exc)))
            if not args.continue_on_error:
                break

    skipped = len(run_dirs) - plotted - len(failed)

    print("Summary:")
    print(f"  Found: {len(run_dirs)}")
    print(f"  Plotted: {plotted}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {len(failed)}")
    if failed:
        for run_dir, message in failed:
            print(f"  - {run_dir}: {message}")

    return 1 if run_dirs and plotted == 0 and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
