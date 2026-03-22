from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


STEP_METRIC_FIELDS = [
    "frame",
    "step",
    "time",
    "iters_used",
    "max_violation",
    "active_body_count",
    "active_bond_count",
    "exported_body_count",
    "exported_bond_count",
    "contact_count",
]


def normalize_frame_counts(frame_result: Any) -> tuple[int, int]:
    if frame_result is None:
        return 0, 0
    try:
        n_bodies, n_bonds = frame_result
        return int(n_bodies), int(n_bonds)
    except Exception:
        return 0, 0


def normalize_step_metrics(metrics_result: Any) -> tuple[int, int, float, int, int, int]:
    if metrics_result is None:
        return 0, 0, 0.0, 0, 0, 0
    try:
        step_count, iters_used, max_violation, active_body_count, active_bond_count, contact_count = metrics_result
        return (
            int(step_count),
            int(iters_used),
            float(max_violation),
            int(active_body_count),
            int(active_bond_count),
            int(contact_count),
        )
    except Exception:
        return 0, 0, 0.0, 0, 0, 0


def append_step_metrics_row(
    csv_path: str | Path,
    *,
    frame: int,
    step: int,
    time: float,
    iters_used: int,
    max_violation: float,
    active_body_count: int,
    active_bond_count: int,
    exported_body_count: int,
    exported_bond_count: int,
    contact_count: int,
) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STEP_METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "frame": int(frame),
                "step": int(step),
                "time": float(time),
                "iters_used": int(iters_used),
                "max_violation": float(max_violation),
                "active_body_count": int(active_body_count),
                "active_bond_count": int(active_bond_count),
                "exported_body_count": int(exported_body_count),
                "exported_bond_count": int(exported_bond_count),
                "contact_count": int(contact_count),
            }
        )
