"""
Core physics engine utilities. 
Handles simulation configuration and the headless run loop.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import time
import sys
from pathlib import Path
from tqdm import tqdm
from util.export_metrics import append_step_metrics_row, normalize_frame_counts, normalize_step_metrics

# --- 1. Simulation Configuration ---

@dataclass
class SimulationSetup:
    """
    Standard configuration for a simulation test.
    """
    bodies: List[Any]
    constraints: List[Any] = field(default_factory=list)
    dt: float = 1 / 1000
    iterations: int = 20
    gravity: float = -9.81
    friction: float = 0.3
    sync_bodies: bool = False  # Default to False for speed in headless mode
    python_solver_params: Dict[str, Any] = field(default_factory=dict)

    # AMR
    amr_params: Dict[str, Any] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Headless run config
    headless_steps: Optional[int] = None
    headless_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # (Legacy viewer kwargs are ignored in the new pipeline, but kept 
    # so old scripts don't crash immediately if passed)
    viewer_kwargs: Dict[str, Any] = field(default_factory=dict)


# --- 2. Solver Factory ---

def build_solver(setup: SimulationSetup, solver_type: str = "hybrid"):
    """
    Instantiates the solver (Julia Hybrid or Python) based on the setup.
    """
    solver_choice = (solver_type or "hybrid").lower()
    
    # 1. Julia Hybrid Solver
    if solver_choice == "hybrid":
        from jl_solver.hybrid_solver import HybridWorld
        
        solver = HybridWorld(
            setup.bodies,
            setup.constraints,
            dt=setup.dt,
            iterations=setup.iterations,
            gravity=setup.gravity,
            friction=setup.friction,
            sync_bodies=setup.sync_bodies,
            amr=getattr(setup, "amr_params", None),
            solver_params=getattr(setup, "python_solver_params", None),
        )
        return solver

    # 2. Legacy Python Solver (Optional backup)
    elif solver_choice == "python":
        from py_solver.solver_4 import Solver
        # Note: You might need to handle specific constraint conversions here 
        # if your Python solver doesn't support BondData directly.
        solver = Solver(dt=setup.dt, num_iterations=setup.iterations, gravity=setup.gravity)
        
        # Apply params
        params = {
            "mu": 0.3, "post_stabilize": True, "beta": 10000, 
            "alpha": 0.95, "gamma": 0.99
        }
        params.update(setup.python_solver_params or {})
        for attr, value in params.items():
            if hasattr(solver, attr):
                setattr(solver, attr, value)
                
        # Add bodies/constraints
        if hasattr(solver, "add_body"):
            for b in setup.bodies: solver.add_body(b)
        if setup.constraints and hasattr(solver, "add_persistent_constraints"):
            solver.add_persistent_constraints(setup.constraints)
            
        return solver

    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


# --- 3. Headless Execution Loop ---

def run_headless(
    solver,
    num_steps: int,
    export_dir: str,
    steps_per_export: int = 1,
    show_progress: bool = True,
    profile_timings: bool = False,
    **kwargs # Absorb extra kwargs safely
):
    """
    Runs the simulation loop and writes binary frames to disk.
    """
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    steps_per_export = max(1, int(steps_per_export))
    
    # Validate solver capability
    if not hasattr(solver, "write_frame"):
        print("Warning: Solver does not support 'write_frame'. No data will be saved.")
        # Create a dummy function to prevent crash
        solver.write_frame = lambda x: None

    if hasattr(solver, "write_bond_metadata"):
        solver.write_bond_metadata(str(export_path / "bond_meta.bin"))

    # Write initial state (Frame 0)
    frame_counts = normalize_frame_counts(solver.write_frame(str(export_path / "frame_0000.bin")))
    if hasattr(solver, "write_energy_csv"):
        solver.write_energy_csv(str(export_path / "energy_0000.csv"), 0)
    step_metrics = normalize_step_metrics(
        solver.get_last_step_metrics() if hasattr(solver, "get_last_step_metrics") else None
    )
    active_body_count = step_metrics[3] or frame_counts[0]
    active_bond_count = step_metrics[4] or frame_counts[1]
    append_step_metrics_row(
        export_path / "step_metrics.csv",
        frame=0,
        step=0,
        time=0.0,
        iters_used=step_metrics[1],
        max_violation=step_metrics[2],
        active_body_count=active_body_count,
        active_bond_count=active_bond_count,
        exported_body_count=frame_counts[0],
        exported_bond_count=frame_counts[1],
        contact_count=step_metrics[5],
    )

    # Setup Loop
    progress_bar = tqdm(total=num_steps, desc="Simulating", unit="step") if show_progress else None
    
    steps_done = 0
    frame_idx = 0
    start_time = time.time()
    
    # Check for efficient stepping (Julia bridge batching)
    has_step_many = hasattr(solver, "step_many")
    progress_update_steps = kwargs.get("progress_update_steps")
    if progress_update_steps is None:
        # Keep the progress bar responsive without forcing extra frame exports.
        progress_update_steps = max(1, min(steps_per_export, num_steps // 100))
    progress_update_steps = max(1, min(int(progress_update_steps), steps_per_export))

    while steps_done < num_steps:
        # Determine chunk size
        remaining = num_steps - steps_done
        chunk = min(steps_per_export, remaining)
        
        # Step Physics
        chunk_done = 0
        while chunk_done < chunk:
            subchunk = min(progress_update_steps, chunk - chunk_done)
            if has_step_many:
                solver.step_many(subchunk)
            else:
                for _ in range(subchunk):
                    solver.step()

            steps_done += subchunk
            chunk_done += subchunk

            if progress_bar:
                progress_bar.update(subchunk)
        
        # Export Data
        frame_idx += 1
        filename = export_path / f"frame_{frame_idx:04d}.bin"
        frame_counts = normalize_frame_counts(solver.write_frame(str(filename)))
        if hasattr(solver, "write_energy_csv"):
            energy_file = export_path / f"energy_{frame_idx:04d}.csv"
            solver.write_energy_csv(str(energy_file), frame_idx)
        step_metrics = normalize_step_metrics(
            solver.get_last_step_metrics() if hasattr(solver, "get_last_step_metrics") else None
        )
        step_value = step_metrics[0] or steps_done
        active_body_count = step_metrics[3] or frame_counts[0]
        active_bond_count = step_metrics[4] or frame_counts[1]
        append_step_metrics_row(
            export_path / "step_metrics.csv",
            frame=frame_idx,
            step=step_value,
            time=step_value * float(getattr(solver, "dt", 0.0)),
            iters_used=step_metrics[1],
            max_violation=step_metrics[2],
            active_body_count=active_body_count,
            active_bond_count=active_bond_count,
            exported_body_count=frame_counts[0],
            exported_bond_count=frame_counts[1],
            contact_count=step_metrics[5],
        )

    if progress_bar:
        progress_bar.close()

    total_time = time.time() - start_time
    print(f"Simulation finished in {total_time:.2f}s "
          f"({num_steps/total_time:.0f} steps/s)")

    return {
        "num_steps": steps_done,
        "frames_written": frame_idx + 1
    }
