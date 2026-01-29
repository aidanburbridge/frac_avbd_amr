"""
Unified Simulation Runner & Exporter.
Usage: python -m util.simulate [test_name]
"""

import argparse
import importlib
import inspect
import sys
import json
import struct
import shutil
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Imports
from util.vtk_exporter import VTKExporter
from util.engine import SimulationSetup, build_solver, run_headless

# Path helper
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ==========================================
# 1. SETUP HELPERS
# ==========================================

def _get_save_dir(experiment_name: str) -> Path:
    base_dir = REPO_ROOT / "output" / experiment_name
    base_dir.mkdir(parents=True, exist_ok=True)

    existing = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    next_id = 1
    if existing:
        ids = []
        for p in existing:
            try: ids.append(int(p.name.split("_")[1]))
            except: pass
        if ids: next_id = max(ids) + 1

    run_dir = base_dir / f"run_{next_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _load_setup(module_name: str) -> SimulationSetup:
    if module_name.endswith(".py"):
        module_name = module_name[:-3]
    module_path = f"tests.{module_name}"

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Error: Test '{module_name}' not found under tests/.") from exc

    if not hasattr(module, "build_setup"):
        raise SystemExit(f"Error: Test '{module_name}' missing 'build_setup()' function.")

    # Call setup (handling sync_bodies param if present)
    # We default sync_bodies=False for pure headless performance
    setup_fn = module.build_setup
    kwargs = {}
    if "sync_bodies" in inspect.signature(setup_fn).parameters:
        kwargs["sync_bodies"] = False 

    setup = setup_fn(**kwargs)
    
    # Enforce performance defaults
    if setup.headless_kwargs is None:
        setup.headless_kwargs = {}
    setup.sync_bodies = False

    return setup


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _save_metadata(run_dir: Path, setup: SimulationSetup, args: argparse.Namespace, total_steps: int):
    voxel_count = len(setup.bodies) if setup.bodies else 0
    bond_count = len(setup.constraints) if setup.constraints else 0

    system_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_name": args.test,
        "solver": args.solver,
        "voxel_count": voxel_count,
        "bond_count": bond_count,
        "dt": setup.dt,
        "iterations": setup.iterations,
        "gravity": setup.gravity,
        "friction": setup.friction,
        "total_steps": total_steps,
    }

    solver_params = dict(setup.python_solver_params or {})
    user_metadata = dict(getattr(setup, "metadata", None) or {})

    full_data = {}
    full_data.update(system_data)
    full_data.update(solver_params)
    full_data.update(user_metadata)

    with open(run_dir / "meta_data.json", "w") as f:
        json.dump(full_data, f, indent=4, default=_json_default)

    print(f"[Metadata] Saved meta_data.json with {len(full_data)} variables.")


# ==========================================
# 2. DATA CONVERSION (Bin -> VTK)
# ==========================================

class DummyBody:
    """Helper to structure raw data for the VTK exporter"""
    def __init__(self, pos, quat, size, assembly_id):
        self.pos = pos
        self.quat = quat
        self.size = size
        self.assembly_id = assembly_id
        
    @property
    def position(self): return np.concatenate([self.pos, self.quat])

    def get_corners(self):
        # Calculate rotated box corners
        w, h, d = self.size
        dx, dy, dz = w/2, h/2, d/2
        local = np.array([
            [ dx, dy, dz], [-dx, dy, dz], [-dx,-dy, dz], [ dx,-dy, dz],
            [ dx, dy,-dz], [-dx, dy,-dz], [-dx,-dy,-dz], [ dx,-dy,-dz]
        ])
        q_w, q_vec = self.quat[0], self.quat[1:]
        t = 2.0 * np.cross(q_vec, local)
        rotated = local + q_w * t + np.cross(q_vec, t)
        return rotated + self.pos

def convert_results(run_dir: Path):
    raw_dir = run_dir / "raw"
    vtk_dir = run_dir / "vtk"
    
    files = sorted(list(raw_dir.glob("*.bin")))
    if not files: return

    print(f"\n[Exporter] Converting {len(files)} frames to VTK...")
    vtk_dir.mkdir(parents=True, exist_ok=True)
    exporter = VTKExporter(str(vtk_dir))

    energy_files = sorted(list(raw_dir.glob("energy_*.csv")))
    for csv_path in energy_files:
        shutil.copy2(csv_path, vtk_dir / csv_path.name)

    
    # Reuse buffers to reduce allocs
    dt_head = np.dtype('i4,i4,f4') # n_bodies, n_bonds, dt
    dt_body = np.dtype([('p','3f4'),('q','4f4'),('s','3f4'),('id','i4'),('str','6f4')])

    for f_path in tqdm(files, unit="frame"):
        with open(f_path, "rb") as f:
            head = np.frombuffer(f.read(12), dtype=dt_head, count=1)[0]
            n_bod, n_bnd = head[0], head[1]

            # Read Bodies
            raw_b = np.frombuffer(f.read(n_bod * 68), dtype=dt_body)
            bodies = [DummyBody(b['p'], b['q'], b['s'], b['id']) for b in raw_b]
            stress = raw_b['str'] # (N, 6)

            # Read Bonds
            bonds = []
            if n_bnd > 0:
                remaining = f_path.stat().st_size - f.tell()
                if remaining % n_bnd != 0:
                    raise ValueError(f"Unsupported bond payload in {f_path.name}.")
                bytes_per_bond = remaining // n_bnd
                bond_bytes = f.read(n_bnd * bytes_per_bond)
                if bytes_per_bond == 16:
                    dt_bond = np.dtype([('a','i4'),('b','i4'),('max','f4'),('cur','f4')])
                    raw_bd = np.frombuffer(bond_bytes, dtype=dt_bond)
                    bonds = np.column_stack((raw_bd['a'], raw_bd['b'], raw_bd['cur'], raw_bd['max']))
                elif bytes_per_bond == 20:
                    dt_bond = np.dtype([('a','i4'),('b','i4'),('max','f4'),('cur','f4'),('damage','f4')])
                    raw_bd = np.frombuffer(bond_bytes, dtype=dt_bond)
                    bonds = np.column_stack((raw_bd['a'], raw_bd['b'], raw_bd['cur'], raw_bd['max'], raw_bd['damage']))
                elif bytes_per_bond == 24:
                    dt_bond = np.dtype([
                        ('idxA', 'i4'),
                        ('idxB', 'i4'),
                        ('max_strain', 'f4'),
                        ('curr_strain', 'f4'),
                        ('tensile', 'f4'),
                        ('compression', 'f4'),
                    ])
                    raw_bd = np.frombuffer(bond_bytes, dtype=dt_bond)
                    bonds = np.column_stack((
                        raw_bd['idxA'],
                        raw_bd['idxB'],
                        raw_bd['curr_strain'],
                        raw_bd['max_strain'],
                        raw_bd['tensile'],
                        raw_bd['compression'],
                    ))
                elif bytes_per_bond == 32:
                    dt_bond = np.dtype([
                        ('idxA', 'i4'),
                        ('idxB', 'i4'),
                        ('max_strain', 'f4'),
                        ('curr_strain', 'f4'),
                        ('damage', 'f4'),
                        ('k_n', 'f4'),
                        ('k_t1', 'f4'),
                        ('k_t2', 'f4'),
                    ])
                    raw_bd = np.frombuffer(bond_bytes, dtype=dt_bond)
                    bonds = np.column_stack((
                        raw_bd['idxA'],
                        raw_bd['idxB'],
                        raw_bd['curr_strain'],
                        raw_bd['max_strain'],
                        raw_bd['damage'],
                        raw_bd['k_n'],
                        raw_bd['k_t1'],
                        raw_bd['k_t2'],
                    ))
                else:
                    raise ValueError(f"Unsupported bond record size {bytes_per_bond} in {f_path.name}.")

            exporter.export(bodies, stress, bonds)


# ==========================================
# 3. MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test", help="Test name (e.g. ISO_20753)")
    parser.add_argument("--solver", default="hybrid", help="Solver backend")
    parser.add_argument("--keep-raw", action="store_true", help="Keep binary files")
    args = parser.parse_args()

    # 1. Load
    setup = _load_setup(args.test)
    run_dir = _get_save_dir(args.test)
    raw_dir = run_dir / "raw"
    
    print(f"--- Running {args.test} in {run_dir} ---")

    # 2. Configure headless run and save metadata
    steps = setup.headless_steps or 1000
    kwargs = setup.headless_kwargs or {}

    _save_metadata(run_dir, setup, args, total_steps=steps)

    # 3. Run Simulation
    solver = build_solver(setup, solver_type=args.solver)
    
    run_headless(
        solver, 
        num_steps=steps, 
        export_dir=str(raw_dir), 
        **kwargs
    )

    # 4. Export
    convert_results(run_dir)

    # 5. Clean
    if not args.keep_raw and raw_dir.exists():
        shutil.rmtree(raw_dir)
        print(f"[Exporter] VTK/VTU files are saved to: {run_dir}")
        print("[Cleanup] Binary files removed.")

if __name__ == "__main__":
    main()
