import argparse
import struct
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
from util.vtk_exporter import VTKExporter

FRAME_MAGIC = b"AVB2"

# We need a dummy object to mimic the Body class for the exporter
class DummyBody:
    def __init__(self, pos, quat, size, assembly_id, body_id=None):
        self.pos = pos
        self.quat = quat
        self.size = size
        self.assembly_id = assembly_id
        self.body_id = int(body_id) if body_id is not None else None
        
    @property
    def position(self):
        # Return concatenated pos+quat for compatibility if needed
        return np.concatenate([self.pos, self.quat])

    def get_corners(self):
        # Reconstruct corners from Pos, Quat, Size
        w, h, d = self.size
        # Local corners (unrotated)
        # Pattern: (+ + +), (- + +), ... standard box corners
        dx, dy, dz = w/2, h/2, d/2
        local_corners = np.array([
            [ dx,  dy,  dz], [-dx,  dy,  dz], [-dx, -dy,  dz], [ dx, -dy,  dz],
            [ dx,  dy, -dz], [-dx,  dy, -dz], [-dx, -dy, -dz], [ dx, -dy, -dz]
        ])
        
        # Rotate
        # q = [w, x, y, z]
        q_w = self.quat[0]
        q_vec = self.quat[1:]
        
        # Optimized rotation: v + 2*cross(q_vec, cross(q_vec, v) + q_w*v)
        # Use numpy broadcasting for all 8 corners at once
        t = 2.0 * np.cross(q_vec, local_corners)
        rotated = local_corners + q_w * t + np.cross(q_vec, t)
        
        return rotated + self.pos

def process_frame(file_path, exporter):
    with open(file_path, "rb") as f:
        prefix = f.read(4)
        if prefix == FRAME_MAGIC:
            n_bodies, n_bonds, dt = struct.unpack("iif", f.read(12))

            dt_body = np.dtype([
                ('pos', '3f4'),
                ('quat', '4f4'),
                ('size', '3f4'),
                ('body_id', 'i4'),
                ('assembly_id', 'i4'),
                ('stress', '6f4'),
            ])
            raw_bodies = np.frombuffer(f.read(n_bodies * dt_body.itemsize), dtype=dt_body)

            bodies = []
            stress_tensor = np.zeros((n_bodies, 6), dtype=np.float32)
            for idx, b in enumerate(raw_bodies):
                bodies.append(DummyBody(b['pos'], b['quat'], b['size'], b['assembly_id'], body_id=b['body_id']))
                stress_tensor[idx] = b['stress']

            bond_export = []
            if n_bonds > 0:
                dt_bond = np.dtype([
                    ('bond_id', 'i4'),
                    ('bodyA_id', 'i4'),
                    ('bodyB_id', 'i4'),
                    ('C', '3f4'),
                    ('rest', '3f4'),
                    ('penalty_k', '3f4'),
                    ('damage', 'f4'),
                    ('is_broken', 'u1'),
                    ('is_cohesive', 'u1'),
                    ('_pad', 'u2'),
                ])
                raw_bonds = np.frombuffer(f.read(n_bonds * dt_bond.itemsize), dtype=dt_bond)
                bond_export = np.column_stack((
                    raw_bonds['bond_id'],
                    raw_bonds['bodyA_id'],
                    raw_bonds['bodyB_id'],
                    raw_bonds['C'],
                    raw_bonds['rest'],
                    raw_bonds['penalty_k'],
                    raw_bonds['damage'],
                    raw_bonds['is_broken'],
                    raw_bonds['is_cohesive'],
                ))

            exporter.export(bodies, stress_tensor, bond_export)
            return

        # --- HEADER ---
        # 2 Int32 (N_bodies, N_bonds), 1 Float32 (dt)
        header = prefix + f.read(8)
        if not header: return
        n_bodies, n_bonds, dt = struct.unpack("iif", header)
        
        # --- BODIES ---
        # Layout: Pos(3f) + Quat(4f) + Size(3f) + ID(1i) + Stress(6f) = 68 bytes

        body_bytes = f.read(n_bodies * 68)

        # Use numpy struct array for fast parsing
        dt_body = np.dtype([
            ('pos', '3f4'),
            ('quat', '4f4'),
            ('size', '3f4'),
            ('id', 'i4'),
            ('stress', '6f4')
        ])

        raw_bodies = np.frombuffer(body_bytes, dtype=dt_body)
        
        bodies = []
        ids = []
        stress_tensor = np.zeros((n_bodies, 6), dtype=np.float32)

        for idx, b in enumerate(raw_bodies):
            bodies.append(DummyBody(b['pos'], b['quat'], b['size'], b['id']))
            ids.append(b['id'])
            stress_tensor[idx] = b['stress']
        # --- BONDS ---
        bond_export = []
        if n_bonds > 0:
            remaining = Path(file_path).stat().st_size - f.tell()
            if remaining % n_bonds != 0:
                raise ValueError(f"Unsupported bond payload in {Path(file_path).name}.")
            bytes_per_bond = remaining // n_bonds
            bond_bytes = f.read(n_bonds * bytes_per_bond)
            if bytes_per_bond == 16:
                dt_bond = np.dtype([
                    ('idxA', 'i4'),
                    ('idxB', 'i4'),
                    ('max_strain', 'f4'),
                    ('curr_strain', 'f4'),
                ])
                raw_bonds = np.frombuffer(bond_bytes, dtype=dt_bond)
                bond_export = np.column_stack((
                    raw_bonds['idxA'],
                    raw_bonds['idxB'],
                    raw_bonds['curr_strain'],
                    raw_bonds['max_strain'],
                ))
            elif bytes_per_bond == 20:
                dt_bond = np.dtype([
                    ('idxA', 'i4'),
                    ('idxB', 'i4'),
                    ('max_strain', 'f4'),
                    ('curr_strain', 'f4'),
                    ('damage', 'f4'),
                ])
                raw_bonds = np.frombuffer(bond_bytes, dtype=dt_bond)
                bond_export = np.column_stack((
                    raw_bonds['idxA'],
                    raw_bonds['idxB'],
                    raw_bonds['curr_strain'],
                    raw_bonds['max_strain'],
                    raw_bonds['damage'],
                ))
            elif bytes_per_bond == 24:
                dt_bond = np.dtype([
                    ('idxA', 'i4'),
                    ('idxB', 'i4'),
                    ('max_strain', 'f4'),
                    ('curr_strain', 'f4'),
                    ('tensile', 'f4'),
                    ('compression', 'f4'),
                ])
                raw_bonds = np.frombuffer(bond_bytes, dtype=dt_bond)
                bond_export = np.column_stack((
                    raw_bonds['idxA'],
                    raw_bonds['idxB'],
                    raw_bonds['curr_strain'],
                    raw_bonds['max_strain'],
                    raw_bonds['tensile'],
                    raw_bonds['compression'],
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
                raw_bonds = np.frombuffer(bond_bytes, dtype=dt_bond)
                bond_export = np.column_stack((
                    raw_bonds['idxA'],
                    raw_bonds['idxB'],
                    raw_bonds['curr_strain'],
                    raw_bonds['max_strain'],
                    raw_bonds['damage'],
                    raw_bonds['k_n'],
                    raw_bonds['k_t1'],
                    raw_bonds['k_t2'],
                ))
            else:
                raise ValueError(f"Unsupported bond record size {bytes_per_bond} in {Path(file_path).name}.")

        # Export
        exporter.export(bodies, stress_tensor, bond_export)

def process_run(run_dir_str):
    run_dir = Path(run_dir_str)
    raw_dir = run_dir / "raw"
    vtk_dir = run_dir / "vtk"
    
    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist.")
        return

    vtk_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize exporter
    exporter = VTKExporter(str(vtk_dir), bond_meta_path=raw_dir / "bond_meta.bin")
    
    files = sorted(list(raw_dir.glob("frame_*.bin")))
    print(f"Processing {len(files)} frames...")

    # Preserve per-frame energy ledgers if they were written by the solver.
    for csv_path in sorted(raw_dir.glob("energy_*.csv")):
        shutil.copy2(csv_path, vtk_dir / csv_path.name)
    step_metrics_path = raw_dir / "step_metrics.csv"
    if step_metrics_path.exists():
        shutil.copy2(step_metrics_path, vtk_dir / step_metrics_path.name)
    bond_meta_path = raw_dir / "bond_meta.bin"
    if bond_meta_path.exists():
        shutil.copy2(bond_meta_path, vtk_dir / bond_meta_path.name)
    
    for f in tqdm(files):
        process_frame(f, exporter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to run folder (e.g. output/double_beam/run_01)")
    args = parser.parse_args()
    
    process_run(args.run_dir)
