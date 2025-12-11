import argparse
import struct
import numpy as np
from pathlib import Path
from tqdm import tqdm
from util.vtk_exporter import VTKExporter

# We need a dummy object to mimic the Body class for the exporter
class DummyBody:
    def __init__(self, pos, quat, size, assembly_id):
        self.pos = pos
        self.quat = quat
        self.size = size
        self.assembly_id = assembly_id
        
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
        # --- HEADER ---
        # 2 Int32 (N_bodies, N_bonds), 1 Float32 (dt)
        header = f.read(12) 
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
        # Layout: IdxA(1i), IdxB(1i), MaxStrain(1f), CurrStrain(1f) = 16 bytes
        bond_bytes = f.read(n_bonds * 16)
        

        bond_export = []
        if n_bonds > 0:
            dt_bond = np.dtype([('idxA', 'i4'), ('idxB', 'i4'), ('max_strain', 'f4'), ('curr_strain', 'f4')])
            raw_bonds = np.frombuffer(bond_bytes, dtype=dt_bond)

            # Exporter expects: [idxA, idxB, current_strain, max_strain]
            bond_export = np.column_stack((
                raw_bonds['idxA'],
                raw_bonds['idxB'],
                raw_bonds['curr_strain'],
                raw_bonds['max_strain']
            ))

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
    exporter = VTKExporter(str(vtk_dir))
    
    files = sorted(list(raw_dir.glob("*.bin")))
    print(f"Processing {len(files)} frames...")
    
    for f in tqdm(files):
        process_frame(f, exporter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to run folder (e.g. output/double_beam/run_01)")
    args = parser.parse_args()
    
    process_run(args.run_dir)
