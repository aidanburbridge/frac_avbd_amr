import numpy as np
import pyvista as pv
from pathlib import Path

class VTKExporter:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0

    def export(self, bodies: list, stress_tensor: np.ndarray, bond_data: np.ndarray, active_mask=None):
        """
        Export the current state to VTK files for ParaView.
        
        Args:
            bodies: List of python Body objects (must be synced with current positions)
            stress_tensor: (N, 6) [XX, YY, ZZ, XY, YZ, ZX] from solver
            bond_data: (M, 4..8) [IdxA, IdxB, Strain, MaxStrain, (Damage | Tensile, Compression[, Damage] | Damage, K_eff[3])] from solver
            active_mask: Optional boolean mask (len == bodies) to filter to active bodies and
                remap bond indices to the filtered list.
        """

        if active_mask is not None:
            mask = np.asarray(active_mask, dtype=bool)
            if mask.shape[0] != len(bodies):
                raise ValueError("active_mask must match the length of bodies.")

            if not np.all(mask):
                active_idx = np.flatnonzero(mask)
                bodies = [bodies[i] for i in active_idx]

                stress_tensor = np.asarray(stress_tensor)
                if stress_tensor.ndim == 2 and stress_tensor.shape[0] == mask.shape[0]:
                    stress_tensor = stress_tensor[active_idx]

                bond_arr = np.asarray(bond_data)
                if bond_arr.ndim == 2 and bond_arr.shape[0] > 0:
                    index_map = {old: new for new, old in enumerate(active_idx)}
                    remapped = []
                    for row in bond_arr:
                        idxA, idxB = int(row[0]), int(row[1])
                        if idxA in index_map and idxB in index_map:
                            new_row = row.copy()
                            new_row[0] = index_map[idxA]
                            new_row[1] = index_map[idxB]
                            remapped.append(new_row)
                    bond_data = np.array(remapped, dtype=bond_arr.dtype) if remapped else np.zeros((0, bond_arr.shape[1]), dtype=bond_arr.dtype)

        # --- 1. Export Voxels (Unstructured Grid) ---
        points = []
        cells = []
        
        # Data Arrays
        tensors = [] # Full 9-component tensor for ParaView
        ids = []
        
        offset = 0
        for i, b in enumerate(bodies):
            # Geometry (8 corners per box)
            corners = b.get_corners() 
            points.append(corners)
            
            # Topology (Hexahedron cell)
            # [8, p0, p1, ..., p7]
            cell_ids = np.arange(offset, offset+8)
            cells.append(np.concatenate(([8], cell_ids)))
            offset += 8
            
            # Stress Data
            # Map symmetric 6-comp [XX, YY, ZZ, XY, YZ, ZX] 
            # to full 9-comp [XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ]
            if i < len(stress_tensor):
                xx, yy, zz, xy, yz, zx = stress_tensor[i]
                tensors.append([xx, xy, zx,  xy, yy, yz,  zx, yz, zz])
            else:
                tensors.append([0]*9)
            
            ids.append(getattr(b, 'assembly_id', -1))

        # Create Grid
        if points:
            points = np.vstack(points)
            cells = np.hstack(cells)
            grid = pv.UnstructuredGrid(cells, np.array([pv.CellType.HEXAHEDRON]*len(bodies)), points)
            
            # Attach Data
            grid.cell_data["Assembly_ID"] = np.array(ids)
            grid.cell_data["Stress_Tensor"] = np.array(tensors)
            
            # Save
            filename = self.save_dir / f"voxels_{self.frame_count:04d}.vtu"
            grid.save(filename)

        # --- 2. Export Bonds (Lines) ---
        if len(bond_data) > 0:
            line_pts = []
            lines = []
            strains = []
            max_strains = []
            tensile_strains = []
            compressive_strains = []
            damages = []
            eff_stiffness = []
            
            # Pre-calculate centers for speed
            centers = np.array([b.position[:3] for b in bodies])
            
            pt_off = 0
            for row in bond_data:
                idxA, idxB = int(row[0]), int(row[1])
                curr = row[2] if len(row) > 2 else 0.0
                max_s = row[3] if len(row) > 3 else 0.0
                tensile = 0.0
                compressive = 0.0
                damage = 0.0
                k_eff = (0.0, 0.0, 0.0)
                if len(row) >= 8:
                    damage = row[4]
                    k_eff = (row[5], row[6], row[7])
                elif len(row) >= 7:
                    tensile = row[4]
                    compressive = row[5]
                    damage = row[6]
                elif len(row) >= 6:
                    tensile = row[4]
                    compressive = row[5]
                elif len(row) >= 5:
                    damage = row[4]
                
                # Safety check
                if idxA < len(centers) and idxB < len(centers):
                    line_pts.append(centers[idxA])
                    line_pts.append(centers[idxB])
                    # Line connection: [2 points, idx1, idx2]
                    lines.append([2, pt_off, pt_off+1])
                    pt_off += 2
                    
                    strains.append(curr)
                    max_strains.append(max_s)
                    tensile_strains.append(tensile)
                    compressive_strains.append(compressive)
                    damages.append(damage)
                    eff_stiffness.append(k_eff)
            
            if line_pts:
                # Build a PolyData without the default per-point vertices so cell
                # counts line up with our bonds-only data arrays.
                poly = pv.PolyData()
                poly.points = np.array(line_pts)
                poly.lines = np.hstack(lines)
                poly.cell_data["Strain"] = np.array(strains)
                poly.cell_data["Max_Strain"] = np.array(max_strains)
                poly.cell_data["Tensile_Strain"] = np.array(tensile_strains)
                poly.cell_data["Compression_Strain"] = np.array(compressive_strains)
                poly.cell_data["Damage"] = np.array(damages)
                poly.cell_data["Effective_Stiffness"] = np.array(eff_stiffness)
                
                filename = self.save_dir / f"bonds_{self.frame_count:04d}.vtp"
                poly.save(filename)

        self.frame_count += 1
