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
            bond_data:
                - legacy (M, 4..8): [IdxA, IdxB, Strain, MaxStrain, ...]
                - raw v2 (M, 15): [BondID, BodyA_ID, BodyB_ID, C1, C2, C3, Rest1, Rest2, Rest3,
                                   PenaltyK1, PenaltyK2, PenaltyK3, Damage, IsBroken, IsCohesive]
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
                    if bond_arr.shape[1] >= 15:
                        active_body_ids = {
                            int(old if getattr(bodies[new], "body_id", None) is None else getattr(bodies[new], "body_id"))
                            for new, old in enumerate(active_idx)
                        }
                        remapped = [
                            row.copy()
                            for row in bond_arr
                            if int(row[1]) in active_body_ids and int(row[2]) in active_body_ids
                        ]
                    else:
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
        assembly_ids = []
        body_ids = []
        
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
            
            assembly_ids.append(getattr(b, 'assembly_id', -1))
            body_id = getattr(b, 'body_id', None)
            body_ids.append(i if body_id is None else body_id)

        # Create Grid
        if points:
            points = np.vstack(points)
            cells = np.hstack(cells)
            grid = pv.UnstructuredGrid(cells, np.array([pv.CellType.HEXAHEDRON]*len(bodies)), points)
            
            # Attach Data
            grid.cell_data["Assembly_ID"] = np.array(assembly_ids)
            grid.cell_data["Body_ID"] = np.array(body_ids)
            grid.cell_data["Stress_Tensor"] = np.array(tensors)
            
            # Save
            filename = self.save_dir / f"voxels_{self.frame_count:04d}.vtu"
            grid.save(filename)

        # --- 2. Export Bonds (Lines) ---
        if len(bond_data) > 0:
            line_pts = []
            lines = []
            
            # Pre-calculate centers for speed
            centers = np.array([b.position[:3] for b in bodies])
            body_id_to_local = {
                int(i if getattr(b, "body_id", None) is None else getattr(b, "body_id")): i
                for i, b in enumerate(bodies)
            }
            
            pt_off = 0
            bond_arr = np.asarray(bond_data)
            raw_v2 = bond_arr.ndim == 2 and bond_arr.shape[1] >= 15

            if raw_v2:
                bond_ids = []
                bodyA_ids = []
                bodyB_ids = []
                c_local = []
                rest_local = []
                penalty_k = []
                damages = []
                is_broken = []
                is_cohesive = []

                for row in bond_arr:
                    bond_id = int(row[0])
                    bodyA_id = int(row[1])
                    bodyB_id = int(row[2])
                    idxA = body_id_to_local.get(bodyA_id)
                    idxB = body_id_to_local.get(bodyB_id)
                    if idxA is None or idxB is None:
                        continue

                    line_pts.append(centers[idxA])
                    line_pts.append(centers[idxB])
                    lines.append([2, pt_off, pt_off+1])
                    pt_off += 2

                    bond_ids.append(bond_id)
                    bodyA_ids.append(bodyA_id)
                    bodyB_ids.append(bodyB_id)
                    c_local.append(row[3:6])
                    rest_local.append(row[6:9])
                    penalty_k.append(row[9:12])
                    damages.append(row[12])
                    is_broken.append(int(row[13]))
                    is_cohesive.append(int(row[14]))
            else:
                strains = []
                max_strains = []
                tensile_strains = []
                compressive_strains = []
                damages = []
                eff_stiffness = []

                for row in bond_arr:
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

                    if idxA < len(centers) and idxB < len(centers):
                        line_pts.append(centers[idxA])
                        line_pts.append(centers[idxB])
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
                if raw_v2:
                    poly.cell_data["Bond_ID"] = np.array(bond_ids, dtype=np.int32)
                    poly.cell_data["BodyA_ID"] = np.array(bodyA_ids, dtype=np.int32)
                    poly.cell_data["BodyB_ID"] = np.array(bodyB_ids, dtype=np.int32)
                    poly.cell_data["C_Local"] = np.array(c_local, dtype=np.float32)
                    poly.cell_data["Rest_Local"] = np.array(rest_local, dtype=np.float32)
                    poly.cell_data["Penalty_K"] = np.array(penalty_k, dtype=np.float32)
                    poly.cell_data["Damage"] = np.array(damages, dtype=np.float32)
                    poly.cell_data["Is_Broken"] = np.array(is_broken, dtype=np.int32)
                    poly.cell_data["Is_Cohesive"] = np.array(is_cohesive, dtype=np.int32)
                else:
                    poly.cell_data["Strain"] = np.array(strains)
                    poly.cell_data["Max_Strain"] = np.array(max_strains)
                    poly.cell_data["Tensile_Strain"] = np.array(tensile_strains)
                    poly.cell_data["Compression_Strain"] = np.array(compressive_strains)
                    poly.cell_data["Damage"] = np.array(damages)
                    poly.cell_data["Effective_Stiffness"] = np.array(eff_stiffness)
                
                filename = self.save_dir / f"bonds_{self.frame_count:04d}.vtp"
                poly.save(filename)

        self.frame_count += 1
