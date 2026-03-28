import struct
from pathlib import Path

import numpy as np
import pyvista as pv


BOND_META_MAGIC_V1 = b"ABM1"
BOND_META_MAGIC_V2 = b"ABM2"
_BOND_META_DTYPE_V1 = np.dtype(
    [
        ("bond_id", "i4"),
        ("bodyA_id", "i4"),
        ("bodyB_id", "i4"),
        ("area", "f4"),
    ]
)
_BOND_META_DTYPE_V2 = np.dtype(
    [
        ("bond_id", "i4"),
        ("bodyA_id", "i4"),
        ("bodyB_id", "i4"),
        ("area", "f4"),
        ("f_min", "3f4"),
        ("f_max", "3f4"),
    ]
)


def _load_bond_meta(path: str | Path | None) -> dict[int, dict[str, np.ndarray | float | None]]:
    if path is None:
        return {}

    meta_path = Path(path)
    if not meta_path.exists():
        return {}

    with meta_path.open("rb") as f:
        magic = f.read(4)
        count = struct.unpack("i", f.read(4))[0]
        if magic == BOND_META_MAGIC_V2:
            raw = np.frombuffer(f.read(count * _BOND_META_DTYPE_V2.itemsize), dtype=_BOND_META_DTYPE_V2)
            return {
                int(row["bond_id"]): {
                    "area": float(row["area"]),
                    "f_min": np.asarray(row["f_min"], dtype=float),
                    "f_max": np.asarray(row["f_max"], dtype=float),
                }
                for row in raw
            }
        if magic == BOND_META_MAGIC_V1:
            raw = np.frombuffer(f.read(count * _BOND_META_DTYPE_V1.itemsize), dtype=_BOND_META_DTYPE_V1)
            return {
                int(row["bond_id"]): {
                    "area": float(row["area"]),
                    "f_min": None,
                    "f_max": None,
                }
                for row in raw
            }

    raise ValueError(f"Unsupported bond metadata format in {meta_path}")


def _stress_proxy_fields(stress_tensor: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    stress = np.asarray(stress_tensor, dtype=float)
    if stress.ndim != 2 or stress.shape[0] == 0:
        empty_tensor = np.zeros((0, 9), dtype=np.float32)
        empty_scalar = np.zeros((0,), dtype=np.float32)
        return empty_tensor, {
            "Max_Principal_Stress_Proxy": empty_scalar.copy(),
            "Von_Mises_Stress_Proxy": empty_scalar.copy(),
            "Hydrostatic_Stress_Proxy": empty_scalar.copy(),
            "Deviatoric_Stress_Norm_Proxy": empty_scalar.copy(),
        }

    xx = stress[:, 0]
    yy = stress[:, 1]
    zz = stress[:, 2]
    xy = stress[:, 3]
    yz = stress[:, 4]
    zx = stress[:, 5]

    full9 = np.column_stack((xx, xy, zx, xy, yy, yz, zx, yz, zz)).astype(np.float32, copy=False)

    mats = np.zeros((stress.shape[0], 3, 3), dtype=float)
    mats[:, 0, 0] = xx
    mats[:, 1, 1] = yy
    mats[:, 2, 2] = zz
    mats[:, 0, 1] = mats[:, 1, 0] = xy
    mats[:, 1, 2] = mats[:, 2, 1] = yz
    mats[:, 0, 2] = mats[:, 2, 0] = zx
    max_principal = np.linalg.eigvalsh(mats)[:, -1]

    hydrostatic = (xx + yy + zz) / 3.0
    sxx = xx - hydrostatic
    syy = yy - hydrostatic
    szz = zz - hydrostatic
    deviatoric_norm = np.sqrt(
        np.square(sxx)
        + np.square(syy)
        + np.square(szz)
        + 2.0 * (np.square(xy) + np.square(yz) + np.square(zx))
    )
    von_mises = np.sqrt(1.5) * deviatoric_norm

    return full9, {
        "Max_Principal_Stress_Proxy": max_principal.astype(np.float32, copy=False),
        "Von_Mises_Stress_Proxy": von_mises.astype(np.float32, copy=False),
        "Hydrostatic_Stress_Proxy": hydrostatic.astype(np.float32, copy=False),
        "Deviatoric_Stress_Norm_Proxy": deviatoric_norm.astype(np.float32, copy=False),
    }


def _bond_proxy_fields(
    bond_arr: np.ndarray,
    bond_meta: dict[int, dict[str, np.ndarray | float | None]],
) -> dict[str, np.ndarray]:
    if bond_arr.ndim != 2 or bond_arr.shape[0] == 0 or bond_arr.shape[1] < 15:
        return {}

    n = int(bond_arr.shape[0])
    eps = np.finfo(float).eps

    bond_ids = bond_arr[:, 0].astype(np.int32)
    C = np.asarray(bond_arr[:, 3:6], dtype=float)
    rest = np.asarray(bond_arr[:, 6:9], dtype=float)
    penalty_k = np.asarray(bond_arr[:, 9:12], dtype=float)

    area = np.full((n,), np.nan, dtype=float)
    f_min = np.full((n, 3), np.nan, dtype=float)
    f_max = np.full((n, 3), np.nan, dtype=float)
    for i, bond_id in enumerate(bond_ids):
        rec = bond_meta.get(int(bond_id))
        if rec is None:
            continue
        area[i] = float(rec["area"])
        if rec["f_min"] is None or rec["f_max"] is None:
            continue
        f_min[i, :] = np.asarray(rec["f_min"], dtype=float)
        f_max[i, :] = np.asarray(rec["f_max"], dtype=float)

    L0 = np.linalg.norm(rest, axis=1)
    normal_sign = np.where(np.abs(rest[:, 0]) > eps, np.sign(rest[:, 0]), 1.0)
    safe_length = np.where(L0 > eps, L0, np.nan)
    eps_n = normal_sign * C[:, 0] / safe_length
    gamma_t1 = C[:, 1] / safe_length
    gamma_t2 = C[:, 2] / safe_length
    gamma_eq = np.sqrt(np.square(gamma_t1) + np.square(gamma_t2))

    safe_area = np.where(area > eps, area, np.nan)
    caps_mask = np.isfinite(safe_area) & np.all(np.isfinite(f_min), axis=1) & np.all(np.isfinite(f_max), axis=1)

    row_force = np.full((n, 3), np.nan, dtype=float)
    sigma_n = np.full((n,), np.nan, dtype=float)
    tau_t1 = np.full((n,), np.nan, dtype=float)
    tau_t2 = np.full((n,), np.nan, dtype=float)
    tau_eq = np.full((n,), np.nan, dtype=float)
    normal_util = np.full((n,), np.nan, dtype=float)
    shear_util = np.full((n,), np.nan, dtype=float)
    mixed_mode_util = np.full((n,), np.nan, dtype=float)

    if np.any(caps_mask):
        row_force[caps_mask] = np.clip(penalty_k[caps_mask] * C[caps_mask], f_min[caps_mask], f_max[caps_mask])
        sigma_n[caps_mask] = row_force[caps_mask, 0] / safe_area[caps_mask]
        tau_t1[caps_mask] = row_force[caps_mask, 1] / safe_area[caps_mask]
        tau_t2[caps_mask] = row_force[caps_mask, 2] / safe_area[caps_mask]
        tau_eq[caps_mask] = np.sqrt(np.square(tau_t1[caps_mask]) + np.square(tau_t2[caps_mask]))

        normal_cap = np.where(
            normal_sign[caps_mask] >= 0.0,
            np.maximum(f_max[caps_mask, 0], eps),
            np.maximum(-f_min[caps_mask, 0], eps),
        )
        shear_cap_t1 = np.maximum(np.maximum(np.abs(f_min[caps_mask, 1]), np.abs(f_max[caps_mask, 1])), eps)
        shear_cap_t2 = np.maximum(np.maximum(np.abs(f_min[caps_mask, 2]), np.abs(f_max[caps_mask, 2])), eps)
        fn_open = np.maximum(0.0, normal_sign[caps_mask] * row_force[caps_mask, 0])
        ft1 = np.abs(row_force[caps_mask, 1])
        ft2 = np.abs(row_force[caps_mask, 2])

        normal_util[caps_mask] = fn_open / normal_cap
        shear_util[caps_mask] = np.sqrt(np.square(ft1 / shear_cap_t1) + np.square(ft2 / shear_cap_t2))
        mixed_mode_util[caps_mask] = np.sqrt(np.square(normal_util[caps_mask]) + np.square(shear_util[caps_mask]))

    return {
        "Area": area.astype(np.float32, copy=False),
        "L0": L0.astype(np.float32, copy=False),
        "Eps_N": eps_n.astype(np.float32, copy=False),
        "Gamma_T1": gamma_t1.astype(np.float32, copy=False),
        "Gamma_T2": gamma_t2.astype(np.float32, copy=False),
        "Gamma_Eq": gamma_eq.astype(np.float32, copy=False),
        "Row_Force_Local": row_force.astype(np.float32, copy=False),
        "Sigma_N": sigma_n.astype(np.float32, copy=False),
        "Tau_T1": tau_t1.astype(np.float32, copy=False),
        "Tau_T2": tau_t2.astype(np.float32, copy=False),
        "Tau_Eq": tau_eq.astype(np.float32, copy=False),
        "Normal_Traction_Utilization": normal_util.astype(np.float32, copy=False),
        "Shear_Traction_Utilization": shear_util.astype(np.float32, copy=False),
        "Mixed_Mode_Traction_Utilization": mixed_mode_util.astype(np.float32, copy=False),
    }


class VTKExporter:
    def __init__(self, save_dir: str, bond_meta_path: str | Path | None = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0
        self.bond_meta = _load_bond_meta(bond_meta_path)

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
                    bond_data = (
                        np.array(remapped, dtype=bond_arr.dtype)
                        if remapped
                        else np.zeros((0, bond_arr.shape[1]), dtype=bond_arr.dtype)
                    )

        points = []
        cells = []
        assembly_ids = []
        body_ids = []
        stress_array = np.asarray(stress_tensor, dtype=float)
        if stress_array.ndim != 2 or stress_array.shape[1] != 6:
            stress_array = np.zeros((len(bodies), 6), dtype=float)
        elif stress_array.shape[0] < len(bodies):
            padded = np.zeros((len(bodies), 6), dtype=float)
            padded[: stress_array.shape[0], :] = stress_array
            stress_array = padded
        elif stress_array.shape[0] > len(bodies):
            stress_array = stress_array[: len(bodies), :]

        offset = 0
        for i, b in enumerate(bodies):
            corners = b.get_corners()
            points.append(corners)

            cell_ids = np.arange(offset, offset + 8)
            cells.append(np.concatenate(([8], cell_ids)))
            offset += 8

            assembly_ids.append(getattr(b, "assembly_id", -1))
            body_id = getattr(b, "body_id", None)
            body_ids.append(i if body_id is None else body_id)

        if points:
            points = np.vstack(points)
            cells = np.hstack(cells)
            grid = pv.UnstructuredGrid(cells, np.array([pv.CellType.HEXAHEDRON] * len(bodies)), points)

            full_stress_tensor, stress_proxies = _stress_proxy_fields(stress_array)
            grid.cell_data["Assembly_ID"] = np.array(assembly_ids)
            grid.cell_data["Body_ID"] = np.array(body_ids)
            grid.cell_data["Stress_Tensor"] = full_stress_tensor
            for name, values in stress_proxies.items():
                grid.cell_data[name] = values

            filename = self.save_dir / f"voxels_{self.frame_count:04d}.vtu"
            grid.save(filename)

        if len(bond_data) > 0:
            line_pts = []
            lines = []
            centers = np.array([b.position[:3] for b in bodies])
            body_id_to_local = {
                int(i if getattr(b, "body_id", None) is None else getattr(b, "body_id")): i
                for i, b in enumerate(bodies)
            }

            pt_off = 0
            bond_arr = np.asarray(bond_data)
            raw_v2 = bond_arr.ndim == 2 and bond_arr.shape[1] >= 15

            if raw_v2:
                kept_rows = []
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
                    lines.append([2, pt_off, pt_off + 1])
                    pt_off += 2

                    kept_rows.append(np.asarray(row, dtype=float))
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
                        lines.append([2, pt_off, pt_off + 1])
                        pt_off += 2

                        strains.append(curr)
                        max_strains.append(max_s)
                        tensile_strains.append(tensile)
                        compressive_strains.append(compressive)
                        damages.append(damage)
                        eff_stiffness.append(k_eff)

            if line_pts:
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
                    kept_arr = np.vstack(kept_rows) if kept_rows else np.zeros((0, bond_arr.shape[1]), dtype=float)
                    for name, values in _bond_proxy_fields(kept_arr, self.bond_meta).items():
                        poly.cell_data[name] = values
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
