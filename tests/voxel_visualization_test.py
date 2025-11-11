# Fast voxelize-and-view with PyVista (no argparse)
import os
import numpy as np
import pyvista as pv
import trimesh
from tqdm import trange

# ---- CONFIG ----
#STL_PATH = r"C:\Users\aidan\Documents\TUM\Thesis\CactusPot.stl"  # leave blank to autogen cube
#STL_PATH = r"C:\Users\aidan\Documents\TUM\Thesis\simple_plane.stl"
STL_PATH = r"C:\Users\aidan\Documents\TUM\Thesis\10x10x50_beam.stl"

TARGET_VOXELS = 1000     # approximate occupied voxels
VOXEL_H = None            # set a float to force fixed voxel size (edge length)
PAD = 1                   # padding voxels around AABB
FLOOD_FILL = True         # fill interior cavities
REPAIR_MESH = True       # trimesh repair on import
# -----------------

# your modules
import geometry.octree as oct
from geometry.voxelizer import STLVoxelizer

def generate_cube_stl(path, size=1.0):
    box = trimesh.creation.box(extents=(size, size, size))
    box.export(path)
    return path

def uniform_grid_from_occ(occ: np.ndarray, origin, h):
    """
    Build a VTK UniformGrid from a boolean occupancy grid.
    We store 'occ' as CELL data (one scalar per voxel).
    VTK expects grid.dimensions = (#points) = (#cells + 1) along each axis.
    """
    nx, ny, nz = occ.shape
    try:
        grid = pv.UniformGrid()
    except AttributeError:
        # Some builds expose it as ImageData instead of UniformGrid
        from pyvista import ImageData
        grid = ImageData()
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        grid.origin = origin  # (x0, y0, z0)
        grid.spacing = (h, h, h)

    # Cell data must be length nx*ny*nz, Fortran-ordered for VTK
    grid.cell_data.clear()
    grid.cell_data["occ"] = occ.astype(np.uint8).ravel(order="F")
    return grid

def main():
    stl_path = STL_PATH if STL_PATH else ""
    if not stl_path or not os.path.isfile(stl_path):
        stl_path = os.path.abspath("test_cube.stl")
        print(f"[info] generating cube at {stl_path}")
        generate_cube_stl(stl_path, size=1.0)

    # --- voxelize STL ---
    vx = STLVoxelizer(
        stl_path,
        pad_voxels=int(PAD),
        flood_fill=bool(FLOOD_FILL),
        repair=bool(REPAIR_MESH),
    )

    if VOXEL_H is not None:
        occ, origin, h = vx.voxelize_to_h(float(VOXEL_H))
    else:
        occ, origin, h = vx.voxelize_to_resolution(int(TARGET_VOXELS))

    stats = vx.stats()
    print(f"[voxelizer] voxels={stats['voxels']}, grid={stats['grid_shape']}, h={stats['h']:.6f}")

    # --- boxes (if you still need them for sim) ---
    leaves, h_base = oct.octree_from_occ(occ, h_base=stats["h"])
    boxes, mapping = oct.instantiate_boxes_from_tree(
        leaves, origin=stats["origin"], h_base=h_base,
        density=1.0, penalty_gain=1e5, static=True
    )
    print(f"[octree] leaves={len(leaves)}, boxes={len(boxes)}")
    

    # --- build VTK grids/meshes for display ---
    # Left: original STL
    stl_mesh = pv.read(stl_path)  # fast, native VTK loader

    # Right: voxel surface from UniformGrid threshold
    grid = uniform_grid_from_occ(occ, origin=stats["origin"], h=stats["h"])
    vox_surf = grid.threshold(0.5, scalars="occ").extract_surface()  # isosurface of occupied cells

    # --- PyVista viewer (side-by-side) ---
    pv.set_plot_theme("document")
    plotter = pv.Plotter(shape=(1, 2), border=False, window_size=(1300, 700))

    # Left panel: STL
    plotter.subplot(0, 0)
    plotter.add_mesh(stl_mesh, opacity=0.35, color="lightsteelblue", show_edges=True)
    plotter.add_axes()
    plotter.enable_anti_aliasing("msaa")
    plotter.camera_position = "iso"

    # Right panel: voxelized surface
    plotter.subplot(0, 1)
    plotter.add_mesh(vox_surf, color="royalblue", opacity=0.6, show_edges=False)
    plotter.add_axes()
    plotter.enable_anti_aliasing("msaa")
    plotter.camera_position = "iso"

    plotter.link_views()  # sync cameras
    plotter.show(title="STL vs Voxelized (PyVista)")

if __name__ == "__main__":
    main()
