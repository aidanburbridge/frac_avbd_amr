# Simple voxelize-and-view (click Run, no argparse)
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import trimesh

# ---- CONFIG (edit these, then click Run) ----
STL_PATH = r"C:\Users\aidan\Documents\TUM\Thesis\CactusPot.stl" # e.g., r"C:\Users\aidan\Documents\part.stl". Leave empty or None to auto-generate a cube.
TARGET_VOXELS = 10000     # approximate occupied voxels
VOXEL_H = None           # set a float (edge length) to use fixed voxel size instead of TARGET_VOXELS
PAD = 1                  # padding voxels around AABB
FLOOD_FILL = True        # True fills interior cavities, False keeps only center-in-solid voxels
REPAIR_MESH = False      # try trimesh repair on import
MAX_BOXES_TO_DRAW = 8000 # downsample viewer drawing for speed
# ---------------------------------------------

# Your modules
import geometry.octree as oct
from geometry.voxelizer import STLVoxelizer

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0]); x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0]); y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0]); z_mid = np.mean(z_limits)
    r = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - r, x_mid + r])
    ax.set_ylim3d([y_mid - r, y_mid + r])
    ax.set_zlim3d([z_mid - r, z_mid + r])

def draw_trimesh(ax, mesh, alpha=0.25):
    verts = mesh.vertices; faces = mesh.faces
    tris = [verts[f] for f in faces]
    poly = Poly3DCollection(tris, alpha=alpha, linewidths=0.2, edgecolors='k')
    ax.add_collection3d(poly)
    ax.auto_scale_xyz(verts[:,0], verts[:,1], verts[:,2])

def box_edges_from_corners(c):
    idx = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
    return [np.vstack([c[i], c[j]]) for (i,j) in idx]

def draw_boxes_wire(ax, boxes, max_boxes_to_draw=5000):
    segments = []
    n = len(boxes)
    step = max(1, n // max_boxes_to_draw)
    for b in boxes[::step]:
        c = b.get_corners()
        segments.extend(box_edges_from_corners(c))
    lc = Line3DCollection(segments, linewidths=0.4)
    ax.add_collection3d(lc)
    if len(boxes) > 0:
        c0 = boxes[0].get_corners()
        mins = np.min(c0, axis=0).copy()
        maxs = np.max(c0, axis=0).copy()
        for b in boxes[1::step]:
            c = b.get_corners()
            mins = np.minimum(mins, np.min(c, axis=0))
            maxs = np.maximum(maxs, np.max(c, axis=0))
        ax.auto_scale_xyz([mins[0], maxs[0]], [mins[1], maxs[1]], [mins[2], maxs[2]])

def draw_boxes_surface(ax, boxes, alpha=0.2, max_boxes=2000):
    step = max(1, len(boxes) // max_boxes)
    for b in boxes[::step]:
        corners = b.get_corners()
        faces = [
            [corners[j] for j in [0,1,2,3]],  # bottom
            [corners[j] for j in [4,5,6,7]],  # top
            [corners[j] for j in [0,1,5,4]],
            [corners[j] for j in [2,3,7,6]],
            [corners[j] for j in [1,2,6,5]],
            [corners[j] for j in [4,7,3,0]],
        ]
        poly = Poly3DCollection(faces, alpha=alpha, linewidths=0.05)
        poly.set_facecolor("blue")
        ax.add_collection3d(poly)

def generate_cube_stl(path, size=1.0):
    box = trimesh.creation.box(extents=(size, size, size))
    box.export(path)
    return path

def main():
    stl_path = STL_PATH if STL_PATH else ""
    if not stl_path or not os.path.isfile(stl_path):
        stl_path = os.path.abspath("test_cube.stl")
        print(f"[info] generating cube at {stl_path}")
        generate_cube_stl(stl_path, size=1.0)

    vx = STLVoxelizer(
        stl_path,
        pad_voxels=int(PAD),
        flood_fill=bool(FLOOD_FILL),
        repair=bool(REPAIR_MESH)
    )

    if VOXEL_H is not None:
        occ, origin, h = vx.voxelize_to_h(float(VOXEL_H))
    else:
        occ, origin, h = vx.voxelize_to_resolution(int(TARGET_VOXELS))

    stats = vx.stats()
    print(f"[voxelizer] voxels={stats['voxels']}, grid={stats['grid_shape']}, h={stats['h']:.6f}")

    leaves, h_base = oct.octree_from_occ(occ, h_base=stats["h"])
    boxes, mapping = oct.instantiate_boxes_from_tree(
        leaves,
        origin=stats["origin"],
        h_base=h_base,
        density=1.0,
        penalty_gain=1e5,
        static=True
    )
    print(f"[octree] leaves={len(leaves)}, boxes={len(boxes)}")

    mesh = trimesh.load(stl_path, force='mesh')

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title("Original STL")
    draw_trimesh(ax1, mesh, alpha=0.25)
    set_axes_equal(ax1)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title("Voxelized (boxes)")
    draw_boxes_surface(ax2, boxes, max_boxes=int(MAX_BOXES_TO_DRAW))
    set_axes_equal(ax2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()