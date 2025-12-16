from geometry.primitives import box_3D
import pyvista as pv
from py_solver import collisions
import numpy as np


box1 = box_3D((1.0, 1.0, 3.0), (0.6502878, 0, 0, -0.7596879), (0,0,0), (0,0,0), 1000, 10, (2,2,2), static=True)
box2 = box_3D((2.0, 2.0, 3.8), (0., 0., 0., 0.), (0,0,0), (0,0,0), 1000, 10, (2,2,2), static=True)

# box1 = box_3D((1.0, 1.0, 3.0), (0,0,0,0), (0,0,0), (0,0,0), 1000, 10, (2,2,2), static=True)
# box2 = box_3D((2.0, 2.0, 3.0), (0,0,0,0), (0,0,0), (0,0,0), 1000, 10, (2,2,2), static=True)

broads = collisions.broad_phase([box1, box2],[])

broad1, broad2 = broads[0][0], broads[0][1]

best_axis, overlap, tag, candidates = collisions._sat_and_overlap(broad1, broad2)
print("Number of candidate axes: ", len(candidates))
print("Candidate axes: ", candidates)

### -------------------- HELPER -------------------- ###
def make_translation(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    trans_mat = np.eye(4, dtype=float)
    trans_mat[:3,:3] = R
    trans_mat[:3,3] = t
    return trans_mat

def create_cube_mesh(b: box_3D, color="blue", opacity=0.5, show_edges=True):
    w,h,d = b.size
    cube = pv.Cube(center=(0,0,0),x_length=w, y_length=h, z_length=d)
    R = b.rotmat()
    t = b.position[:3]
    transM = make_translation(R,t)
    cube.transform(transM, inplace=True)
    return cube, dict(color=color, opacity=opacity, show_edges=show_edges)

def color_points(plotter: pv.Plotter, points, point_radius=0.05, point_color="blue"):
    for p in points:
        plotter.add_mesh(pv.Sphere(radius=point_radius, center=p), color=point_color, smooth_shading=True)

def color_axes(plotter: pv.Plotter, axes, origin=(0,0,0), scale=0.75, axis_color="black"):
    origin = np.asarray([origin])
    for a in axes:
        a = a/ (np.linalg.norm(a) + 1e-12)
        plotter.add_arrows(origin, np.array([a])*scale, color=axis_color)

### -------------------- MAIN -------------------- ###
def main():
    pv.set_plot_theme("document")
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.show_grid(color="gray")

    mesh1, mesh1_dict = create_cube_mesh(box1)
    mesh2, mesh2_dict = create_cube_mesh(box2)

    plotter.add_mesh(mesh1, **mesh1_dict)
    plotter.add_mesh(mesh2, **mesh2_dict)

    candidate_ax_arr = []
    for candidate_axis, _ in candidates:
        candidate_ax_arr.append(candidate_axis)

    color_axes(plotter, candidate_ax_arr, axis_color="yellow")
    color_axes(plotter, np.asarray([best_axis]), axis_color="red")


    plotter.camera_position = "iso"
    plotter.show(title="Testing box contacts.")

if __name__ == "__main__":
    main()
