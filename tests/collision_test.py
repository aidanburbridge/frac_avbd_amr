from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.primitives import box_3D
import pyvista as pv
from py_solver import collisions, constraints
import numpy as np

BODY_A_COLOR = "#c43c39"
BODY_B_COLOR = "#2f6db3"
NORMAL_COLOR = "black"
TANGENT_COLOR = "#2f9e44"
MANIFOLD_COLOR = "#ffd84d"
MANIFOLD_EDGE_COLOR = "black"

# wrong way
boxA = box_3D((0., 0.5, 2.9), (0.2, 0.1, 0.4, 0.5), (0,0,0), (0,0,0), 1000, 10, (2,2,2), static=False)
boxB = box_3D((0.8, 1., 1.2), (0.0, 0.0, 0.0, 0.0), (0,0,0), (0,0,0), 1000, 10, (2,2,2), static=False)
#Best axis:  [ 0.26086957  0.13043478 -0.95652174] ('FA', 1)
#Contact normal:  [-0.26086957 -0.13043478  0.95652174]
#Reference normal sign:  1.0
# No flip
#Ref and contact dot prod:  0.9565217391304348

# wrong way
# boxA = box_3D((0., 0.5, 1), (0.12, 0.4, 0.7, 0.7), (0,0,0), (0,0,0), 1000, 10, (2,1,2), static=False)
# boxB = box_3D((0.5, 2, 1.6), (0.06, 0.04, 0.30, 0.5), (0,0,0), (0,0,0), 1000, 10, (1,2,1), static=False)
#Best axis:  [0.22016222 0.85515643 0.46929316] ('FA', 2)
#Contact normal:  [-0.22016222 -0.85515643 -0.46929316]
#Reference normal sign:  -1.0
# No flip

# wrong way
# boxA = box_3D((0.8, 0.5, 1.9), (0.12, 0.4, 0.4, 0.7), (0,0,0), (0,0,0), 1000, 10, (2,1,2), static=False)
# boxB = box_3D((0.2, 2, 1.6), (0.06, 0.04, 0.8, 0.5), (0,0,0), (0,0,0), 1000, 10, (1,2,1), static=False)
#Best axis:  [ 0.4572144  -0.80044708  0.38760735] ('EE', 1, 2)
#Contact normal:  [-0.4572144   0.80044708 -0.38760735]
#Reference normal sign:  -1.0
# No flip
#Ref and contact dot prod:  0.8545119413843618

# wrong way
# boxA = box_3D((0.5, 0.4, 0), (0.5, 0.4, 0.4, 0.7), (0,0,0), (0,0,0), 1000, 10, (1,1,2), static=False)
# boxB = box_3D((0.2, 0, 1), (0.06, 0.4, 0.8, 0.5), (0,0,0), (0,0,0), 1000, 10, (1,2,1), static=False)
#Best axis:  [ 0.24084764  0.58595004 -0.77372796] ('EE', 2, 0)
#Contact normal:  [ 0.24084764  0.58595004 -0.77372796]
#Reference normal sign:  1.0
# No flip

# wrong way
# boxA = box_3D((0.1, 0.5, 1), (0.12, 0.4, 0.7, 0.7), (0,0,0), (0,0,0), 1000, 10, (2,1,2), static=False)
# boxB = box_3D((0.2, 2, 2), (0.06, 0.04, 0.30, 0.5), (0,0,0), (0,0,0), 1000, 10, (1,2,1), static=False)
#Best axis:  [0.22016222 0.85515643 0.46929316] ('FA', 2)
#Contact normal:  [-0.22016222 -0.85515643 -0.46929316]
#Reference normal sign:  -1.0
# No flip
#Ref and contact dot prod:  1.0
 
# IDK wtf is going on here?? - could be correct??
# boxA = box_3D((0.1, 0.0, 0), (0.5, 0., 0., 0.7), (0,0,0), (0,0,0), 1000, 10, (1,1,2), static=False)
# boxB = box_3D((0.5, 1, 1), (0.6, 0.4, 0.8, 0.5), (0,0,0), (0,0,0), 1000, 10, (1,2,1), static=False)
# Reference normal sign:  -1.0
# Best axis:  [ 0.94594595  0.32432432 -0.        ] ('FB', 1)
# Contact normal:  [-0.94594595 -0.32432432  0.        ]

broads = collisions.broad_phase([boxA, boxB],[])

broad1 = broads[0][0]
broad2 = broads[0][1]

#print("Broad phase collisions: ", broads)
sat_result, overlap, label = collisions._sat_and_overlap(broad1, broad2)

# print("SAT results: ", sat_result)
# print("Overlap: ", overlap)
# print("Num candidates: ", len(candidates),"\nCandidates: ", candidates)

#contacts, clip = collisions._build_box_box_manifold(broad1, broad2, sat_result, overlap)


boxA.body_id = 1
boxB.body_id = 2
contact_list, clipped_poly, ref_face, ref_axis, inc_face, inc_axis, contact_normal, contact_pts, depths = collisions._build_contact_manifold(boxA, boxB, sat_result, overlap,debug=True)

if contact_list:
    ref_body = contact_list[0].bodyA
    inc_body = contact_list[0].bodyB
else:
    ref_body = boxA
    inc_body = boxB

# print("Num contacts: ", len(contacts))
# print("Clipped poly: ", clip)

# for i, contact in enumerate(contacts):
#     print(f"Contact {i} normal: ", contact.normal)
#     print(f"Contact {i} points: ", contact.point)

bodies_list = [boxA, boxB]
coll_contacts = collisions.get_collisions(bodies_list)
print("Contact constraints: ", coll_contacts)

const_list = []
pts_to_color = []
for c in coll_contacts:
    con_const = constraints.ContactConstraint(contact=c, friction=0.6)
    con_const.initialize()
    con_const.compute_constraint(alpha=0.99)
    # print(f"Contact constraint: {con_const}, list: {con_const.point_list}")
    const_list.append(con_const)
    con_const.compute_derivatives(body=boxA)
    con_const.compute_derivatives(body=boxB)

    for p in con_const.point_list:
        pts_to_color.append(p)


#################### FUNCTIONS ####################
def make_translation(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    trans_mat = np.eye(4, dtype=float)
    trans_mat[:3,:3] = R
    trans_mat[:3,3] = t
    return trans_mat

def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    return v / (np.linalg.norm(v) + 1e-12)

def fallback_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = unit(normal)
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    t1 = unit(np.cross(n, ref))
    t2 = unit(np.cross(n, t1))
    return t1, t2

def contact_center() -> np.ndarray:
    if isinstance(contact_pts, np.ndarray) and len(contact_pts):
        return np.mean(contact_pts, axis=0)
    if isinstance(clipped_poly, np.ndarray) and len(clipped_poly):
        return np.mean(clipped_poly, axis=0)
    return 0.5 * (ref_body.position[:3] + inc_body.position[:3])

def add_labels(plotter: pv.Plotter, points, labels, text_color="black", font_size=18):
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return
    plotter.add_point_labels(
        pts,
        labels,
        text_color=text_color,
        font_size=font_size,
        show_points=False,
        fill_shape=False,
        always_visible=True,
    )

def add_labeled_vector(
    plotter: pv.Plotter,
    origin: np.ndarray,
    direction: np.ndarray,
    label: str,
    color: str,
    length: float,
    label_offset: float,
):
    origin = np.asarray(origin, dtype=float).reshape(1, 3)
    direction = unit(direction).reshape(1, 3) * float(length)
    plotter.add_arrows(origin, direction, mag=1.0, color=color)
    label_pos = origin[0] + direction[0] + unit(direction[0]) * label_offset
    add_labels(plotter, [label_pos], [label], text_color=color, font_size=20)

def create_cube_mesh(b: box_3D, color="blue", opacity=0.5, show_edges=True):
    w,h,d = b.size
    cube = pv.Cube(center=(0,0,0),x_length=w, y_length=h, z_length=d)
    R = b.rotmat()
    t = b.position[:3]
    transM = make_translation(R,t)
    cube.transform(transM, inplace=True)
    return cube, dict(color=color, opacity=opacity, show_edges=show_edges)

def color_contacts(plotter: pv.Plotter, contacts, normal_scale=0.5, point_radius=0.05,
                   point_color="green", arrow_color="red"):
    if not contacts:
        return
    pts, dirs = [], []
    for c in contacts:
        p = np.asarray(c.point, float).reshape(3)
        n = np.asarray(c.normal, float).reshape(3)
        n /= (np.linalg.norm(n) + 1e-12)
        pts.append(p); dirs.append(n)
        plotter.add_mesh(pv.Sphere(radius=point_radius, center=p),
                         color=point_color, smooth_shading=True)

    pts = np.asarray(pts); dirs = np.asarray(dirs)
    # Color arrows directly; no scalar bar, no update needed
    plotter.add_arrows(pts, dirs, mag=normal_scale, color=arrow_color)

def color_points(plotter: pv.Plotter, points, point_radius=0.05, point_color="purple"):
    for p in points:
        plotter.add_mesh(pv.Sphere(radius=point_radius, center=p), color=point_color, smooth_shading=True)

def color_point(plotter: pv.Plotter, point, point_radius=0.05, point_color="purple"):
    plotter.add_mesh(pv.Sphere(radius=point_radius, center=point), color=point_color, smooth_shading=True)

def color_vectors(plotter: pv.Plotter, points, directions, length=1.0, arrow_color="purple"):
    directions /= np.linalg.norm(directions + 1e-12)
    plotter.add_arrows(points, directions, mag=length, color=arrow_color)

def color_axes(plotter: pv.Plotter, axes, origin=(0,0,0), scale=0.75, axis_color="black"):
    origin = np.asarray([origin])
    for a in axes:
        a = a/ (np.linalg.norm(a) + 1e-12)
        plotter.add_arrows(origin, np.array([a])*scale, color=axis_color)

def paint_face(
    plotter: pv.Plotter,
    vertices: np.ndarray,
    face_color=MANIFOLD_COLOR,
    edge_color=MANIFOLD_EDGE_COLOR,
    opacity=0.98,
    line_width=6,
):
    if vertices is None or len(vertices) < 3:
        return
    face_ids = np.hstack([[len(vertices)], np.arange(len(vertices), dtype=int)])
    face = pv.PolyData(np.asarray(vertices, dtype=float), face_ids)
    plotter.add_mesh(
        face,
        color=face_color,
        opacity=opacity,
        show_edges=True,
        edge_color=edge_color,
        line_width=line_width,
        smooth_shading=True,
    )

def main():
    pv.set_plot_theme("document")
    plotter = pv.Plotter(window_size=(1600, 1200))
    plotter.set_background("white")

    mesh1, mesh1_dict = create_cube_mesh(boxA, color=BODY_A_COLOR, opacity=0.28, show_edges=False)
    mesh2, mesh2_dict = create_cube_mesh(boxB, color=BODY_B_COLOR, opacity=0.28, show_edges=False)

    plotter.add_mesh(mesh1, **mesh1_dict)
    plotter.add_mesh(mesh2, **mesh2_dict)

    patch_center = contact_center()
    if const_list:
        t1, t2 = (unit(const_list[0].tangents[0]), unit(const_list[0].tangents[1]))
    else:
        t1, t2 = fallback_tangent_basis(contact_normal)

    scene_scale = max(
        np.linalg.norm(boxA.position[:3] - boxB.position[:3]),
        np.max(boxA.size),
        np.max(boxB.size),
        1.0,
    )
    vector_length = 0.45 * scene_scale
    label_offset = 0.06 * scene_scale
    body_label_offset = 0.18 * scene_scale

    paint_face(plotter, clipped_poly)

    add_labeled_vector(plotter, patch_center, contact_normal, "n", NORMAL_COLOR, vector_length, label_offset)
    add_labeled_vector(plotter, patch_center, t1, "t1", TANGENT_COLOR, 0.9 * vector_length, label_offset)
    add_labeled_vector(plotter, patch_center, t2, "t2", TANGENT_COLOR, 0.9 * vector_length, label_offset)

    body_label_dir = unit(contact_normal + 0.35 * t2)
    add_labels(
        plotter,
        [boxA.position[:3] - body_label_dir * body_label_offset],
        ["Body A"],
        text_color=BODY_A_COLOR,
    )
    add_labels(
        plotter,
        [boxB.position[:3] + body_label_dir * body_label_offset],
        ["Body B"],
        text_color=BODY_B_COLOR,
    )
    add_labels(
        plotter,
        [patch_center + (0.18 * scene_scale) * (0.4 * t1 + t2)],
        ["Contact manifold"],
        text_color="black",
    )
    
    print(f"\nNum contacts: {len(contact_list)}")

    print(f"Initial box body id A: {boxA}, body B: {boxB}")

    for c in const_list:
        jA = c.JA
        jB = c.JB
        pA, pB = c.point_list
        bodyA = c.bodyA
        bodyB = c.bodyB
        print(f"Constraint body id A: {bodyA}, body B: {bodyB}")

        print(f"\njA: \t{jA[0][:3]}\njB: \t{jB[0][:3]}\npA: \t{pA}\npB: \t{pB}\ndepth: \t{c.depth}")
        constraint_calc = float(np.dot(c.contact.normal, (pA - pB))) + 1e-9
        print(f"\nConstraint calculation: {constraint_calc}")



    print("Contact manifold (manual build):\n")
    for con in contact_list:
        print(f"\t Pair ({id(con.bodyA)%1000}, {id(con.bodyB)%1000}), point: {con.point.round(3)}, normal: {con.normal.round(3)}, depth: {con.depth.round(3)}")

    print("Contact constraints (collision pipeline):\n")
    for con in coll_contacts:
        print(f"\t Pair ({id(con.bodyA)%1000}, {id(con.bodyB)%1000}), point: {con.point.round(3)}, normal: {con.normal.round(3)}, depth: {con.depth.round(3)}")


    camera_dir = unit(0.45 * contact_normal + 0.95 * t1 + 0.35 * t2)
    plotter.camera_position = (
        patch_center + 3.0 * scene_scale * camera_dir,
        patch_center,
        t2,
    )
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.6)
    plotter.show(title="Contact manifold visualization")

if __name__ == "__main__":
    main()
