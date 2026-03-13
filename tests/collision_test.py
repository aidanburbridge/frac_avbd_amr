from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.primitives import box_3D
import pyvista as pv
from py_solver import collisions, constraints
import numpy as np

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
print("Coll contacts: ", coll_contacts)

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

def paint_face(plotter:pv.Plotter, vertices:np.ndarray, face_color="red"):
    faces = np.hstack([[4, 0, 1, 2, 3]])
    face = pv.PolyData(vertices, faces)

    plotter.add_mesh(face, color=face_color)#, opacity=1)

def main():
    pv.set_plot_theme("document")
    
    plotter = pv.Plotter()
    plotter.add_axes()
    #plotter.show_grid(color="gray")

    mesh1, mesh1_dict = create_cube_mesh(boxA, color="blue")    # A is blue
    mesh2, mesh2_dict = create_cube_mesh(boxB, color="red")     # B is red

    plotter.add_mesh(mesh1, **mesh1_dict)
    plotter.add_mesh(mesh2, **mesh2_dict)

    # print("Best axis: ", sat_result, label)
    # print("Contact normal: ", contact_normal)

    #paint_face(plotter, ref_face, face_color="black")
    #paint_face(plotter, inc_face, face_color="white")

    color_axes(plotter, np.asarray([contact_normal]), origin=ref_body.position[:3], axis_color="green", scale=1)
    color_axes(plotter, np.asarray([ref_axis]), origin=ref_body.position[:3], axis_color="black")
    color_axes(plotter, np.asarray([inc_axis]), origin=inc_body.position[:3], axis_color="white")

    # for c in contact_points:
    #     color_points(plotter, c.point)
    
    #color_contacts(plotter, contacts)
    #color_points(plotter, clip)

    #color_points(plotter, clipped_poly, point_color="purple", point_radius=.02)
    #color_points(plotter, contact_pts, point_color="yellow", point_radius=.03)

    # Color constraint points
    # for i, pt in enumerate(contact_pts):
    #     color_vectors(plotter, pt, contact_normal, depths[i])
    
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
        # A
        color_vectors(plotter, pA, jA[0][:3], c.depth, arrow_color="blue")
        color_point(plotter, pA, point_color="blue", point_radius=.03)
        # B
        color_vectors(plotter, pB, jB[0][:3], c.depth, arrow_color="red")
        color_point(plotter, pB, point_color="red", point_radius=.03)    
        constraint_calc = float(np.dot(c.contact.normal, (pA - pB))) + 1e-9
        print(f"\nConstraint calculation: {constraint_calc}")



    print("Manual test:\n")
    for con in contact_list:
        print(f"\t Pair ({id(con.bodyA)%1000}, {id(con.bodyB)%1000}), point: {con.point.round(3)}, normal: {con.normal.round(3)}, depth: {con.depth.round(3)}")

    print("Collision test:\n")
    for con in coll_contacts:
        print(f"\t Pair ({id(con.bodyA)%1000}, {id(con.bodyB)%1000}), point: {con.point.round(3)}, normal: {con.normal.round(3)}, depth: {con.depth.round(3)}")


    plotter.camera_position = "iso"
    plotter.enable_parallel_projection()  # Enable orthographic (parallel) projection
    plotter.show(title="Testing box contacts.")

if __name__ == "__main__":
    main()
