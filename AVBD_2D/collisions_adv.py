"""Collision detection utilities shared by the 2D/3D AVBD prototype path."""

import numpy as np
from dataclasses import dataclass
from bodies import Body, CollidableShape, AABB

### -------------------------- Tolerances -------------------------- ###

_EPS_N = 1e-9                                                   # axis/cross tolerance
_EPS_CLIP = 1e-7                                                # clipping tolerance
_EPS_DUP = 1e-6                                                 # duplicate tolerance

### -------------------------- Data Classes -------------------------- ###
@dataclass
class Contact:
    bodyA: Body
    bodyB: Body
    normal: np.ndarray                                          # Unit vector pointing from body A to body B
    depth: float                                                # The amount of penetration between bodies
    point: np.ndarray                                           # The point of collision 
    feature_id : int = 0                                        # persistant ID for warm start

@dataclass
class Endpoint:
    value: float                                                # the x-coordinate
    is_min: bool                                                # basically just says is this the start or the end of the AABB
    body: CollidableShape                                       # which body we are referencing

### -------------------------- Broad Phase -------------------------- ###

def broad_phase(bodies: list[CollidableShape], ignore_ids: set[tuple[int, int]]) ->list[tuple[CollidableShape, CollidableShape]]:
    """
    Find all potential colliding pairs using the Sweep and Prune algorithm.
    Speeds up calculation from nested for loops of O(n^2) to something like O(nlogn)
    """
    endpoints: list[Endpoint] = []                              # List of the start and end of the AABBs in the x-axis
    dim: int = bodies[0].get_dim()                              # Check dimension of boides

    aabbs: dict[CollidableShape, AABB_ND] = {body: body.get_aabb() for body in bodies} # Precaclulate aabbs

    for body in bodies:
        aabb = aabbs[body]
        endpoints.append(Endpoint(value=aabb.min_x, is_min=True, body=body))
        endpoints.append(Endpoint(value=aabb.max_x, is_min=False, body=body))

    endpoints.sort(key=lambda e:e.value)                        # Sort the endpoints list in order by their value only, not the other stuff

    active_list: list[CollidableShape] = []                     # Initialize active list - list of active bodies as we cycle through AABBs

    potential_pairs: set[tuple[int, int]] = set()               # Uses set because this handles duplicate pairs like (A, B) or (B, A)

    for endpoint in endpoints:                                  # Sweep through the sorted list and form overlapping pairs, checks if they both contain same x value, then add to list if y vlaue also has overlap
        if endpoint.is_min:                                     # when we encounter a new body while cycling through sorted endpoints we then want to compare y values
            for other_body in active_list:                      # By definition these bpdies overlap on x
                # aabb1 = endpoint.body.get_aabb()
                # aabb2 = other_body.get_aabb()
                aabb1 = aabbs[endpoint.body]
                aabb2 = aabbs[other_body]

                y_overlap = (aabb1.min_y <= aabb2.max_y) and (aabb2.min_y <= aabb1.max_y) # Check for overlap on y-axis

                if dim == 2:
                    if y_overlap:                               # If y overlaps and x already overlaps than the AABBs collide, add to detailed shape collision check
                        pair = tuple(sorted((id(endpoint.body), id(other_body))))
                        if ignore_ids and pair in ignore_ids:   # Don't add in ignored contacts
                            continue
                        potential_pairs.add(pair)

                else: # dim == 3
                    z_overlap = (aabb1.min_z <= aabb2.max_z) and (aabb2.min_z <= aabb1.max_z)
                    if z_overlap and y_overlap:
                        pair = tuple(sorted((id(endpoint.body), id(other_body))))
                        if ignore_ids and pair in ignore_ids:   # Don't add in ignored contacts
                            continue
                        potential_pairs.add(pair)

            active_list.append(endpoint.body)                   # Add the current body we are inside to the active list - on first run goes right to this to start building the active list

        else:                                                   # The else is when we leave the domain of the current AABB - remove from active list
            try:                                                # Try and Except are just safer ways to remove stuff, all it is really
                active_list.remove(endpoint.body)
            except ValueError:
                pass

    body_map = {id(b): b for b in bodies}                       # Somehow this maps the ids in the potential_pairs set back to the bodies
    return [(body_map[id1], body_map[id2]) for id1, id2 in potential_pairs] 


### -------------------------- SAT (2D & 3D) -------------------------- ###

def _sat_and_overlap(A: CollidableShape, B: CollidableShape):
    """
    Performs SAT and returns (normal, overlap) if intersecting, else (None, 0).
        2D: face normals from both
        3D: face normals + 9 cross axes
    """
    
    dim = A.get_dim()
    axes_A = A.get_axes()
    axes_B = B.get_axes()

    candidate_axes = []

    # Face normals (unit vectors)
    for i in range(axes_A.shape[0]):
        candidate_axes.append(_unit(axes_A[i]))
    for i in range(axes_B.shape[0]):
        candidate_axes.append(_unit(axes_B[i]))

    if dim == 3:                                                # Compute candidate axes from the cross product of the two bodies' 3D axes
        for i in range(dim):
            for j in range(dim):
                c = np.cross(axes_A[i], axes_B[j])
                if np.linalg.norm(c) > _EPS_N:                  # If axes are nearly parallel or anit-parallel, then they will be below the threshold, do not add because they will not seperate along on this axis easily
                    candidate_axes.append(_unit(c))

    best_axis = None
    min_overlap = float('inf')

    for axis in candidate_axes:
        minA, maxA = _project_on_axis(A, axis)
        minB, maxB = _project_on_axis(B, axis)
        if (maxA <= minB) or (maxB <= minA):
            return None, 0.0
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = axis
    
    # Orient axis from A -> B
    if np.dot(_center(B) - _center(A), best_axis) < 0.0:
        best_axis = -best_axis

    return best_axis, float(min_overlap)

### -------------------------- Manifold Builder (3D) -------------------------- ###

def _build_box_box_manifold(A: CollidableShape, B: CollidableShape, normal: np.ndarray, min_overlap: float) -> list[Contact]:
    """
    Build incident face from two colliding bodies and return up to 4 Contact points. 
    Method uses an incident face and a reference face.    
    """

    dim = A.get_dim()
    assert dim in (2,3)

    axes_A = A.get_axes()
    he_A = _half_extents(A)
    cen_A = _center(A)

    axes_B = B.get_axes()
    he_B = _half_extents(B)
    cen_B = _center(B)

    # Get axis most aligned with the normal (reference axis/vector)
    # First get dot product between vectors and collision normal - projecting axes onto collision normal
    # Largest dot product for each box's axes is the axis most aligned with collision normal
    dotprods_A = axes_A @ normal
    dotprods_B = axes_B @ normal

    # Get maximum 
    max_idx_A = int(np.argmax(np.abs(dotprods_A)))
    max_idx_B = int(np.argmax(np.abs(dotprods_B)))

    # Assign reference and incidence
    if abs(dotprods_A[max_idx_A]) >= abs(dotprods_B[max_idx_B]):    # A is reference
        ref = A
        inc = B
        axes_ref, he_ref, cen_ref = axes_A, he_A, cen_A
        axes_inc, he_inc, cen_inc = axes_B, he_B, cen_B
        ref_idx = max_idx_A
    else:                                                           # B is reference
        ref = B
        inc = A
        axes_ref, he_ref, cen_ref = axes_B, he_B, cen_B
        axes_inc, he_inc, cen_inc = axes_A, he_A, cen_A
        ref_idx = max_idx_B
    
    sign_ref = 1.0 if (axes_ref[ref_idx] @ normal) >= 0.0 else -1.0 # Correct the sign of the reference, have it point along ref normal
    n_ref = sign_ref * axes_ref[ref_idx]                            # Contact plane normal
    fc_ref = _face_center(axes_ref, he_ref, cen_ref, ref_idx, sign_ref)    # center of reference face in global coords

    dots_inc = axes_inc @ normal                                    # Choose face on other box whose outward normal is most opposed to collision normal
    inc_idx = int(np.argmin(dots_inc))
    sign_inc = 1.0 if dots_inc[inc_idx] < 0.0 else -1.0
    incident_poly = _face_vertices(inc, inc_idx, sign_inc)          # 2/4 corners of the incident face in global coords

    others_ref = _other_axes_indices(axes_ref.shape[0], ref_idx)    # Returns all other axes other than ref_idx, so the two in-plane directions of the ref face
    
    if dim == 2:                                                    # dim == 2, a 1-D line segment can describe the contact, this has 2 endpoints so we have 2 planes to bound endpoints
        t1 = axes_ref[others_ref[0]]
        he_ref1 = he_ref[others_ref[0]]

        planes = [
            (t1, np.dot(t1, fc_ref) + he_ref1),
            (-t1, np.dot(-t1, fc_ref) + he_ref1),
        ]

    else:                                                           # dim == 3, a 2-D rectangle can desribe the contact so it has 4 planes to bound the intersecting rectangle (2D contact patch plane)
        t1 = axes_ref[others_ref[0]]
        he_ref1 = he_ref[others_ref[0]]
        t2 = axes_ref[others_ref[1]]
        he_ref2 = he_ref[others_ref[1]]

        planes = [
            ( t1, np.dot( t1, fc_ref) + he_ref1),
            (-t1, np.dot(-t1, fc_ref) + he_ref1),
            ( t2, np.dot( t2, fc_ref) + he_ref2),
            (-t2, np.dot(-t2, fc_ref) + he_ref2),
        ]
    
    clipped = incident_poly

    for n_plane, c_plane in planes:
        clipped = _clip_poly_to_plane(clipped, n_plane, c_plane)    # CLipped portion of incidenct face that lies inside the reference face rectangle
        if clipped.shape[0] == 0:
            return []
    
    clipped = _remove_duplicates(clipped)
    if clipped.shape[0] == 0:
        return []

    # Project onto reference plane and compute point separations, turn polygon vertices into contacts on ref plane
    d0 = float(np.dot(n_ref, fc_ref))
    contacts: list[Contact] = []
    sep_points = []

    for v in clipped:
        s = float(np.dot(n_ref, v) - d0)
        sep_points.append(s)
    
    idxs = [i for i, s in enumerate(sep_points) if s <= _EPS_CLIP]
    if not idxs:
        return []
    
    idxs.sort(key=lambda i:sep_points[i])                           # deepest first (more negative)
    idxs = idxs[:4]                                                 # keep deepest 4
    AtoB = _center(B) - _center(A)
    out_n = n_ref if np.dot(AtoB, n_ref) >= 0.0 else -n_ref         # flip to correct sign

    for i in idxs:
        v = clipped[i]
        sep_pt = sep_points[i]
        depth = float(min(-sep_pt, min_overlap + 1e-7))             # how far on normal point is penetrating, clamped to SAT overlap

        pt_on_plane = v - sep_pt * n_ref                            # Orthogonal projection of v onto ref plane


        # Stable feature id (per manifold point)
        fid = _feature_id_hash(A, B, ref_idx if ref is A else -ref_idx-1,
                               inc_idx if inc is B else -inc_idx-1, pt_on_plane)

        contacts.append(Contact(
            bodyA=A,
            bodyB=B,
            normal=out_n,
            depth=depth,
            point=pt_on_plane,
            feature_id=fid
        ))

    return contacts


### -------------------------- Narrow Phase + Get Collisions  -------------------------- ###

def narrow_phase(pair: tuple[CollidableShape, CollidableShape]):
    """
    SAT (2D/3D) + manifold clipping for OBB boxes.
    Returns a list of Contacts (0-4). None if not intersecting.
    """

    bodyA, bodyB = pair
        
    if bodyA.static and bodyB.static:
        return None                                             # No contacts between the two static, unmoveable objects

    if bodyA.static and not bodyB.static:
        bodyA, bodyB = bodyB, bodyA                             # Swap bodies if A is static - mathematical convention

    n, ov = _sat_and_overlap(bodyA, bodyB)
    if n is None:
        return None
    
    # Build contacts
    contacts = _build_box_box_manifold(bodyA, bodyB, n, ov)

    return contacts if contacts else None

def get_collisions(bodies: list[Body], ignore_ids: set[tuple[int, int]] | None = None) -> list[Contact]:
    """
    Pass in a list of bodies and detect_collision will return a list of colliding bodies.
    Runs the broad & narrow phases of collision detection.
    """

    all_contacts: list[Contact]  = []

    collidable_bodies = [b for b in bodies if isinstance(b, CollidableShape)]

    # Phase 1: Broad Phase
    potential_pairs = broad_phase(collidable_bodies, ignore_ids= (ignore_ids or set()))

    # Phase 2: Narrow Phase + manifold
    for pair in potential_pairs:
        contact_info = narrow_phase(pair)
        if contact_info:
            #all_contacts.append(contact_info)
            all_contacts.extend(contact_info)


    return all_contacts

############################################ ^OLD^ ############################################

### -------------------------- Helper Functions -------------------------- ##

def _unit(vector: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vector)
    if n == 0.0:
        return vector
    return vector / n

def _center(body: CollidableShape) -> np.ndarray:
    dim = body.get_dim()
    return body.position[:dim]

def _project_on_axis(body: CollidableShape, axis: np.ndarray) -> tuple[float, float]:
    """
    Helper function for the SAT. Project the corners of a body onto an axis, think of it like a shadow of the box onto the given axis.
    """
    axis = _unit(axis)
    corners = body.get_corners()
    projections = corners @ axis                                # Matrix multiplication to project corner vectors onto the axis

    return float(np.min(projections)), float(np.max(projections))

def _half_extents(body: CollidableShape) -> np.ndarray:
    axes = body.get_axes()
    c = _center(body)
    rel = body.get_corners() - c
    h = []

    for i in range(axes.shape[0]):
        vals = rel @ axes[i]
        h.append(float(np.max(np.abs(vals))))
    return np.array(h, dtype=float)

def _other_axes_indices(num_axes: int, idx: int):
    """Return all axis indices except the reference-face axis."""
    return [j for j in range(num_axes) if j!= idx]

def _face_center(axes: np.ndarray, half_exts: np.ndarray, center: np.ndarray,
                  face_axis_idx: int, face_sign: float) -> np.ndarray:
    return center + (face_sign * axes[face_axis_idx]) * half_exts[face_axis_idx]

def _face_vertices(body: CollidableShape, axis_idx: int, outward_sign: float) -> np.ndarray:
    """ 
    Return the face vertices whose outward normal is `outward_sign * axes[axis_idx]`.
    """
    axes = body.get_axes()
    h = _half_extents(body)
    c = _center(body)
    dim = body.get_dim()

    face_center = _face_center(axes, h, c, axis_idx, outward_sign)
    others = _other_axes_indices(dim, axis_idx)
    
    if dim == 2:
        t1 = axes[others[0]]
        h1 = h[others[0]]                  
        verts = np.stack([face_center + t1*h1, face_center - t1*h1], axis=0)
        return verts
    
    else: #dim == 3
        t1 = axes[others[0]]
        h1 = h[others[0]]
        t2 = axes[others[1]]
        h2 = h[others[1]]
        verts = []
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                verts.append(face_center + s1*t1*h1 + s2*t2*h2)
        return np.asarray(verts)

def  _clip_poly_to_plane(vertices: np.ndarray, plane_n: np.ndarray, plane_c: float) -> np.ndarray:
    """
    Sutherland-Hodgman style for 3D (and 2D) polygons.
    Creates new list of vertices for the overlapping/instersection of two planes.
    """
    if vertices.shape[0] == 0:
        return vertices
    
    keep = []
    n = plane_n
    c = plane_c

    def inside(p):
        """Return True when `p` lies on or behind the clipping plane."""
        return (np.dot(n, p) - c) <= _EPS_CLIP
    
    prev = vertices[-1]
    prev_in = inside(prev)                                      # Check to see if previous vertex is inside shape

    for current in vertices:
        current_in = inside(current)

        if current_in and prev_in:                              # Both vertices inside, keep current vertex
            keep.append(current)
        
        elif prev_in and not current_in:                        # (inside -> outside): keep the intersection vertex (out-in) aka (current - prev)
            denom = np.dot(n, (current - prev))
            if abs(denom) > _EPS_N:
                t = (c - np.dot(n, prev)) / denom               # t is for interpolation to intersection point
                keep.append(prev + t * (current - prev))
            
        elif (not prev_in) and current_in:                      # (outside -> inside): keep the intersection (prev-current) & the current 
            denom = np.dot(n, (current - prev))
            if abs(denom) > _EPS_N:
                t = (c -np.dot(n, prev)) / denom
                keep.append(prev + t * (current - prev))
            keep.append(current)

        # Outside -> outside (do not keep)
        prev, prev_in = current, current_in

    if len(keep) == 0:                                          # No clipping done
        return np.empty((0, vertices.shape[1]), dtype=float)
    
    out = [keep[0]]
    for p in keep[1:]:
        if np.linalg.norm(p - out[-1]) > _EPS_DUP:              # Check if close points are within a threshold, these would be considered duplicates
            out.append(p)
    
    if len(out) >= 2 and np.linalg.norm(out[0] - out[-1]) <= _EPS_DUP:
        out.pop()                                               # Remove duplicate point from the output list of vertices
    return np.asarray(out)   

def _remove_duplicates(points: np.ndarray) -> np.ndarray:           # Checks if any points are a duplcate and returns array of unique points
    if len(points) <= 1:
        return points
    unique = []
    for p in points:
        if not any(np.linalg.norm(p - q) <= _EPS_DUP for q in unique):
            unique.append(p)
    return np.asarray(unique)

def _feature_id_hash(bodyA: Body, bodyB: Body, ref_idx: int, inc_idx: int, p: np.ndarray) -> int:
    # Quantize the contact point so the ID is stable across nearby frames.
    q = np.round(p * 1e4).astype(int)
    return (hash((id(bodyA) >> 4, id(bodyB) >> 4, ref_idx, inc_idx, int(q.sum()))) & 0x7fffffff)
