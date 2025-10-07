# COLLISIONS
import numpy as np
from dataclasses import dataclass
from geometry.primitives import Body, CollidableShape, AABB_ND

### -------------------------- Tolerances -------------------------- ###

_AXIS_TOL = 1e-12                                                # axis/cross tolerance
_ANG_TOL = 1e-12
_EPS_CLIP = 1e-7                                                # clipping tolerance
_TOL_PNT_DUP = 1e-6                                                 # duplicate tolerance


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

def _sat_and_overlap(A: CollidableShape, B: CollidableShape): # TODO add face bias
    """
    Performs SAT and returns (normal, overlap) if intersecting, else (None, 0).
        2D: face normals from both
        3D: face normals + 9 cross axes
    """
    
    dim = A.get_dim()
    axes_A = A.get_axes()
    axes_B = B.get_axes()

    candidate_axes = []                                         # List of (axis, contact tag)

    # Face normals (unit vectors)
    for i in range(axes_A.shape[0]):
        candidate_axes.append((_unit(axes_A[i]), ("FA", i)))      # face-normal from A
    for j in range(axes_B.shape[0]):
        candidate_axes.append((_unit(axes_B[j]), ("FB", j)))      # face-normal from B

    if dim == 3:                                                # Compute candidate axes from the cross product of the two bodies' 3D axes
        for i in range(dim):
            for j in range(dim):
                c = np.cross(axes_A[i], axes_B[j])
                if np.linalg.norm(c) > _AXIS_TOL:               # If axes are nearly parallel or anit-parallel, then they will be below the threshold, do not add because they will not seperate along on this axis easily
                    candidate_axes.append((_unit(c), ("EE", i, j)))
    cand = []
    for ax, tag in candidate_axes:
        u = _canonical_axes(_unit(ax))
        cand.append((u,tag))

    cand = _dedupe_axes_w_tag(cand)         # Don't really need to dedupe - could lose accuracy

    best_axis, best_tag = None, None
    min_overlap = float('inf')

    for axis, tag in cand:
        #print("Candidate axes tags: ", tag)
        minA, maxA = _project_on_axis(A, axis)                  # Gets min and max points on the possible separation axis 
        minB, maxB = _project_on_axis(B, axis)
        if (maxA <= minB) or (maxB <= minA):
            return None, 0.0, None
        overlap = min(maxA, maxB) - max(minA, minB)             # Gets the overlap between the max and min points
        #print("Overlap: ", overlap)
        if overlap +_AXIS_TOL < min_overlap:
            min_overlap = overlap
            best_axis, best_tag = axis, tag
            #print("New best axis: ", best_axis)

    return best_axis, float(min_overlap), best_tag, cand

### -------------------------- Manifold Builder (3D) -------------------------- ###

def _build_contact_manifold(A: CollidableShape, B:CollidableShape, sat_normal:np.ndarray) -> list[Contact]:

    # Need which vertices to clip to

    # First find plane that is most close to separation normal - on which cube this is?
    # How do we determine which cube?
    # Find which plane on the reference cube (one with plane most normal to sat_normal)
    # Then project the other cube's vertices onto this refernce plane
    # Then clip the other cube's projected vertices to this reference plane 
    # Then choose the points with the deepest penetration here (?) - how is that determined? If the points lie on the plane then what?

    # 1. choose which box is reference based on which axis is most aligned with SAT axis
    # do this with the dot_product - compare one dot_prod to the other
    # this gets reference normal and then from this get reference face

    # 2.clip the incident cube's incident face to the planes that define reference face on the ref cube
    # Bascially take two rectangular faces and clip one to the edges of another
    # do so by picking reference face- this is the one most parallel to SAT normal
    # the incident face is the face most anti-parallel to the SAT normal so like parallel but opposite direction (negative)

    # Choose reference and incident
    axes_A = A.get_axes()
    axes_B = B.get_axes()
    sat_normal = _unit(sat_normal)
    
    dot_prod_A = np.dot(axes_A, sat_normal)
    max_idx_A = int(np.argmax(np.abs(dot_prod_A)))

    dot_prod_B = np.dot(axes_B, sat_normal)
    max_idx_B = (np.argmax(np.abs(dot_prod_B)))

    # Override default if need be
    # SAT by default always points from A -> B 
    # Manifold should point Ref -> Inc
    # If Ref is A -- no change
    # If Ref is B -- flip sign of sat_normal
    if dot_prod_A[max_idx_A] >= dot_prod_B[max_idx_B] - _AXIS_TOL:
        ref_body, inc_body = A, B
        contact_normal = sat_normal.copy()
        ref_axis_idx = max_idx_A
    else:
        ref_body, inc_body = B, A
        contact_normal = -sat_normal.copy()
        ref_axis_idx = max_idx_B
    
    ref_cen, inc_cen = _center(ref_body), _center(inc_body)


    # Point from reference -> incident
    if np.dot(inc_cen - ref_cen, contact_normal) < 0.0:
        contact_normal = -contact_normal
    
    ref_axes = ref_body.get_axes()
    ref_he = _half_extents(ref_body)

    raw_ref_axis = ref_axes[ref_axis_idx]
    ref_sign = np.sign(np.dot(raw_ref_axis, contact_normal))
    if ref_sign == 0.0:                                     # If dot product is 0 - parallel
        ref_sign = 1.0

    ref_normal = _unit(raw_ref_axis * ref_sign)

    # get face for reference
    # If the reference face normal is not aligned with the contact normal, flip the face vertices around the center
    # This ensures the face is oriented correctly in global coordinates
    print("Ref and contact dot prod: ", np.dot(ref_normal, contact_normal))
    if np.dot(ref_normal, contact_normal) < 0.0:
        ref_normal = -ref_normal
        ref_sign = -ref_sign
        print("flipped ref normal sign")
    print("Reference normal sign: ", ref_sign)

    ref_face = _face_vertices(ref_body, ref_axis_idx, ref_sign)

    # get incident normal/axis
    inc_axes = inc_body.get_axes()
    dot_prod_inc = np.dot(inc_axes, -contact_normal)     # most anti-parallel
    inc_axis_idx = int(np.argmax(np.abs(dot_prod_inc)))

    # keep same signage - compare to abs but then restore sign
    inc_sign = np.sign(dot_prod_inc[inc_axis_idx]) or -1.0
    inc_normal = _unit(inc_body.get_axes()[inc_axis_idx]) * inc_sign

    # get face for incident
    inc_face = _face_vertices(inc_body, inc_axis_idx, inc_sign)
    
    clipped_poly = inc_face.copy()

    def winding_sign(face, n_out):
        # +1 => CCW wrt n_out,  -1 => CW wrt n_out
        # Use first triangle (or Newell if you want extra robustness)
        tri_n = np.cross(face[1] - face[0], face[2] - face[0])
        s = np.dot(tri_n, n_out)
        return 1.0 if s >= 0.0 else -1.0
    
    wind_dir = winding_sign(ref_face, ref_normal)

    N = len(ref_face)
    for i in range(N):
        v_a = ref_face[i]                       # starting vertex 
        v_b = ref_face[(i+1) % N]               # ending vertex
        edge_ab = _unit(v_b - v_a)              # edge direction intersecting plane and face
        
        # find normal into face with cross product between negative face normal and edge direction (right hand rule - face vertex defined CCW)
        plane_norm = _unit(np.cross(edge_ab, ref_normal* wind_dir))
        plane_offset = float(np.dot(plane_norm, v_a))


        # clip incident face at 4 reference planes
        clipped_poly = _clip_poly_to_plane(clipped_poly, plane_norm, plane_offset)
        
        if clipped_poly.shape[0] == 0:
            # No overlap region on faces; return empty polygon with consistent outputs
            return (np.empty((0, ref_face.shape[1])), ref_face, ref_normal,
                    inc_face, inc_normal, contact_normal)

    clipped_poly = _clip_poly_to_plane(clipped_poly, ref_normal, float(np.dot(ref_normal,ref_face[0])))

    clipped_poly = _dedupe_points(clipped_poly)
    # project clipped points onto reference

    sep = []
    proj = []

    d_ref = float(np.dot(ref_normal, ref_face[0]))
    for pt in clipped_poly:
        s = float(np.dot(ref_normal, pt) - d_ref)
        sep.append(s)
        proj.append(pt -s*ref_normal)


    # keep points that are on/inside the plane (tolerant)
    idxs = [i for i, s in enumerate(sep) if s >= -_EPS_CLIP]
    if not idxs:
        return (np.empty((0, ref_face.shape[1])), ref_face, ref_normal,
                inc_face, inc_normal, contact_normal)

    # choose up to 4 deepest (largest s)
    idxs.sort(key=lambda i: -sep[i])
    idxs = idxs[:4]

    # --- Build contacts (normal points ref -> inc) ---
    contacts: list[Contact] = []

    # (Optional) clamp depth by SAT overlap if you have it; otherwise use s directly.
    # depth = min(sep[i], min_overlap + 1e-7)   # if you pass min_overlap to this function
    # For now:
    for i in idxs:
        pt_on_plane = proj[i]
        depth = sep[i]                             # already positive penetration along +ref_normal

        # If your engine expects normal from body A -> body B, align here:
        AtoB = _center(B) - _center(A)
        out_n = contact_normal if np.dot(AtoB, contact_normal) >= 0.0 else -contact_normal

        fid = _feature_id_hash(A, B,
                            ref_axis_idx if ref_body is A else -ref_axis_idx-1,
                            inc_axis_idx if inc_body is B else -inc_axis_idx-1,
                            pt_on_plane)

        contacts.append(Contact(
            bodyA=A, bodyB=B,
            normal=_unit(out_n),
            depth=float(depth),
            point=pt_on_plane,
            feature_id=fid
        ))
        
    return contacts, np.asarray(proj), ref_face, ref_normal, inc_face, inc_normal, contact_normal
#    return clipped_poly, ref_face, ref_normal, inc_face, inc_normal, contact_normal

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

    n, ov, tag, _ = _sat_and_overlap(bodyA, bodyB)
    if n is None:
        return None
    
    kind = tag[0]
    # face-point contact
    if kind in ("FA", "FB"):                                    # If face-point clip the voxel
        contacts, _ = _build_box_box_manifold(bodyA,bodyB,n,ov)
    # edge-edge contact
    else:
        i, j = tag[1], tag[2]
        # Build contacts
        contacts,_ = _build_edge_edge_contact(bodyA, bodyB, n, ov, i, j)

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

### -------------------------- Edge-edge Functions -------------------------- ##

def _box_edges_along_n(body, dir_indx, n):
    pass

def _build_edge_edge_contact(A: CollidableShape, B: CollidableShape, normal: np.ndarray, min_overlap: float, i, j) -> list[Contact]:
    # calculate distance 
    pass


# TODO move the geometric helper functions into primatives
### -------------------------- Helper Functions -------------------------- ###

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

def _other_axes_indices(num_axes: int, idx: int):               # TODO what is this?
    return [j for j in range(num_axes) if j!= idx]

def _face_center(axes: np.ndarray, half_exts: np.ndarray, center: np.ndarray,
                  face_axis_idx: int, face_sign: float) -> np.ndarray:
    return center + (face_sign * axes[face_axis_idx]) * half_exts[face_axis_idx]

def _face_vertices(body: CollidableShape, axis_idx: int, outward_sign: float) -> np.ndarray: #TODO what does outward normal = outwardsign * axes[axis_idx] mean?
    """ 
    Returns the 4 vertices of the face from the body whose outward normal is outward_sign * axes[axis_idx]
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
        verts = np.stack([face_center - t1*h1, face_center + t1*h1], axis=0)
        return verts
    
    else: #dim == 3
        t1 = axes[others[0]]
        h1 = h[others[0]]
        t2 = axes[others[1]]
        h2 = h[others[1]]
        verts = []
        v00 = face_center - t1*h1 - t2*h2
        v10 = face_center + t1*h1 - t2*h2
        v11 = face_center + t1*h1 + t2*h2
        v01 = face_center - t1*h1 + t2*h2
        return np.asarray([v00, v10, v11, v01], dtype=float)
    
def _canonical_axes(axes):
    """
    Flip the sign so the axis lives in a canonical hemisphere.
    """
    x,y,z=axes

    if abs(x) > _AXIS_TOL:                                      # If x is non-zero
        return axes if x >= 0 else -axes                        # Flip the axes if x is negative
    
    elif abs(y) > _AXIS_TOL:                                      # If y is non-zero
        return axes if y >= 0 else -axes
    
    elif abs(z) > _AXIS_TOL:
        return axes if z >= 0 else -axes
    
    return axes

def _dedupe_axes(axes):
    """
    Make list of axes unique.
    """
    out = []
    for a in axes:
        if np.linalg.norm(a) < _AXIS_TOL:
            continue
        u = _canonical_axes(_unit(a))
        # Go through saved axes and check if current axes is not parallel (same) then add to output
        if all(abs(np.dot(u,b)) <= (1.0 - _ANG_TOL) for b in out):
            out.append(u)

    return out

def _dedupe_axes_w_tag(axes):
    """
    Make list of axes unique, preserving their tags.
    Input: list of (axis, tag)
    Output: list of (axis, tag)
    """
    out = []
    for a, tag in axes:
        if np.linalg.norm(a) < _AXIS_TOL:
            continue
        u = _canonical_axes(_unit(a))
        # Go through saved axes and check if current axes is not parallel (same) then add to output
        if all(abs(np.dot(u, b)) <= (1.0 - _ANG_TOL) for b, _ in out):
            out.append((u, tag))
    return out


def  _clip_poly_to_plane(vertices: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """
    Sutherland-Hodgman style for 3D (and 2D) polygons.
    Creates new list of vertices for the overlapping/instersection of two planes.\
    Normal: vector normal to the clipping plane.
    Offset: scalar offset of the plane from origin (along normal).
    """
    if vertices.shape[0] == 0:
        return vertices
    
    keep = []
    n = _unit(normal)
    c = offset

    def inside(p): # TODO - what is this doing? checking to see if p plane p along normal is inside plane c by some tolerance?
        return (np.dot(n, p) - c) <= _EPS_CLIP
    
    prev = vertices[-1]
    prev_in = inside(prev)                                      # Check to see if previous vertex is inside shape

    for current in vertices:
        current_in = inside(current)

        if current_in and prev_in:                              # Both vertices inside, keep current vertex
            keep.append(current)
        
        elif prev_in and not current_in:                        # (inside -> outside): keep the intersection vertex (out-in) aka (current - prev)
            denom = np.dot(n, (current - prev))
            if abs(denom) > _AXIS_TOL:
                t = (c - np.dot(n, prev)) / denom               # t is for interpolation to intersection point
                keep.append(prev + t * (current - prev))
            
        elif (not prev_in) and current_in:                      # (outside -> inside): keep the intersection (prev-current) & the current 
            denom = np.dot(n, (current - prev))
            if abs(denom) > _AXIS_TOL:
                t = (c -np.dot(n, prev)) / denom
                keep.append(prev + t * (current - prev))
            keep.append(current)

        # Outside -> outside (do not keep)
        prev, prev_in = current, current_in

    if len(keep) == 0:                                          # No clipping done
        return np.empty((0, vertices.shape[1]), dtype=float)
    
    out = [keep[0]]
    for p in keep[1:]:
        if np.linalg.norm(p - out[-1]) > _TOL_PNT_DUP:              # Check if close points are within a threshold, these would be considered duplicates
            out.append(p)
    
    if len(out) >= 2 and np.linalg.norm(out[0] - out[-1]) <= _TOL_PNT_DUP:
        out.pop()                                               # Remove duplicate point from the output list of vertices
    return np.asarray(out)   

def _build_clipping_planes(face_vertices: np.ndarray, n_ref: np.ndarray):
    """
    Returns 2 to 4 planes, depending on the dimensionality (2D/3D) that define the given edge/face.
    Use these values in the Sutherland-Hodgman clipping algorithm.
    """
    face_vertices = np.asarray(face_vertices, dtype=float)
    pass
    

def _dedupe_points(points: np.ndarray) -> np.ndarray:           # Checks if any points are a duplcate and returns array of unique points
    if len(points) <= 1:
        return points
    unique = []
    for p in points:
        if not any(np.linalg.norm(p - q) <= _TOL_PNT_DUP for q in unique):
            unique.append(p)
    return np.asarray(unique)

def _feature_id_hash(bodyA: Body, bodyB: Body, ref_idx: int, inc_idx: int, p: np.ndarray) -> int: # TODO wtf is going on here?
    # Quantize point for ID stability (helps warm-starting)
    q = np.round(p * 1e4).astype(int)
    return (hash((id(bodyA) >> 4, id(bodyB) >> 4, ref_idx, inc_idx, int(q.sum()))) & 0x7fffffff)