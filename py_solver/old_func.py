
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
    
    if np.dot(cen_inc - cen_ref, normal) < 0.0:
        normal = -normal

    # sign_ref = 1.0 if (axes_ref[ref_idx] @ normal) >= 0.0 else -1.0 # Correct the sign of the reference, have it point along ref normal
    # n_ref = sign_ref * axes_ref[ref_idx]                            # Contact plane normal
    # fc_ref = _face_center(axes_ref, he_ref, cen_ref, ref_idx, sign_ref)    # center of reference face in global coords

    # Reference face
    ref_dots = axes_ref @ normal
    ref_idx  = int(np.argmax(ref_dots))     # NO abs()
    n_ref    = axes_ref[ref_idx]            # outward normal of that face
    fc_ref   = _face_center(axes_ref, he_ref, cen_ref, ref_idx, +1.0)

    # Incident face
    dots_inc = axes_inc @ normal                                    # Choose face on other box whose outward normal is most opposed to collision normal
    inc_idx = int(np.argmin(dots_inc))
    sign_inc = 1.0 if dots_inc[inc_idx] < 0.0 else -1.0
    incident_poly = _face_vertices(inc, inc_idx, sign_inc)          # 2/4 corners of the incident face in global coords
    fc_inc   = _face_center(axes_inc, he_inc, cen_inc, inc_idx, sign_inc)

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
    
    clipped = _dedupe_points(clipped)
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

    return contacts, clipped

