module Collisions

using LinearAlgebra
using StaticArrays

# Assumes Maths module exists as per original file
include("maths.jl")
using .Maths

export Body, Contact, get_collisions, update_rotation!

# --- Constants (Matched to Python) ---
const TOL_ANG = 1e-12
const TOL_AXIS = 1e-12
const EPS_CLIP = 1e-7
const TOL_PNT_DUP = 1e-6

# --- Data Structures ---

mutable struct Body
    id::Int
    assembly_id::Int # 0 or -1 implies "None"/No Assembly
    is_static::Bool

    pos::Vec3
    quat::Quat
    size::Vec3
    vel::Vec3
    ang_vel::Vec3

    mass::Float64
    inertia::Mat3
    inv_mass::Float64
    inv_inertia::Mat3

    pos_prev::Vec3
    quat_prev::Quat
    rot_mat::Mat3

    # Constructor matches original for compatibility
    function Body(id, asm_id, static, pos, quat, size, mass; vel=Vec3(0, 0, 0), ang_vel=Vec3(0, 0, 0))
        R = quat_to_rotmat(quat)
        m_val = static ? Inf : mass
        inv_m = static ? 0.0 : 1.0 / m_val

        # Inertia tensor (Box)
        sx, sy, sz = size...
        ixx = (1.0 / 12.0) * m_val * (sy^2 + sz^2)
        iyy = (1.0 / 12.0) * m_val * (sx^2 + sz^2)
        izz = (1.0 / 12.0) * m_val * (sx^2 + sy^2)
        I_body = Mat3(ixx, 0, 0, 0, iyy, 0, 0, 0, izz)
        inv_I = static ? zero(Mat3) : inv(I_body)

        new(id, asm_id, static, pos, quat, size, vel, ang_vel,
            m_val, I_body, inv_m, inv_I, pos, quat, R)
    end
end

struct Contact
    body_idx_a::Int
    body_idx_b::Int
    normal::Vec3        # A -> B
    point::Vec3         # World space
    depth::Float64
    feature_id::Int
end

struct AABB
    min::Vec3
    max::Vec3
end

struct Endpoint
    val::Float64
    is_min::Bool
    body_idx::Int
end

# --- Geometry Helpers (Ported from Python) ---

function update_rotation!(b::Body)
    b.rot_mat = quat_to_rotmat(b.quat)
end

@inline function get_axes(b::Body)
    return b.rot_mat
end

@inline function get_center(b::Body)
    return b.pos
end

@inline function get_half_extents(b::Body)
    return b.size * 0.5
end

function get_corners(b::Body)
    half = get_half_extents(b)
    center = b.pos
    R = b.rot_mat

    corners = MVector{8,Vec3}(undef)
    idx = 1
    for k in (-1, 1), j in (-1, 1), i in (-1, 1)
        corners[idx] = center + R * Vec3(i * half[1], j * half[2], k * half[3])
        idx += 1
    end
    return corners
end

function get_aabb(b::Body)
    corners = get_corners(b)
    min_v = corners[1]
    max_v = corners[1]
    for i in 2:8
        min_v = min.(min_v, corners[i])
        max_v = max.(max_v, corners[i])
    end
    return AABB(min_v, max_v)
end

# Python: _unit
function unit_vec(v::Vec3)
    n = norm(v)
    if n < 1e-16
        return v
    end
    return v / n
end

# Python: _canonical_axes
function canonical_axis(axis::Vec3)
    x, y, z = axis
    if abs(x) > TOL_AXIS
        return x >= 0 ? axis : -axis
    elseif abs(y) > TOL_AXIS
        return y >= 0 ? axis : -axis
    elseif abs(z) > TOL_AXIS
        return z >= 0 ? axis : -axis
    end
    return axis
end

# Python: _face_center
function get_face_center(b::Body, axis_idx::Int, sign_val::Float64)
    return b.pos + (b.rot_mat[:, axis_idx] * sign_val * get_half_extents(b)[axis_idx])
end

# Python: _face_vertices
function get_face_vertices(b::Body, axis_idx::Int, sign_val::Float64)
    R = b.rot_mat
    half = get_half_extents(b)
    center = get_face_center(b, axis_idx, sign_val)

    # Determine tangent axes indices (1-based for Julia)
    # Python logic: others = [j for j in range(dim) if j != idx]
    # We map 1->(2,3), 2->(1,3), 3->(1,2)
    idx1 = (axis_idx % 3) + 1
    idx2 = ((axis_idx + 1) % 3) + 1

    t1 = R[:, idx1] * half[idx1]
    t2 = R[:, idx2] * half[idx2]

    # Return 4 vertices (CCW winding implicitly)
    return [
        center - t1 - t2,
        center + t1 - t2,
        center + t1 + t2,
        center - t1 + t2
    ]
end

# Python: _project_on_axis
function project_on_axis(b::Body, axis::Vec3)
    axis = unit_vec(axis)
    corners = get_corners(b)
    # Project all corners
    min_p = dot(corners[1], axis)
    max_p = min_p
    for i in 2:8
        val = dot(corners[i], axis)
        if val < min_p
            min_p = val
        end
        if val > max_p
            max_p = val
        end
    end
    return (min_p, max_p)
end

# Python: _clip_poly_to_plane (Sutherland-Hodgman)
function clip_poly_to_plane(poly::Vector{Vec3}, normal::Vec3, offset::Float64)
    if isempty(poly)
        return poly
    end

    out_poly = Vec3[]
    sizehint!(out_poly, length(poly) + 2)

    # Check if point is "inside" (dist <= epsilon)
    # Python: (np.dot(n, p) - c) <= _EPS_CLIP
    is_inside(p) = (dot(normal, p) - offset) <= EPS_CLIP

    prev = poly[end]
    prev_in = is_inside(prev)

    for curr in poly
        curr_in = is_inside(curr)

        if curr_in
            if !prev_in
                # Enter
                denom = dot(normal, curr - prev)
                if abs(denom) > TOL_AXIS
                    t = (offset - dot(normal, prev)) / denom
                    push!(out_poly, prev + t * (curr - prev))
                end
            end
            push!(out_poly, curr)
        elseif prev_in
            # Leave
            denom = dot(normal, curr - prev)
            if abs(denom) > TOL_AXIS
                t = (offset - dot(normal, prev)) / denom
                push!(out_poly, prev + t * (curr - prev))
            end
        end
        prev = curr
        prev_in = curr_in
    end

    # Dedupe points (Python: _dedupe_points logic embedded or called after)
    if length(out_poly) < 2
        return out_poly
    end

    # Simple dedupe
    unique_out = Vec3[]
    for p in out_poly
        is_dup = false
        for exist in unique_out
            if norm(p - exist) < TOL_PNT_DUP
                is_dup = true
                break
            end
        end
        if !is_dup
            push!(unique_out, p)
        end
    end

    # Remove closure duplicate if present
    if length(unique_out) > 1 && norm(unique_out[1] - unique_out[end]) < TOL_PNT_DUP
        pop!(unique_out)
    end

    return unique_out
end

# --- Broad Phase (Sweep & Prune) ---

function broad_phase(bodies::Vector{Body})
    n = length(bodies)
    if n < 2
        return Tuple{Int,Int}[]
    end

    # 1. Build Endpoints
    endpoints = Vector{Endpoint}(undef, n * 2)
    aabbs = Vector{AABB}(undef, n)

    for i in 1:n
        aabbs[i] = get_aabb(bodies[i])
        endpoints[2i-1] = Endpoint(aabbs[i].min[1], true, i)
        endpoints[2i] = Endpoint(aabbs[i].max[1], false, i)
    end

    # 2. Sort
    sort!(endpoints, by=e -> e.val)

    # 3. Sweep
    potential_pairs = Tuple{Int,Int}[]
    active_indices = Int[]
    sizehint!(active_indices, 32)

    for ep in endpoints
        idx = ep.body_idx
        b1 = bodies[idx]

        if ep.is_min
            for other_idx in active_indices
                b2 = bodies[other_idx]

                # Assembly ID Check (Python: if id is not None and equal, continue)
                # Assuming 0 or -1 indicates "No Assembly". 
                if b1.assembly_id > 0 && b2.assembly_id > 0 && b1.assembly_id == b2.assembly_id
                    continue
                end

                # Check Y and Z overlap
                aabb1, aabb2 = aabbs[idx], aabbs[other_idx]
                y_overlap = (aabb1.min[2] <= aabb2.max[2]) && (aabb2.min[2] <= aabb1.max[2])
                z_overlap = (aabb1.min[3] <= aabb2.max[3]) && (aabb2.min[3] <= aabb1.max[3])

                if y_overlap && z_overlap
                    # Store sorted tuple to handle uniqueness
                    pair = idx < other_idx ? (idx, other_idx) : (other_idx, idx)
                    push!(potential_pairs, pair)
                end
            end
            push!(active_indices, idx)
        else
            filter!(x -> x != idx, active_indices)
        end
    end

    return unique(potential_pairs)
end

# --- Narrow Phase (SAT) ---

function sat_check(bA::Body, bB::Body)
    axesA = get_axes(bA)
    axesB = get_axes(bB)

    # Gather candidate axes
    candidates = Vec3[]
    sizehint!(candidates, 15)

    # Face Normals
    for i in 1:3
        push!(candidates, unit_vec(axesA[:, i]))
    end
    for i in 1:3
        push!(candidates, unit_vec(axesB[:, i]))
    end

    # Cross Products
    for i in 1:3, j in 1:3
        c = cross(axesA[:, i], axesB[:, j])
        if norm(c) > TOL_AXIS
            push!(candidates, unit_vec(c))
        end
    end

    best_axis = Vec3(0, 0, 0)
    min_overlap = Inf
    found_separation = false

    # Check candidates
    # Note: Python does dedupe here, simplified by skipping parallel checks implicitly via min logic
    for axis in candidates
        # Canonicalize
        axis = canonical_axis(axis)

        mnA, mxA = project_on_axis(bA, axis)
        mnB, mxB = project_on_axis(bB, axis)

        if (mxA <= mnB) || (mxB <= mnA)
            return (nothing, 0.0) # Separated
        end

        overlap = min(mxA, mxB) - max(mnA, mnB)

        if overlap < min_overlap
            min_overlap = overlap
            best_axis = axis
        end
    end

    return (best_axis, min_overlap)
end

# --- Manifold Generation (Clipping) ---

function set_feature_id(bA::Body, bB::Body, p::Vec3)
    # Python: hash( (min_id, max_id, ref_face, inc_face, quantized_p) )
    # Simplified here to body IDs and quantized point
    ida, idb = bA.id, bB.id
    # Round point for stability
    px = round(p[1], digits=3)
    py = round(p[2], digits=3)
    pz = round(p[3], digits=3)

    h = hash((min(ida, idb), max(ida, idb), px, py, pz))
    return Int(h % typemax(Int))
end

function build_manifold(bA::Body, bB::Body, sat_normal::Vec3, overlap::Float64)
    # 1. Determine Reference vs Incident
    axesA = get_axes(bA)
    axesB = get_axes(bB)

    # Python: argmax(abs(dot))
    dotsA = [dot(axesA[:, i], sat_normal) for i in 1:3]
    idxA = argmax(abs.(dotsA))

    dotsB = [dot(axesB[:, i], sat_normal) for i in 1:3]
    idxB = argmax(abs.(dotsB))

    # Python logic for Ref/Inc
    contact_normal = sat_normal
    ref_body, inc_body = bA, bB
    ref_idx = idxA
    flip = false

    if abs(dotsA[idxA]) >= abs(dotsB[idxB]) - TOL_AXIS
        # A is reference
        ref_body, inc_body = bA, bB
        ref_idx = idxA
        contact_normal = sat_normal
        flip = false
    else
        # B is reference
        ref_body, inc_body = bB, bA
        ref_idx = idxB
        contact_normal = -sat_normal # Flip normal relative to A
        flip = true
    end

    # Ensure normal points Ref -> Inc
    # Python: if dot(inc_cen - ref_cen, n) < 0: n = -n
    if dot(inc_body.pos - ref_body.pos, contact_normal) < 0
        contact_normal = -contact_normal
    end

    # Ref Normal & Sign
    ref_R = ref_body.rot_mat
    raw_ref_axis = ref_R[:, ref_idx]
    ref_sign = sign(dot(raw_ref_axis, contact_normal))
    if ref_sign == 0.0
        ref_sign = 1.0
    end

    ref_normal = raw_ref_axis * ref_sign

    # If Ref normal opposes Contact normal, flip it (Python logic)
    if dot(ref_normal, contact_normal) < 0
        ref_normal = -ref_normal
        ref_sign = -ref_sign
    end

    # Get Reference Face
    ref_face = get_face_vertices(ref_body, ref_idx, ref_sign)

    # Incident Face
    inc_R = inc_body.rot_mat
    # Find most anti-parallel face on Incident
    dotsInc = [dot(inc_R[:, i], -contact_normal) for i in 1:3]
    inc_idx = argmax(abs.(dotsInc))
    # Python: inc_sign = 1.0 if dot >= 0 else -1.0
    inc_sign = dot(inc_R[:, inc_idx], -contact_normal) >= 0.0 ? 1.0 : -1.0

    inc_face = get_face_vertices(inc_body, inc_idx, inc_sign)

    # 2. Clipping (Sutherland-Hodgman)
    clipped_poly = copy(inc_face)

    # Winding Sign Helper
    function winding_sign(face, n_out)
        tri_n = cross(face[2] - face[1], face[3] - face[1])
        return dot(tri_n, n_out) >= 0 ? 1.0 : -1.0
    end

    wind_dir = winding_sign(ref_face, ref_normal)

    # Clip against Reference Side Planes
    n_verts = length(ref_face)
    for i in 1:n_verts
        v_a = ref_face[i]
        v_b = ref_face[mod1(i + 1, n_verts)] # 1-based wrap
        edge_ab = unit_vec(v_b - v_a)

        # Dynamic plane normal (Python logic)
        plane_norm = unit_vec(cross(edge_ab, ref_normal * wind_dir))
        plane_offset = dot(plane_norm, v_a)

        clipped_poly = clip_poly_to_plane(clipped_poly, plane_norm, plane_offset)
        if isempty(clipped_poly)
            return Contact[]
        end
    end

    # Clip against Reference Face Plane
    # Keep points BEHIND reference face
    ref_plane_d = dot(ref_normal, ref_face[1])
    # Python calls a final clip here, but we usually project after.
    # The python script calls clip_poly_to_plane(..., ref_normal, ref_plane_d)
    # But usually for contacts we want the points *past* the plane. 
    # Let's follow Python:
    clipped_poly = clip_poly_to_plane(clipped_poly, ref_normal, ref_plane_d)

    # 3. Project to find Depths
    contacts = Contact[]
    depths = Float64[]

    # Python uses contact_normal for final projection?
    # Python: "Project clipped_poly points onto ref_face plane along contact_normal" (comment)
    # Python Code: p = vert - s * n_ref; where s is dist to plane.

    for vert in clipped_poly
        # dist to plane
        s = dot(ref_normal, vert) - ref_plane_d

        # We only want points "inside" the reference body (s <= epsilon)
        if s <= EPS_CLIP
            depth = max(0.0, -s)

            # Project point onto reference plane
            contact_p = vert - s * ref_normal

            # Handle flipping IDs for output
            idxA = flip ? inc_body.id : ref_body.id
            idxB = flip ? ref_body.id : inc_body.id
            final_n = flip ? -contact_normal : contact_normal

            fid = set_feature_id(bA, bB, contact_p)

            push!(contacts, Contact(idxA, idxB, final_n, contact_p, depth, fid))
            push!(depths, depth)
        end
    end

    # 4. Reduction (Keep deepest 4)
    if length(contacts) > 4
        # Sort decreasing by depth
        perm = sortperm(depths, rev=true)
        keep_idxs = perm[1:4]
        return contacts[keep_idxs]
    end

    return contacts
end

# --- Main Entry Point ---

function get_collisions(bodies::Vector{Body})
    # Update rotations
    for b in bodies
        update_rotation!(b)
    end

    potential_pairs = broad_phase(bodies)
    contacts = Contact[]

    for (idxA, idxB) in potential_pairs
        bA = bodies[idxA]
        bB = bodies[idxB]

        if bA.is_static && bB.is_static
            continue
        end

        axis, overlap = sat_check(bA, bB)

        if axis !== nothing
            # Collision detected
            new_contacts = build_manifold(bA, bB, axis, overlap)
            append!(contacts, new_contacts)
        end
    end

    return contacts
end

end # module