"""
Broad-phase and narrow-phase collision detection for voxel bodies.

The module stores rigid-body state, updates oriented box transforms, performs
sweep-and-prune broad phase culling, and generates contact candidates for the
manifold cache used by the AVBD solver.
"""
module Collisions

using LinearAlgebra
using StaticArrays
using Base.Threads

include("maths.jl")
using .Maths: Vec3, Mat3, Quat, quat_to_rotmat

export Body, Contact, AABB, BroadPhaseState
export update_rotation!, update_inv_inertia_world!, broad_phase!, broad_phase_active!, sat_check, build_manifold!, get_collisions!, get_collisions, get_collisions_active

# --- Constants ---
const TOL_AXIS = 1e-12
const EPS_CLIP = 1e-7
const TOL_PNT_DUP = 1e-6

# --- Type Definitions ---

mutable struct Body
    id::Int
    assembly_id::Int
    is_static::Bool

    pos::Vec3
    quat::Quat
    size::Vec3
    vel::Vec3
    ang_vel::Vec3

    mass::Float64
    inv_mass::Float64
    inertia_diag::SVector{3,Float64}
    inv_inertia_diag::SVector{3,Float64}
    inv_inertia_world::SMatrix{3,3,Float64,9}

    pos_prev::Vec3
    quat_prev::Quat
    pos0::Vec3
    quat0::Quat
    pos_inertia::Vec3
    quat_inertia::Quat
    rot_mat::Mat3

    # AMR 
    is_active::Bool
    level::Int
    parent_id::Int
    child_start::Int
    child_count::Int

    function Body(id, asm_id, static, pos, quat, size, mass; vel=Vec3(0, 0, 0), ang_vel=Vec3(0, 0, 0))
        R = quat_to_rotmat(quat)
        m_val = static ? Inf : mass
        inv_m = static ? 0.0 : 1.0 / m_val

        sx = size[1]
        sy = size[2]
        sz = size[3]
        ixx = (1.0 / 12.0) * m_val * (sy^2 + sz^2)
        iyy = (1.0 / 12.0) * m_val * (sx^2 + sz^2)
        izz = (1.0 / 12.0) * m_val * (sx^2 + sy^2)
        inertia_diag = Vec3(ixx, iyy, izz)
        inv_diag = static ? Vec3(0.0, 0.0, 0.0) : Vec3(1.0 / ixx, 1.0 / iyy, 1.0 / izz)

        inv_world = static ? zero(Mat3) : R * diag3(inv_diag) * transpose(R)

        # AMR DEFAULT values
        active = true
        level = 0
        parent = -1
        child_start = -1
        child_count = 0

        new(id, asm_id, static, pos, quat, size, vel, ang_vel,
            m_val, inv_m, inertia_diag, inv_diag, inv_world,
            pos, quat, pos, quat, pos, quat, R,
            active, level, parent, child_start, child_count)
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

mutable struct BroadPhaseState
    aabbs::Vector{AABB}
    endpoints::Vector{Endpoint}
    potential_pairs::Vector{Tuple{Int,Int}}
    active::Vector{Int}
    pair_count::Int
    capacity::Int
end

function BroadPhaseState(capacity::Int)
    cap = max(capacity, 4)
    aabbs = Vector{AABB}(undef, cap)
    endpoints = Vector{Endpoint}(undef, cap * 2)
    pair_cap = max(cap * (cap - 1) ÷ 2, 8)
    potential_pairs = Vector{Tuple{Int,Int}}(undef, pair_cap)
    active = Vector{Int}(undef, cap)
    return BroadPhaseState(aabbs, endpoints, potential_pairs, active, 0, cap)
end

# --- Helpers ---

@inline diag3(v::SVector{3,Float64}) = SMatrix{3,3,Float64,9}(v[1], 0.0, 0.0, 0.0, v[2], 0.0, 0.0, 0.0, v[3])

@inline update_rotation!(b::Body) = (b.rot_mat = quat_to_rotmat(b.quat))

@inline function update_inv_inertia_world!(b::Body)
    if b.is_static
        b.inv_inertia_world = zero(Mat3)
    else
        b.inv_inertia_world = b.rot_mat * diag3(b.inv_inertia_diag) * transpose(b.rot_mat)
    end
end

@inline get_axes(b::Body) = b.rot_mat
@inline get_center(b::Body) = b.pos
@inline get_half_extents(b::Body) = b.size * 0.5

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
    @inbounds for i in 2:8
        min_v = min.(min_v, corners[i])
        max_v = max.(max_v, corners[i])
    end
    return AABB(min_v, max_v)
end

@inline function canonical_axis(axis::Vec3)
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

# --- Broad Phase ---

function ensure_capacity!(state::BroadPhaseState, n_bodies::Int)
    if n_bodies <= state.capacity
        return
    end
    new_cap = max(state.capacity * 2, n_bodies)
    resize!(state.aabbs, new_cap)
    resize!(state.endpoints, new_cap * 2)
    resize!(state.active, new_cap)

    needed_pairs = max(new_cap * (new_cap - 1) ÷ 2, length(state.potential_pairs))
    resize!(state.potential_pairs, needed_pairs)
    state.capacity = new_cap
end

function broad_phase!(state::BroadPhaseState, bodies::Vector{Body})
    n = length(bodies)
    state.pair_count = 0
    if n < 2
        return 0
    end

    ensure_capacity!(state, n)

    aabbs = state.aabbs
    endpoints = state.endpoints

    @threads for i in 1:n
        aabbs[i] = get_aabb(bodies[i])
        endpoints[2i-1] = Endpoint(aabbs[i].min[1], true, i)
        endpoints[2i] = Endpoint(aabbs[i].max[1], false, i)
    end

    sort!(view(endpoints, 1:2n), by=e -> e.val)

    active = state.active
    active_count = 0
    pair_count = 0

    @inbounds for k in 1:(2n)
        ep = endpoints[k]
        idx = ep.body_idx
        if ep.is_min
            for j in 1:active_count
                other_idx = active[j]
                b1 = bodies[idx]
                b2 = bodies[other_idx]
                if b1.assembly_id >= 0 && b2.assembly_id >= 0 && b1.assembly_id == b2.assembly_id
                    continue
                end
                aabb1 = aabbs[idx]
                aabb2 = aabbs[other_idx]
                y_overlap = (aabb1.min[2] <= aabb2.max[2]) && (aabb2.min[2] <= aabb1.max[2])
                z_overlap = (aabb1.min[3] <= aabb2.max[3]) && (aabb2.min[3] <= aabb1.max[3])
                if y_overlap && z_overlap
                    pair_count += 1
                    state.potential_pairs[pair_count] = (idx, other_idx)
                end
            end
            active_count += 1
            active[active_count] = idx
        else
            for j in 1:active_count
                if active[j] == idx
                    active[j] = active[active_count]
                    active_count -= 1
                    break
                end
            end
        end
    end

    state.pair_count = pair_count
    return pair_count
end

function broad_phase_active!(state::BroadPhaseState, bodies::Vector{Body}, active_ids::Vector{Int})
    n = length(active_ids)
    state.pair_count = 0
    if n < 2
        return 0
    end

    ensure_capacity!(state, n)

    aabbs = state.aabbs
    endpoints = state.endpoints

    @threads for i in 1:n
        bi = active_ids[i]
        aabbs[i] = get_aabb(bodies[bi])
        endpoints[2i-1] = Endpoint(aabbs[i].min[1], true, i)
        endpoints[2i] = Endpoint(aabbs[i].max[1], false, i)
    end

    sort!(view(endpoints, 1:2n), by=e -> e.val)

    active = state.active
    active_count = 0
    pair_count = 0

    @inbounds for k in 1:(2n)
        ep = endpoints[k]
        idx = ep.body_idx
        if ep.is_min
            for j in 1:active_count
                other_idx = active[j]
                b1 = bodies[active_ids[idx]]
                b2 = bodies[active_ids[other_idx]]
                if b1.assembly_id >= 0 && b2.assembly_id >= 0 && b1.assembly_id == b2.assembly_id
                    continue
                end
                aabb1 = aabbs[idx]
                aabb2 = aabbs[other_idx]
                y_overlap = (aabb1.min[2] <= aabb2.max[2]) && (aabb2.min[2] <= aabb1.max[2])
                z_overlap = (aabb1.min[3] <= aabb2.max[3]) && (aabb2.min[3] <= aabb1.max[3])
                if y_overlap && z_overlap
                    pair_count += 1
                    state.potential_pairs[pair_count] = (active_ids[idx], active_ids[other_idx])
                end
            end
            active_count += 1
            active[active_count] = idx
        else
            for j in 1:active_count
                if active[j] == idx
                    active[j] = active[active_count]
                    active_count -= 1
                    break
                end
            end
        end
    end

    state.pair_count = pair_count
    return pair_count
end

# --- Narrow Phase Helpers ---

@inline function get_face_vertices(b::Body, axis_idx::Int, sign_val::Float64)
    R = b.rot_mat
    half = get_half_extents(b)
    center = b.pos + (R[:, axis_idx] * sign_val * half[axis_idx])

    idx1 = (axis_idx % 3) + 1
    idx2 = ((axis_idx + 1) % 3) + 1

    t1 = R[:, idx1] * half[idx1]
    t2 = R[:, idx2] * half[idx2]

    return SVector(center - t1 - t2, center + t1 - t2, center + t1 + t2, center - t1 + t2)
end

@inline function project_on_axis(b::Body, axis::Vec3)
    corners = get_corners(b)
    min_p = max_p = dot(corners[1], axis)
    @inbounds for i in 2:8
        val = dot(corners[i], axis)
        min_p = min(min_p, val)
        max_p = max(max_p, val)
    end
    return (min_p, max_p)
end

function sat_check(bA::Body, bB::Body)
    axesA = get_axes(bA)
    axesB = get_axes(bB)

    best_axis = MVector{3,Float64}(0.0, 0.0, 0.0)
    min_overlap = Inf

    @inline function check_axis(axis::Vec3)
        norm_axis = norm(axis)
        if norm_axis < TOL_AXIS
            return false
        end
        dir = canonical_axis(axis / norm_axis)

        mnA, mxA = project_on_axis(bA, dir)
        mnB, mxB = project_on_axis(bB, dir)

        if (mxA <= mnB) || (mxB <= mnA)
            return true
        end

        overlap = min(mxA, mxB) - max(mnA, mnB)
        if overlap < min_overlap
            min_overlap = overlap
            best_axis .= dir
        end
        return false
    end

    @inbounds for i in 1:3
        if check_axis(axesA[:, i])
            return (nothing, 0.0)
        end
    end
    @inbounds for i in 1:3
        if check_axis(axesB[:, i])
            return (nothing, 0.0)
        end
    end

    @inbounds for i in 1:3, j in 1:3
        if check_axis(cross(axesA[:, i], axesB[:, j]))
            return (nothing, 0.0)
        end
    end

    return (Vec3(best_axis[1], best_axis[2], best_axis[3]), min_overlap)
end

const POLY_CLIP_CAP = 16

@inline function clip_poly_to_plane!(poly::MVector{POLY_CLIP_CAP,Vec3}, poly_len::Int, normal::Vec3, offset::Float64, scratch::MVector{POLY_CLIP_CAP,Vec3})
    if poly_len == 0
        return 0
    end

    out_len = 0
    prev = poly[poly_len]
    prev_in = (dot(normal, prev) - offset) <= EPS_CLIP

    @inbounds for i in 1:poly_len
        curr = poly[i]
        curr_in = (dot(normal, curr) - offset) <= EPS_CLIP
        if curr_in
            if !prev_in
                denom = dot(normal, curr - prev)
                if abs(denom) > TOL_AXIS
                    t = (offset - dot(normal, prev)) / denom
                    out_len += 1
                    if out_len <= POLY_CLIP_CAP
                        scratch[out_len] = prev + t * (curr - prev)
                    end
                end
            end
            out_len += 1
            if out_len <= POLY_CLIP_CAP
                scratch[out_len] = curr
            end
        elseif prev_in
            denom = dot(normal, curr - prev)
            if abs(denom) > TOL_AXIS
                t = (offset - dot(normal, prev)) / denom
                out_len += 1
                if out_len <= POLY_CLIP_CAP
                    scratch[out_len] = prev + t * (curr - prev)
                end
            end
        end
        prev = curr
        prev_in = curr_in
    end

    if out_len > POLY_CLIP_CAP
        out_len = POLY_CLIP_CAP
    end

    unique_len = 0
    @inbounds for i in 1:out_len
        p = scratch[i]
        dup = false
        for j in 1:unique_len
            if norm(p - poly[j]) < TOL_PNT_DUP
                dup = true
                break
            end
        end
        if !dup
            unique_len += 1
            poly[unique_len] = p
        end
    end

    if unique_len > 1 && norm(poly[1] - poly[unique_len]) < TOL_PNT_DUP
        unique_len -= 1
    end

    return unique_len
end

function set_feature_id(bA::Body, bB::Body, p::Vec3)
    ida, idb = bA.id, bB.id
    px = round(p[1], digits=3)
    py = round(p[2], digits=3)
    pz = round(p[3], digits=3)
    h = hash((min(ida, idb), max(ida, idb), px, py, pz))
    return Int(h % typemax(Int))
end

# --- Manifold Generation ---

function build_manifold!(bA::Body, bB::Body, sat_normal::Vec3, overlap::Float64,
    contacts::Vector{Contact}, contact_count::Int)
    axesA = get_axes(bA)
    axesB = get_axes(bB)

    dotsA1 = dot(axesA[:, 1], sat_normal)
    dotsA2 = dot(axesA[:, 2], sat_normal)
    dotsA3 = dot(axesA[:, 3], sat_normal)
    idxA = abs(dotsA1) > abs(dotsA2) ? (abs(dotsA1) > abs(dotsA3) ? 1 : 3) : (abs(dotsA2) > abs(dotsA3) ? 2 : 3)

    dotsB1 = dot(axesB[:, 1], sat_normal)
    dotsB2 = dot(axesB[:, 2], sat_normal)
    dotsB3 = dot(axesB[:, 3], sat_normal)
    idxB = abs(dotsB1) > abs(dotsB2) ? (abs(dotsB1) > abs(dotsB3) ? 1 : 3) : (abs(dotsB2) > abs(dotsB3) ? 2 : 3)

    contact_normal = sat_normal
    ref_body = bA
    inc_body = bB
    ref_idx = idxA
    flip = false

    if abs(getindex((dotsA1, dotsA2, dotsA3), idxA)) < abs(getindex((dotsB1, dotsB2, dotsB3), idxB)) - TOL_AXIS
        ref_body, inc_body = bB, bA
        ref_idx = idxB
        contact_normal = -sat_normal
        flip = true
    end

    if dot(inc_body.pos - ref_body.pos, contact_normal) < 0
        contact_normal = -contact_normal
    end

    ref_R = ref_body.rot_mat
    raw_ref_axis = ref_R[:, ref_idx]
    ref_sign = sign(dot(raw_ref_axis, contact_normal))
    if ref_sign == 0.0
        ref_sign = 1.0
    end

    ref_normal = raw_ref_axis * ref_sign
    if dot(ref_normal, contact_normal) < 0
        ref_normal = -ref_normal
        ref_sign = -ref_sign
    end

    ref_face = get_face_vertices(ref_body, ref_idx, ref_sign)

    inc_R = inc_body.rot_mat
    dotsInc = (dot(inc_R[:, 1], -contact_normal), dot(inc_R[:, 2], -contact_normal), dot(inc_R[:, 3], -contact_normal))
    inc_idx = abs(dotsInc[1]) > abs(dotsInc[2]) ? (abs(dotsInc[1]) > abs(dotsInc[3]) ? 1 : 3) : (abs(dotsInc[2]) > abs(dotsInc[3]) ? 2 : 3)
    inc_sign = dot(inc_R[:, inc_idx], -contact_normal) >= 0.0 ? 1.0 : -1.0

    inc_face = get_face_vertices(inc_body, inc_idx, inc_sign)

    clipped_poly = MVector{POLY_CLIP_CAP,Vec3}(undef)
    @inbounds for i in 1:4
        clipped_poly[i] = inc_face[i]
    end
    poly_len = 4
    scratch = MVector{POLY_CLIP_CAP,Vec3}(undef)
    wind_dir = dot(cross(ref_face[2] - ref_face[1], ref_face[3] - ref_face[1]), ref_normal) >= 0 ? 1.0 : -1.0

    for i in 1:4
        v_a = ref_face[i]
        v_b = ref_face[mod1(i + 1, 4)]
        edge_ab = normalize(v_b - v_a)
        plane_norm = normalize(cross(edge_ab, ref_normal * wind_dir))
        poly_len = clip_poly_to_plane!(clipped_poly, poly_len, plane_norm, dot(plane_norm, v_a), scratch)
        if poly_len == 0
            return contact_count
        end
    end

    ref_plane_d = dot(ref_normal, ref_face[1])
    poly_len = clip_poly_to_plane!(clipped_poly, poly_len, ref_normal, ref_plane_d, scratch)
    if poly_len == 0
        return contact_count
    end

    local_points = MVector{POLY_CLIP_CAP,Vec3}(undef)
    local_depths = MVector{POLY_CLIP_CAP,Float64}(undef)
    local_count = 0

    @inbounds for i in 1:poly_len
        vert = clipped_poly[i]
        s = dot(ref_normal, vert) - ref_plane_d
        if s <= EPS_CLIP
            depth = max(0.0, -s)
            contact_p = vert - s * ref_normal
            local_count += 1
            if local_count <= POLY_CLIP_CAP
                local_points[local_count] = contact_p
                local_depths[local_count] = depth
            end
        end
    end

    if local_count > POLY_CLIP_CAP
        local_count = POLY_CLIP_CAP
    end

    if local_count == 0
        return contact_count
    end

    used = MVector{POLY_CLIP_CAP,Bool}(false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false)
    keep = min(local_count, 4)
    for _ in 1:keep
        best_idx = 1
        best_depth = -Inf
        @inbounds for i in 1:local_count
            if !used[i] && local_depths[i] > best_depth
                best_depth = local_depths[i]
                best_idx = i
            end
        end
        used[best_idx] = true

        idxA = flip ? inc_body.id : ref_body.id
        idxB = flip ? ref_body.id : inc_body.id
        final_n = flip ? -contact_normal : contact_normal
        fid = set_feature_id(bA, bB, local_points[best_idx])

        contact_count += 1
        if contact_count > length(contacts)
            resize!(contacts, max(length(contacts) * 2, contact_count + 8))
        end
        contacts[contact_count] = Contact(idxA, idxB, final_n, local_points[best_idx], local_depths[best_idx], fid)
    end

    return contact_count
end

function get_collisions!(state::BroadPhaseState, bodies::Vector{Body}, contacts::Vector{Contact})
    pair_count = broad_phase!(state, bodies)
    count = 0
    for i in 1:pair_count
        idxA, idxB = state.potential_pairs[i]
        bA = bodies[idxA]
        bB = bodies[idxB]
        if bA.is_static && bB.is_static
            continue
        end

        axis, overlap = sat_check(bA, bB)
        if axis !== nothing
            count = build_manifold!(bA, bB, axis, overlap, contacts, count)
        end
    end
    return count
end

function get_collisions_active!(state::BroadPhaseState, bodies::Vector{Body}, active_ids::Vector{Int}, contacts::Vector{Contact})
    pair_count = broad_phase_active!(state, bodies, active_ids)
    count = 0
    for i in 1:pair_count
        idxA, idxB = state.potential_pairs[i]
        bA = bodies[idxA]
        bB = bodies[idxB]
        if bA.is_static && bB.is_static
            continue
        end

        axis, overlap = sat_check(bA, bB)
        if axis !== nothing
            count = build_manifold!(bA, bB, axis, overlap, contacts, count)
        end
    end
    return count
end

function get_collisions(bodies::Vector{Body})
    n = length(bodies)
    state = BroadPhaseState(n)
    contacts = Vector{Contact}(undef, max(4 * n, 16))
    count = get_collisions!(state, bodies, contacts)
    return contacts[1:count]
end

function get_collisions_active(bodies::Vector{Body}, active_ids::Vector{Int})
    n = length(active_ids)
    state = BroadPhaseState(n)
    contacts = Vector{Contact}(undef, max(4 * n, 16))
    count = get_collisions_active!(state, bodies, active_ids, contacts)
    return contacts[1:count]
end

end # module
