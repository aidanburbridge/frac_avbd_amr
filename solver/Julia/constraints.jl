module Constraints

using StaticArrays
using LinearAlgebra

# Import shared types from parent PhysicsCore modules
import ..Maths: Vec3, Mat3, Quat, FLOAT, transform_point, orthonormal_basis
import ..Collisions: Body, Contact

export ContactConstraint, FaceBond, Manifold
export update_manifold!, prepare_contacts!, prepare_bonds!, commit_bond_damage!
export make_bond, get_bond_system, get_contact_system

# --- Constants ---
const COLLISION_MARGIN = 5e-4
const BAUMGARTE = 0.1 # Stabilization factor
const MAX_SEP_FRAMES = 3

# ==============================================================================
# CONTACT CONSTRAINTS & MANIFOLD
# ==============================================================================

mutable struct ContactConstraint
    # Topology
    body_idx_a::Int
    body_idx_b::Int

    # Geometry (World Space)
    point::Vec3
    normal::Vec3         # Points A -> B
    tangent1::Vec3
    tangent2::Vec3
    depth::Float64

    # Solver State (Warm Starting)
    lambda_n::Float64    # Normal impulse
    lambda_t1::Float64   # Tangent 1 friction
    lambda_t2::Float64   # Tangent 2 friction

    # Solver Cache (Precomputed per frame)
    rA::Vec3             # Vector from Center A to Point
    rB::Vec3             # Vector from Center B to Point
    effective_mass::Mat3 # 3x3 block for reduced mass
    bias::Float64        # Position correction term
    friction::Float64

    # For spatial hashing / matching
    feature_id::Int      # From collision engine
    age::Int             # For persistence

    function ContactConstraint(c::Contact, friction::Float64)
        # Create basis
        t1, t2 = orthonormal_basis(c.normal)
        new(c.body_idx_a, c.body_idx_b,
            c.point, c.normal, t1, t2, c.depth,
            0.0, 0.0, 0.0,           # lambdas
            Vec3(0, 0, 0), Vec3(0, 0, 0), # rA, rB
            zeros(Mat3), 0.0, friction,
            c.feature_id, 0)
    end
end

mutable struct Manifold
    contacts::Vector{ContactConstraint}

    function Manifold()
        new(Vector{ContactConstraint}())
    end
end

# --- Manifold Logic (Warm Starting) ---

function match_contact(c::Contact, old_cons::Vector{ContactConstraint})
    # Heuristic: Match feature ID first, then spatial proximity
    best_idx = -1
    min_dist_sq = (2e-3)^2 # 2mm tolerance squared

    for (i, old) in enumerate(old_cons)
        if old.feature_id == c.feature_id
            return i
        end

        # Proximity fallback
        dist_sq = sum(abs2, c.point - old.point)
        if dist_sq < min_dist_sq
            # Also check normal alignment
            if dot(c.normal, old.normal) > 0.98
                min_dist_sq = dist_sq
                best_idx = i
            end
        end
    end
    return best_idx
end

function update_manifold!(manifold::Manifold, new_collision_data::Vector{Contact}, friction::Float64)
    old_contacts = manifold.contacts
    new_contacts = Vector{ContactConstraint}()
    sizehint!(new_contacts, length(new_collision_data))

    # 1. Process new collisions
    for c_raw in new_collision_data
        idx = match_contact(c_raw, old_contacts)

        new_con = ContactConstraint(c_raw, friction)

        if idx != -1
            # Warm Start: Copy accumulated impulses
            old = old_contacts[idx]
            new_con.lambda_n = old.lambda_n
            new_con.lambda_t1 = old.lambda_t1
            new_con.lambda_t2 = old.lambda_t2
            new_con.age = 0 # Reset age

            # Remove from old list so we don't process it again (simple swap-remove if order doesn't matter)
            # For correctness in this simple loop, we just mark it? 
            # Optimization: We rebuild the list anyway.
        end
        push!(new_contacts, new_con)
    end

    # 2. Persistence (Optional: keep separating contacts for a few frames)
    # Note: In the simplified version, we just drop them if they aren't colliding.
    # To implement the "separation_frames" logic from Python, we would check 
    # overlap here. For high-speed simulation, usually just keeping actual contacts is fine.

    manifold.contacts = new_contacts
end

# --- Solver Prep ---

function prepare_contacts!(manifold::Manifold, bodies::Vector{Body}, dt::Float64)
    inv_dt = 1.0 / dt

    for con in manifold.contacts
        bA = bodies[con.body_idx_a+1] # Julia 1-based indexing
        bB = bodies[con.body_idx_b+1]

        # 1. Geometry / Lever Arms
        con.rA = con.point - bA.pos
        con.rB = con.point - bB.pos

        # 2. Baumgarte Stabilization (Bias)
        # C = dot(n, pA - pB) -> we want C >= 0 (no penetration)
        # Actually in the constraint solver: Jv + b = 0
        # b = (beta / dt) * C
        penetration = max(0.0, con.depth - COLLISION_MARGIN)
        con.bias = (BAUMGARTE * inv_dt) * penetration

        # 3. (Optional) Precompute Effective Mass if using PGS
        # K = J * M^-1 * J^T
        # We skip this if using the Cholesky solver approach, 
        # as the Cholesky solver builds the global H matrix directly.
    end
end

# ==============================================================================
# FACE BONDS (Fracture)
# ==============================================================================

mutable struct FaceBond
    # IDs
    body_idx_a::Int
    body_idx_b::Int

    # Local Anchors (Relative to Body Centroid)
    pA_local::Vec3
    pB_local::Vec3

    # Local Basis (Attached to Body A)
    n_local::Vec3
    t1_local::Vec3
    t2_local::Vec3

    # State
    is_broken::Bool
    is_cohesive::Bool
    damage::Float64
    lam_max_committed::Float64
    lam_current::Float64

    # Parameters
    stiffness::Vec3     # kn, kt, kt
    rest_length::Vec3   # Computed at bind time
    limits::SVector{4,Float64} # delta_n0, delta_s0, delta_nc, delta_sc
    age::Int

    function FaceBond(idxA, idxB, pA_loc, pB_loc, n_world, k_n, k_t, area, tensile, fracture_E)
        # We need the bodies to convert n_world to n_local (done at init time in Python usually)
        # Assuming arguments are already transformed or we transform them inside an init wrapper.

        # Placeholder basis (updated in init)
        new(idxA, idxB, pA_loc, pB_loc, Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0),
            false, false, 0.0, 0.0, 0.0,
            Vec3(k_n, k_t, k_t), Vec3(0, 0, 0),
            SVector(0.0, 0.0, 0.0, 0.0), 0)
    end
end

# Helper to build from flat arrays passed from Python
function make_bond(idxA, idxB, pA, pB, n_world, params, bodyA::Body)
    # Unpack params: [kn, kt, area, tensile, Gc]
    kn, kt, area, tensile, Gc = params

    # Compute limits
    delta_n0 = (tensile * area) / kn
    delta_s0 = (tensile * area) / kt
    delta_nc = (2.0 * Gc) / tensile
    delta_sc = (2.0 * Gc) / tensile

    bond = FaceBond(idxA, idxB, pA, pB, n_world, kn, kt, area, tensile, Gc)
    bond.limits = SVector(delta_n0, delta_s0, delta_nc, delta_sc)

    # Set Local Basis relative to Body A
    # n_local = R_A^T * n_world
    R_A = bodyA.rot_mat
    bond.n_local = transpose(R_A) * n_world

    # Tangents
    t1_w, t2_w = orthonormal_basis(n_world)
    bond.t1_local = transpose(R_A) * t1_w
    bond.t2_local = transpose(R_A) * t2_w

    return bond
end

function prepare_bonds!(bonds::Vector{FaceBond}, bodies::Vector{Body})
    # Check fracture and compute current stiffness/forces

    for bond in bonds
        if bond.is_broken
            continue
        end

        bA = bodies[bond.body_idx_a+1]
        bB = bodies[bond.body_idx_b+1]

        # 1. Current Basis (World Space)
        # R_A * n_local
        n_curr = bA.rot_mat * bond.n_local
        t1_curr = bA.rot_mat * bond.t1_local
        t2_curr = bA.rot_mat * bond.t2_local

        # 2. Current Separation
        pA_w = transform_point(bond.pA_local, bA.pos, bA.quat)
        pB_w = transform_point(bond.pB_local, bB.pos, bB.quat)
        dp = pA_w - pB_w

        # Project onto basis
        dn = dot(n_curr, dp)
        ds1 = dot(t1_curr, dp)
        ds2 = dot(t2_curr, dp)

        # Init rest length if first frame (hack: check if zero)
        if norm(bond.rest_length) == 0.0 && bond.age == 0 # You might need an explicit init flag
            bond.rest_length = Vec3(dn, ds1, ds2)
        end

        # Deviations
        err_n = dn - bond.rest_length[1]
        err_s1 = ds1 - bond.rest_length[2]
        err_s2 = ds2 - bond.rest_length[3]

        # 3. Fracture Mechanics
        delta_n0, delta_s0, delta_nc, delta_sc = bond.limits

        d_n_val = max(err_n, 0.0) # Only tension causes damage
        d_s_val = sqrt(err_s1^2 + err_s2^2)

        # Psi check (Crack initiation)
        if !bond.is_cohesive
            Psi = (d_n_val / delta_n0)^2 + (d_s_val / delta_s0)^2
            if Psi >= 1.0
                bond.is_cohesive = true
                # Initial lambda
                bond.lam_max_committed = sqrt(Psi) # Approx
            end
        end

        current_k_scale = 1.0

        if bond.is_cohesive
            # Mixed mode effective displacement
            # Simplified version of the Python logic
            lam = sqrt((d_n_val / delta_nc)^2 + (d_s_val / delta_sc)^2)
            bond.lam_current = max(bond.lam_max_committed, lam)

            if bond.lam_current >= 1.0
                bond.is_broken = true
                current_k_scale = 0.0
            else
                # Softening function
                # Linear damage evolution
                current_k_scale = (1.0 - bond.lam_current) / (1.0 - (delta_n0 / delta_nc)) # Simplified
                current_k_scale = clamp(current_k_scale, 0.0, 1.0)
                bond.damage = 1.0 - current_k_scale
            end
        else
            bond.damage = 0.0
        end

        bond.age += 1
    end
end

function commit_bond_damage!(bonds::Vector{FaceBond})
    for bond in bonds
        if bond.is_cohesive && !bond.is_broken
            bond.lam_max_committed = max(bond.lam_max_committed, bond.lam_current)
            if bond.lam_max_committed >= 1.0
                bond.is_broken = true
            end
        end
    end
end

# --- Assembly of constraint rows for the global solver ---

@inline function _build_J_row(dir::Vec3, rA::Vec3, rB::Vec3)
    # Match Python Jacobian rows: A [+dir, +rA×dir], B [-dir, -rB×dir]
    angA = cross(rA, dir)
    angB = cross(rB, dir)
    J = MVector{12,Float64}(undef)
    J[1] = dir[1];  J[2] = dir[2];  J[3] = dir[3]
    J[4] = angA[1]; J[5] = angA[2]; J[6] = angA[3]
    J[7] = -dir[1]; J[8] = -dir[2]; J[9] = -dir[3]
    J[10] = -angB[1]; J[11] = -angB[2]; J[12] = -angB[3]
    return SVector{12,Float64}(J)
end

@inline function _accumulate_constraint!(H::Matrix{Float64}, f::Vector{Float64},
    dir::Vec3, err::Float64, k::Float64, rA::Vec3, rB::Vec3)
    if k == 0.0
        return
    end
    J = _build_J_row(dir, rA, rB)
    H .+= k .* (J * transpose(J))
    f .+= (-k * err) .* J
end

@inline function bond_basis_world(bond::FaceBond, bA::Body)
    n_curr = bA.rot_mat * bond.n_local
    t1_curr = bA.rot_mat * bond.t1_local
    t2_curr = bA.rot_mat * bond.t2_local
    return n_curr, t1_curr, t2_curr
end

function get_bond_system(bond::FaceBond, bodies::Vector{Body})
    if bond.is_broken
        return zeros(Float64, 12, 12), zeros(Float64, 12)
    end

    bA = bodies[bond.body_idx_a+1]
    bB = bodies[bond.body_idx_b+1]

    # Anchors and lever arms
    pA_w = transform_point(bond.pA_local, bA.pos, bA.quat)
    pB_w = transform_point(bond.pB_local, bB.pos, bB.quat)
    rA = pA_w - bA.pos
    rB = pB_w - bB.pos

    n_curr, t1_curr, t2_curr = bond_basis_world(bond, bA)
    dp = pA_w - pB_w

    if norm(bond.rest_length) == 0.0 && bond.age <= 1
        bond.rest_length = Vec3(dot(n_curr, dp), dot(t1_curr, dp), dot(t2_curr, dp))
    end

    err_n = dot(n_curr, dp) - bond.rest_length[1]
    err_s1 = dot(t1_curr, dp) - bond.rest_length[2]
    err_s2 = dot(t2_curr, dp) - bond.rest_length[3]

    k_scale = (bond.is_cohesive || bond.damage > 0.0) ? (1.0 - bond.damage) : 1.0
    k_vec = bond.stiffness .* k_scale

    H = zeros(Float64, 12, 12)
    f = zeros(Float64, 12)

    _accumulate_constraint!(H, f, n_curr, err_n, k_vec[1], rA, rB)
    _accumulate_constraint!(H, f, t1_curr, err_s1, k_vec[2], rA, rB)
    _accumulate_constraint!(H, f, t2_curr, err_s2, k_vec[3], rA, rB)

    return H, f
end

function get_contact_system(con::ContactConstraint, bodies::Vector{Body})
    bA = bodies[con.body_idx_a+1]
    bB = bodies[con.body_idx_b+1]

    # Normal-only penalty for stability (tunable)
    k_n = 5e5
    H = zeros(Float64, 12, 12)
    f = zeros(Float64, 12)

    # Limit the effective penetration we respond to so deep interpenetrations don't explode
    err = min(con.depth + con.bias, COLLISION_MARGIN * 10)
    if err > 0
        dir = con.normal
        rA = bA.is_static ? Vec3(0, 0, 0) : con.rA
        rB = bB.is_static ? Vec3(0, 0, 0) : con.rB
        _accumulate_constraint!(H, f, dir, err, k_n, rA, rB)
    end

    return H, f
end

# --- Helpers ---

end # module
