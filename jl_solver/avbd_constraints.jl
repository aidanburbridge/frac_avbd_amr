"""
Constraint types and evaluation routines for the AVBD solver.

`ContactConstraint` stores persistent contact manifold data, while
`BondConstraint` stores cohesive voxel bonds that couple preprocessing output to
the Julia solver state.
"""
module AVBDConstraints

using LinearAlgebra
using StaticArrays
import ..Maths: Vec3, orthonormal_basis, rotate_vec, quat_to_rotmat, delta_twist_from
import ..Collisions: Body

export ContactConstraint, BondConstraint, initialize!, compute_constraint!, update_bounds!, eval_bond, update_bond_state!, bond_k_eff, get_effective_stiffness

const CONTACT_MARGIN = 1e-6
const CONTACT_K_MIN = 1e6
const CONTACT_K_MAX = 1e14

# --- Contact Constraints --- #

mutable struct ContactConstraint
    bodyA::Body
    bodyB::Body

    # Contact values
    point::Vec3
    normal::Vec3
    tangents::SVector{2,Vec3}
    depth::Float64
    mu::Float64
    feature_id::Int

    # AVBD Constraint values
    C::Vec3
    C0::Vec3
    JA::SMatrix{3,6,Float64,18}
    JB::SMatrix{3,6,Float64,18}

    # AVBD solver values
    lambda::Vec3 # Warm-start multiplier state.
    penalty_k::Vec3

    stiffness::Vec3 # Material E essentially ?
    k_min::Vec3
    k_max::Vec3

    # Friction bounds
    f_min::Vec3
    f_max::Vec3

    function ContactConstraint(bA::Body, bB::Body, p::Vec3, n::Vec3, depth::Float64, mu::Float64; feature_id::Int=0)
        zeros_Vec3 = @SVector zeros(3)
        zeros_J = @SMatrix zeros(3, 6)

        stiff = @SVector fill(Inf, 3)
        kmin = @SVector fill(CONTACT_K_MIN, 3)
        kmax = @SVector fill(CONTACT_K_MAX, 3)

        fmin = @SVector [0.0, -Inf, -Inf]
        fmax = @SVector [Inf, Inf, Inf]

        new(bA, bB, p, n, (@SVector [zeros_Vec3, zeros_Vec3]), depth, mu, feature_id, zeros_Vec3,
            zeros_Vec3, zeros_J, zeros_J, zeros_Vec3, zeros_Vec3, stiff, kmin, kmax, fmin, fmax)

    end
end

# --- Contact Constraint Functions --- #

function initialize!(con::ContactConstraint)
    t1, t2 = orthonormal_basis(con.normal)
    con.tangents = @SVector [t1, t2]

    dirs = @SVector [con.normal, t1, t2]


    # These points should be based on the collision detection (current geometry)
    # But Jacobians are often built around the PREDICTOR (Inertial) state in AVBD.
    # Assuming con.point and con.normal are from the current detection step.
    pA_world = con.point
    pB_world = pA_world - con.normal * con.depth

    rA0 = pA_world - con.bodyA.pos0
    rB0 = pB_world - con.bodyB.pos0

    # Build jacobians
    JA_row1 = vcat(dirs[1], cross(rA0, dirs[1]))'
    JA_row2 = vcat(dirs[2], cross(rA0, dirs[2]))'
    JA_row3 = vcat(dirs[3], cross(rA0, dirs[3]))'

    JB_row1 = vcat(-dirs[1], -cross(rB0, dirs[1]))'
    JB_row2 = vcat(-dirs[2], -cross(rB0, dirs[2]))'
    JB_row3 = vcat(-dirs[3], -cross(rB0, dirs[3]))'

    con.JA = vcat(JA_row1, JA_row2, JA_row3)
    con.JB = vcat(JB_row1, JB_row2, JB_row3)

    margin = CONTACT_MARGIN
    pA0 = con.bodyA.pos + rA0
    pB0 = con.bodyB.pos + rB0
    sep = pA0 - pB0

    c_n = dot(con.normal, sep) - margin
    c_t1 = dot(t1, sep)
    c_t2 = dot(t2, sep)

    con.C0 = @SVector [c_n, c_t1, c_t2]

    update_bounds!(con)

end

function compute_constraint!(con::ContactConstraint, alpha::Float64)
    dA = delta_twist_from(con.bodyA, con.bodyA.pos0, con.bodyA.quat0)
    dB = delta_twist_from(con.bodyB, con.bodyB.pos0, con.bodyB.quat0)

    con.C = (1.0 - alpha) * con.C0 + con.JA * dA + con.JB * dB
end

function update_bounds!(con::ContactConstraint)

    # Non-negative lambda
    lambda_mag = max(con.lambda[1], 0.0)

    # Normal is no penetration and no pulling
    fn_min = 0.0
    fn_max = Inf

    # Friction tangents
    ft_mag = con.mu * lambda_mag

    con.f_min = @SVector [fn_min, -ft_mag, -ft_mag]
    con.f_max = @SVector [fn_max, ft_mag, ft_mag]

end

# --- End Contact Constraints --- #


# --- Bond Constraints --- #

mutable struct BondConstraint
    id::Int
    bodyA::Body
    bodyB::Body

    # Anchor point values
    pA_local::Vec3
    pB_local::Vec3
    n_local::Vec3
    t1_local::Vec3
    t2_local::Vec3

    # AVBD solver values
    C::Vec3
    JA::SMatrix{3,6,Float64,18}
    JB::SMatrix{3,6,Float64,18}

    # AVBD solver state
    lambda::Vec3
    penalty_k::Vec3
    fracture::Vec3

    # Bond state
    rest::Vec3
    is_broken::Bool
    is_cohesive::Bool
    rest_initialized::Bool
    damage::Float64

    # Fracture history
    max_eff_strain::Float64
    current_eff_strain::Float64
    max_committed_strain::Float64 # Stored separately from the AVBD multiplier lambda.

    # Solver values
    k_eff::Vec3

    # Material config/values
    area::Float64
    stiffness::Vec3 # Undamaged bond stiffness.
    k_min::Vec3
    k_max::Vec3
    f_min::Vec3
    f_max::Vec3

    # CCM fracture limits derived from material params
    limits::SVector{4,Float64} # [delta_n0, delta_s0, delta_nc, delta_sc]

    # Stabilization parameters
    break_counter::Int
    max_break_steps::Int # Default limit; preprocessing may override from a wave-speed estimate.
    viscosity::Float64
    C_prev::Vec3

    # AMR 
    is_active::Bool


    function BondConstraint(bond_id::Int, bA::Body, bB::Body, pA::Vec3, pB::Vec3, n_world::Vec3, kn::Float64, kt::Float64, area::Float64, tensile::Float64, Gc::Float64, damp_val::Float64=0.0)

        d_n0 = (tensile * area) / kn
        d_s0 = (tensile * area) / kt

        # Cohesive failure separations derived from the material inputs.
        d_nc = (2.0 * Gc) / tensile
        d_sc = (2.0 * Gc) / tensile

        limits = @SVector [d_n0, d_s0, d_nc, d_sc]
        stiff = @SVector [kn, kt, kt]

        R_A = quat_to_rotmat(bA.quat)
        n_loc = transpose(R_A) * n_world
        t1_w, t2_w = orthonormal_basis(n_world)
        t1_loc = transpose(R_A) * t1_w
        t2_loc = transpose(R_A) * t2_w

        zeros_Vec3 = @SVector zeros(3)
        zeros_J = @SMatrix zeros(3, 6)

        fracture = @SVector fill(tensile * area, 3)
        fmin = -fracture
        fmax = fracture

        kmin = stiff #@SVector fill(0.0, 3)
        kmax = @SVector fill(1e12, 3)

        lambda = zeros_Vec3
        penalty_k = stiff

        # Conservative defaults until material-specific calibration is supplied.
        break_counter = 0
        max_break_steps = 10
        viscosity = 0.0

        # AMR
        active = true

        new(bond_id, bA, bB,
            pA, pB, n_loc, t1_loc, t2_loc,
            zeros_Vec3, zeros_J, zeros_J,
            lambda, penalty_k, fracture,
            zeros_Vec3, false, false, false, 0.0,
            0.0, 0.0, 0.0, stiff,
            area, stiff, kmin, kmax, fmin, fmax, limits,
            break_counter, max_break_steps, viscosity, zeros_Vec3, active)
    end
end

# --- Bond Constraint Functions --- #

function initialize!(con::BondConstraint)
    R_A = quat_to_rotmat(con.bodyA.quat)
    n_curr = R_A * con.n_local
    t1_curr = R_A * con.t1_local
    t2_curr = R_A * con.t2_local

    dirs = @SVector [n_curr, t1_curr, t2_curr]

    rA = rotate_vec(con.pA_local, con.bodyA.quat)
    rB = rotate_vec(con.pB_local, con.bodyB.quat)

    pA = con.bodyA.pos + rotate_vec(con.pA_local, con.bodyA.quat)
    pB = con.bodyB.pos + rotate_vec(con.pB_local, con.bodyB.quat)
    dp = pA - pB

    if !con.rest_initialized
        # Capture the undeformed bond coordinates on first use.
        con.rest = @SVector [dot(n_curr, dp), dot(t1_curr, dp), dot(t2_curr, dp)]
        con.rest_initialized = true

        # Previous constraint state for viscous regularization.
        con.C_prev = @SVector zeros(3)
    end

    # Build jacobians
    JA_row1 = vcat(dirs[1], cross(rA, dirs[1]))'
    JA_row2 = vcat(dirs[2], cross(rA, dirs[2]))'
    JA_row3 = vcat(dirs[3], cross(rA, dirs[3]))'
    con.JA = vcat(JA_row1, JA_row2, JA_row3)

    JB_row1 = vcat(-dirs[1], -cross(rB, dirs[1]))'
    JB_row2 = vcat(-dirs[2], -cross(rB, dirs[2]))'
    JB_row3 = vcat(-dirs[3], -cross(rB, dirs[3]))'
    con.JB = vcat(JB_row1, JB_row2, JB_row3)

end

function eval_bond(con::BondConstraint)

    rA = rotate_vec(con.pA_local, con.bodyA.quat)
    rB = rotate_vec(con.pB_local, con.bodyB.quat)
    pA = con.bodyA.pos + rA
    pB = con.bodyB.pos + rB
    dp = pA - pB

    rotA = quat_to_rotmat(con.bodyA.quat)
    n_curr = rotA * con.n_local
    t1_curr = rotA * con.t1_local
    t2_curr = rotA * con.t2_local

    # constraint violation
    c1 = dot(n_curr, dp) - con.rest[1]
    c2 = dot(t1_curr, dp) - con.rest[2]
    c3 = dot(t2_curr, dp) - con.rest[3]

    con.C = @SVector [c1, c2, c3]

    con.k_eff = get_effective_stiffness(con)

end


# --- End Bond Constraints --- #

# --- Helper Functions --- #

@inline function get_effective_stiffness(bond::BondConstraint)

    bond.is_broken && return @SVector zeros(3)

    d_factor = max(1e-3, 1.0 - clamp(bond.damage, 0.0, 0.999))

    # Bond normals point A -> B, while C[1] uses pA - pB. Opening is negative C[1].
    opening = bond.C[1] < 0.0
    k_n = opening ? (bond.stiffness[1] * d_factor) : bond.stiffness[1]

    k_t1 = bond.stiffness[2] * d_factor
    k_t2 = bond.stiffness[3] * d_factor

    return SVector{3,Float64}(k_n, k_t1, k_t2)
end

end # module end
