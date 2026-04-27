module Constraints

using LinearAlgebra
using StaticArrays

import ..Maths: Vec3, Mat3, FLOAT, transform_point, orthonormal_basis
import ..Collisions: Body, Contact

export ContactConstraint, FaceBond, prepare_contact!, solve_contact!, prepare_bonds!, solve_bond!, commit_bond_damage!, make_bond, bond_basis_world

# --- Constants ---
const COLLISION_MARGIN = 5e-4
const BAUMGARTE = 0.1
const EPS_MASS = 1e-9

const ZERO_MAT3 = SMatrix{3,3,Float64,9}(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# ==============================================================================
# CONTACT CONSTRAINTS
# ==============================================================================

mutable struct ContactConstraint
    body_idx_a::Int
    body_idx_b::Int

    point::Vec3
    normal::Vec3
    tangent1::Vec3
    tangent2::Vec3
    depth::Float64

    lambda_n::Float64
    lambda_t1::Float64
    lambda_t2::Float64

    effective_mass::SMatrix{3,3,Float64,9}
    rA::Vec3
    rB::Vec3
    bias::Float64
    friction_coeff::Float64
    feature_id::Int
end

@inline function ContactConstraint()
    return ContactConstraint(
        0, 0,
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        0.0,
        0.0, 0.0, 0.0,
        ZERO_MAT3,
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 0.0),
        0.0,
        0.5,
        0
    )
end

@inline function reset_contact!(con::ContactConstraint, c::Contact, friction::Float64)
    con.body_idx_a = c.body_idx_a
    con.body_idx_b = c.body_idx_b
    con.point = c.point
    con.normal = c.normal
    t1, t2 = orthonormal_basis(c.normal)
    con.tangent1 = t1
    con.tangent2 = t2
    con.depth = c.depth
    con.friction_coeff = friction
    con.feature_id = c.feature_id
end

@inline function effective_mass_entry(dir_i::Vec3, dir_j::Vec3, bA::Body, bB::Body, rA::Vec3, rB::Vec3)
    invm = bA.inv_mass + bB.inv_mass
    angA = cross(rA, dir_j)
    angB = cross(rB, dir_j)
    termA = bA.inv_inertia_world * angA
    termB = bB.inv_inertia_world * angB
    ang_term = dot(cross(termA, rA), dir_i) + dot(cross(termB, rB), dir_i)
    return invm * dot(dir_i, dir_j) + ang_term
end

function prepare_contact!(con::ContactConstraint, bA::Body, bB::Body, dt::Float64)
    con.rA = con.point - bA.pos
    con.rB = con.point - bB.pos

    n = con.normal
    t1 = con.tangent1
    t2 = con.tangent2

    k11 = effective_mass_entry(n, n, bA, bB, con.rA, con.rB)
    k12 = effective_mass_entry(n, t1, bA, bB, con.rA, con.rB)
    k13 = effective_mass_entry(n, t2, bA, bB, con.rA, con.rB)
    k22 = effective_mass_entry(t1, t1, bA, bB, con.rA, con.rB)
    k23 = effective_mass_entry(t1, t2, bA, bB, con.rA, con.rB)
    k33 = effective_mass_entry(t2, t2, bA, bB, con.rA, con.rB)

    K = @SMatrix [k11 k12 k13; k12 k22 k23; k13 k23 k33]
    con.effective_mass = inv(K + EPS_MASS * I)

    penetration = max(0.0, con.depth - COLLISION_MARGIN)
    con.bias = (BAUMGARTE / dt) * penetration
end

function solve_contact!(con::ContactConstraint, bA::Body, bB::Body)
    if bA.is_static && bB.is_static
        return
    end

    vA = bA.vel + cross(bA.ang_vel, con.rA)
    vB = bB.vel + cross(bB.ang_vel, con.rB)
    v_rel = vB - vA

    rel_vec = SVector(dot(con.normal, v_rel), dot(con.tangent1, v_rel), dot(con.tangent2, v_rel))
    rhs = rel_vec .* -1.0 + SVector(con.bias, 0.0, 0.0)

    delta = con.effective_mass * rhs

    lam_n = con.lambda_n + delta[1]
    lam_t1 = con.lambda_t1 + delta[2]
    lam_t2 = con.lambda_t2 + delta[3]

    lam_n = max(0.0, lam_n)
    t_limit = con.friction_coeff * lam_n
    lam_t1 = clamp(lam_t1, -t_limit, t_limit)
    lam_t2 = clamp(lam_t2, -t_limit, t_limit)

    d_lam = SVector(lam_n - con.lambda_n, lam_t1 - con.lambda_t1, lam_t2 - con.lambda_t2)
    con.lambda_n = lam_n
    con.lambda_t1 = lam_t1
    con.lambda_t2 = lam_t2

    impulse = con.normal * d_lam[1] + con.tangent1 * d_lam[2] + con.tangent2 * d_lam[3]

    if !bA.is_static
        bA.vel -= impulse * bA.inv_mass
        bA.ang_vel -= bA.inv_inertia_world * cross(con.rA, impulse)
    end
    if !bB.is_static
        bB.vel += impulse * bB.inv_mass
        bB.ang_vel += bB.inv_inertia_world * cross(con.rB, impulse)
    end
end

# ==============================================================================
# FACE BONDS (FRACTURE)
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
    max_eff_strain::Float64
    current_eff_strain::Float64

    # Parameters
    stiffness::Vec3
    current_k::Vec3     # As per AVBD ramping stiffness
    rest_length::Vec3
    limits::SVector{4,Float64}
    age::Int

    # Solver cache
    n_curr::Vec3
    t1_curr::Vec3
    t2_curr::Vec3
    rA::Vec3
    rB::Vec3
    effective_mass::SMatrix{3,3,Float64,9}

    #bias_vec::SVector{3,Float64}
    #lambda::SVector{3,Float64}

    function FaceBond(idxA, idxB, pA_loc, pB_loc, n_world, k_n, k_t, area, tensile, fracture_E)
        new(idxA, idxB, pA_loc, pB_loc, Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0),
            false, false, 0.0, 0.0, 0.0,
            Vec3(k_n, k_t, k_t), Vec3(k_n, k_t, k_t), Vec3(0.0, 0.0, 0.0),
            SVector(0.0, 0.0, 0.0, 0.0), 0,
            Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0),
            ZERO_MAT3)
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
    bond.rest_length = Vec3(0.0, 0.0, 0.0)

    # Set Local Basis relative to Body A
    R_A = bodyA.rot_mat
    bond.n_local = transpose(R_A) * n_world

    t1_w, t2_w = orthonormal_basis(n_world)
    bond.t1_local = transpose(R_A) * t1_w
    bond.t2_local = transpose(R_A) * t2_w

    return bond
end

@inline function bond_basis_world(bond::FaceBond, bA::Body)
    n_curr = bA.rot_mat * bond.n_local
    t1_curr = bA.rot_mat * bond.t1_local
    t2_curr = bA.rot_mat * bond.t2_local
    return n_curr, t1_curr, t2_curr
end

@inline function compliance_val(k::Float64, dt::Float64)
    return k > 0.0 ? (1.0 / k) / (dt * dt) : 1e12
end

function build_K(bA::Body, bB::Body, rA::Vec3, rB::Vec3, dirs::NTuple{3,Vec3})
    K = @MMatrix zeros(3, 3)
    invm = bA.inv_mass + bB.inv_mass
    @inbounds for i in 1:3
        di = dirs[i]
        for j in i:3
            dj = dirs[j]
            base = invm * dot(di, dj)

            angA = bA.inv_inertia_world * cross(rA, dj)
            angB = bB.inv_inertia_world * cross(rB, dj)
            ang_term = dot(cross(angA, rA), di) + dot(cross(angB, rB), di)

            val = base + ang_term
            K[i, j] = val
            K[j, i] = val
        end
    end
    return SMatrix{3,3,Float64,9}(K)
end

function prepare_bonds!(bonds::Vector{FaceBond}, bodies::Vector{Body}, dt::Float64)
    for bond in bonds
        if bond.is_broken
            continue
        end

        bA = bodies[bond.body_idx_a+1]
        bB = bodies[bond.body_idx_b+1]

        bond.n_curr, bond.t1_curr, bond.t2_curr = bond_basis_world(bond, bA)

        pA_w = transform_point(bond.pA_local, bA.pos, bA.quat)
        pB_w = transform_point(bond.pB_local, bB.pos, bB.quat)
        bond.rA = pA_w - bA.pos
        bond.rB = pB_w - bB.pos

        k_scale = (bond.is_cohesive || bond.damage > 0.0) ? (1.0 - bond.damage) : 1.0
        bond.current_k = bond.stiffness .* k_scale

        if k_scale <= 0.0
            bond.effective_mass = ZERO_MAT3
            continue
        end

        dirs = (bond.n_curr, bond.t1_curr, bond.t2_curr)
        K = build_K(bA, bB, bond.rA, bond.rB, dirs)

        comp_n = compliance_val(bond.current_k[1], dt)
        comp_t1 = compliance_val(bond.current_k[2], dt)
        comp_t2 = compliance_val(bond.current_k[3], dt)
        compliance = @SMatrix [comp_n 0.0 0.0; 0.0 comp_t1 0.0; 0.0 0.0 comp_t2]

        bond.effective_mass = inv(K + compliance + EPS_MASS * I)

        bond.age += 1
    end
end

function solve_bond!(bond::FaceBond, bA::Body, bB::Body, dt::Float64)
    if bond.is_broken
        return
    end

    if bond.effective_mass == ZERO_MAT3
        return
    end

    # Calculate constraint error
    pA_w = bA.pos + bond.rA
    pB_w = bB.pos + bond.rB
    dp = pA_w - pB_w

    err_n = dot(bond.n_curr, dp) - bond.rest_length[1]
    err_s1 = dot(bond.t1_curr, dp) - bond.rest_length[2]
    err_s2 = dot(bond.t2_curr, dp) - bond.rest_length[3]

    bias = SVector(err_n, err_s1, err_s2) * (BAUMGARTE / dt)

    vA = bA.vel + cross(bA.ang_vel, bond.rA)
    vB = bB.vel + cross(bB.ang_vel, bond.rB)
    v_rel = vA - vB

    rel_vec = SVector(dot(bond.n_curr, v_rel), dot(bond.t1_curr, v_rel), dot(bond.t2_curr, v_rel))
    delta = bond.effective_mass * (-bias - rel_vec)

    impulse = bond.n_curr * delta[1] + bond.t1_curr * delta[2] + bond.t2_curr * delta[3]

    if !bA.is_static
        bA.vel += impulse * bA.inv_mass
        bA.ang_vel += bA.inv_inertia_world * cross(bond.rA, impulse)
    end
    if !bB.is_static
        bB.vel -= impulse * bB.inv_mass
        bB.ang_vel -= bB.inv_inertia_world * cross(bond.rB, impulse)
    end
end

function commit_bond_damage!(bonds::Vector{FaceBond}, bodies::Vector{Body})
    for bond in bonds
        if bond.is_broken
            continue
        end

        bA = bodies[bond.body_idx_a+1]
        bB = bodies[bond.body_idx_b+1]

        n_curr = bA.rot_mat * bond.n_local
        t1_curr = bA.rot_mat * bond.t1_local
        t2_curr = bA.rot_mat * bond.t2_local

        pA_w = transform_point(bond.pA_local, bA.pos, bA.quat)
        pB_w = transform_point(bond.pB_local, bB.pos, bB.quat)
        dp = pA_w - pB_w

        err_n = dot(n_curr, dp) - bond.rest_length[1]
        err_s1 = dot(t1_curr, dp) - bond.rest_length[2]
        err_s2 = dot(t2_curr, dp) - bond.rest_length[3]

        delta_n0, delta_s0, delta_nc, delta_sc = bond.limits

        d_n_val = max(err_n, 0.0)
        d_s_val = sqrt(err_s1^2 + err_s2^2)

        # Update strain for paraview
        strain = sqrt((d_n_val / delta_nc)^2 + (d_s_val / delta_sc)^2)
        bond.current_eff_strain = strain

        if !bond.is_cohesive
            Psi = (d_n_val / delta_n0)^2 + (d_s_val / delta_s0)^2
            if Psi >= 1.0
                bond.is_cohesive = true
                bond.max_eff_strain = sqrt(Psi)
            end
        end

        if bond.is_cohesive
            # Track the maximum committed effective strain for irreversible damage.
            bond.max_eff_strain = max(bond.max_eff_strain, strain)

            if bond.current_eff_strain >= 1.0
                bond.is_broken = true
                bond.damage = 1.0
            else
                strain_cr = (delta_n0 / delta_nc)
                if d_s_val > 1e-9
                    strain_cr = sqrt(((d_n_val / delta_nc)^2 + (d_s_val / delta_sc)^2) /
                                     ((d_n_val / delta_n0)^2 + (d_s_val / delta_s0)^2))
                end

                if bond.max_eff_strain > strain_cr
                    bond.damage = clamp((bond.max_eff_strain - strain_cr) / (1.0 - strain_cr), 0.0, 1.0)
                else
                    bond.damage = 0.0
                end
            end
        else
            bond.damage = 0.0
        end
    end
end

end # module
