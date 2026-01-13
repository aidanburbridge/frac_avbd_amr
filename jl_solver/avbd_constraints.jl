module AVBDConstraints

using LinearAlgebra
using StaticArrays
import ..Maths: Vec3, Quat, transform_point, quat_mul, integrate_quat, orthonormal_basis, rotate_vec, quat_to_rotmat, quat_inv, quat_to_rotvec, delta_twist_from
import ..Collisions: Body

export ContactConstraint, BondConstraint, initialize!, compute_constraint!, update_bounds!, commit_bond_damage!, eval_bond, update_bond_state!, bond_k_eff #AbstractConstraint, 

##### ---------- Contact Constraint ---------- #####

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
    lambda::Vec3 # TODO Removed is_hard and will instead just set lambda 0 for non-hard constraints -> No check stiffness is Inf for hard or not
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
        kmin = @SVector fill(1.0, 3)
        kmax = @SVector fill(1e12, 3) # TODO Maybe we make this Inf -> could make Inf 

        fmin = @SVector [0.0, -Inf, -Inf]
        fmax = @SVector [Inf, Inf, Inf]

        new(bA, bB, p, n, (@SVector [zeros_Vec3, zeros_Vec3]), depth, mu, feature_id, zeros_Vec3,
            zeros_Vec3, zeros_J, zeros_J, zeros_Vec3, zeros_Vec3, stiff, kmin, kmax, fmin, fmax)

    end
end

# ---------- Contact Constraint Functions ---------- #

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

    margin = 5e-4
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

# TODO do I need to calculate JA and JB?? It's normally just cached, calculated in initialize?? -> No, only for non-linear but ignore for now.
function compute_JA!(con::ContactConstraint)
end

function compute_JB!(con::ContactConstraint)
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

##### ---------- End Contact Constraint ---------- #####


##### ---------- Bond Constraint ---------- #####

mutable struct BondConstraint
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

    # Bond state
    rest::Vec3
    is_broken::Bool
    is_cohesive::Bool
    rest_initialized::Bool
    damage::Float64

    # Fracture histroy
    max_eff_strain::Float64
    current_eff_strain::Float64
    max_committed_strain::Float64 # CCM calls this lambda, but switched for 'strain' to avoid confusion

    # Solver values
    k_tension::Vec3
    k_compression::Vec3

    # Material config/values
    stiffness::Vec3 # base material stiffness (E) will NOT be damaged, use for compression
    k_min::Vec3
    k_max::Vec3
    f_min::Vec3
    f_max::Vec3

    # CCM fracture limits derived from material params
    limits::SVector{4,Float64} # [delta_n0, delta_s0, delta_nc, delta_sc]

    function BondConstraint(bA::Body, bB::Body, pA::Vec3, pB::Vec3, n_world::Vec3, kn::Float64, kt::Float64, area::Float64, tensile::Float64, Gc::Float64)

        d_n0 = (tensile * area) / kn
        d_s0 = (tensile * area) / kt

        # Critical strain value - notable for fracture
        d_nc = (2.0 * Gc) / tensile
        d_sc = (2.0 * Gc) / tensile

        limits = @SVector [d_n0, d_s0, d_nc, d_sc]
        stiff = @SVector [kn, kt, kt] # These will be corrected with SINDy - cool beans man

        R_A = quat_to_rotmat(bA.quat)
        n_loc = transpose(R_A) * n_world
        t1_w, t2_w = orthonormal_basis(n_world)
        t1_loc = transpose(R_A) * t1_w
        t2_loc = transpose(R_A) * t2_w

        zeros_Vec3 = @SVector zeros(3)
        zeros_J = @SMatrix zeros(3, 6)

        fmin = @SVector fill(-Inf, 3)
        fmax = @SVector fill(Inf, 3)
        kmin = @SVector fill(0.0, 3)
        kmax = @SVector fill(1e12, 3)

        new(bA, bB, pA, pB, n_loc, t1_loc, t2_loc, zeros_Vec3, zeros_J, zeros_J,
            zeros_Vec3, false, false, false, 0.0, 0.0, 0.0, 0.0, stiff, stiff,
            stiff, kmin, kmax, fmin, fmax, limits)

    end
end

# ---------- Bond Constraint Functions ---------- #

function initialize!(con::BondConstraint) #TODO do I put in the body list here? bonds do not have bodies?
    # if con.is_broken
    #     con.k_tension = @SVector zeros(3)

    #     return
    # end

    R_A = quat_to_rotmat(con.bodyA.quat) #Unless body has update rotmat in struct
    n_curr = R_A * con.n_local
    t1_curr = R_A * con.t1_local
    t2_curr = R_A * con.t2_local

    dirs = @SVector [n_curr, t1_curr, t2_curr]

    rA = rotate_vec(con.pA_local, con.bodyA.quat)
    rB = rotate_vec(con.pB_local, con.bodyB.quat)

    #TODO do I need to initialize the rest length like I do in python?

    pA = con.bodyA.pos + rotate_vec(con.pA_local, con.bodyA.quat)
    pB = con.bodyB.pos + rotate_vec(con.pB_local, con.bodyB.quat)
    dp = pA - pB

    if !con.rest_initialized
        con.rest = @SVector [dot(n_curr, dp), dot(t1_curr, dp), dot(t2_curr, dp)]
        con.rest_initialized = true
        con.k_tension = con.stiffness * (1.0 - con.damage)
        con.k_compression = con.stiffness
    end

    if con.is_broken
        con.k_tension = @SVector zeros(3)
        con.k_compression = con.stiffness # Ensure compression is ready
    else
        con.k_tension = con.stiffness * (1.0 - con.damage)
        con.k_compression = con.stiffness
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
    # This function purely evaluates C, JA, JB, and K_eff
    # Does NOT modify any values

    # if con.is_broken
    #     z3 = @SVector zeros(3)
    #     zJ = @SMatrix zeros(3, 6)
    #     return z3, zJ, zJ, z3
    # end

    # Read-Only kinematic values
    rA = rotate_vec(con.pA_local, con.bodyA.quat)
    rB = rotate_vec(con.pB_local, con.bodyB.quat)
    pA = con.bodyA.pos + rA
    pB = con.bodyB.pos + rB
    dp = pA - pB

    rotA = quat_to_rotmat(con.bodyA.quat)
    n_curr = rotA * con.n_local
    t1_curr = rotA * con.t1_local
    t2_curr = rotA * con.t2_local

    dirs = @SVector [n_curr, t1_curr, t2_curr]

    # Helper to build 3x6 J for a body
    function make_J(r_vec, sign)
        rows = MMatrix{3,6,Float64}(undef)
        for i in 1:3
            rows[i, 1:3] = dirs[i] * sign
            rows[i, 4:6] = cross(r_vec, dirs[i]) * sign
        end
        return SMatrix(rows)
    end

    JA = make_J(rA, 1.0)
    JB = make_J(rB, -1.0)

    c1 = dot(n_curr, dp) - con.rest[1]
    c2 = dot(t1_curr, dp) - con.rest[2]
    c3 = dot(t2_curr, dp) - con.rest[3]

    C = @SVector [c1, c2, c3]

    # Check for tension or compression, return associated stiffness
    if c1 > 0
        if con.is_broken
            return C, JA, JB, (@SVector zeros(3))
        else
            k_limit = con.stiffness * (1.0 - con.damage)
            return C, JA, JB, k_limit
        end
        #k_eff = con.stiffness * (1.0 - con.damage) # TODO should I just say con.penalty_k??
    else
        return C, JA, JB, con.stiffness
    end

end

# Update bond state - only once at the end of step_simulation
function update_bond_state!(con::BondConstraint)

    if con.is_broken
        return
    end

    rA = rotate_vec(con.pA_local, con.bodyA.quat)
    rB = rotate_vec(con.pB_local, con.bodyB.quat)
    pA = con.bodyA.pos + rA
    pB = con.bodyB.pos + rB
    dp = pA - pB

    rotA = quat_to_rotmat(con.bodyA.quat)
    c_n = dot(rotA * con.n_local, dp) - con.rest[1]
    c_t1 = dot(rotA * con.t1_local, dp) - con.rest[2]
    c_t2 = dot(rotA * con.t2_local, dp) - con.rest[3]
    con.C = @SVector [c_n, c_t1, c_t2]

    d_n = max(c_n, 0.0)
    d_s = sqrt(c_t1^2 + c_t2^2)

    dn0, ds0, dnc, dsc = con.limits
    strain = sqrt((d_n / dnc)^2 + (d_s / dsc)^2)

    # 2. Update History
    con.current_eff_strain = strain
    con.max_eff_strain = max(con.max_eff_strain, strain)

    # 3. Cohesion & Damage Evolution
    if !con.is_cohesive
        Psi = (d_n / dn0)^2 + (d_s / ds0)^2
        if Psi >= 1.0
            con.is_cohesive = true
            con.max_committed_strain = max(con.max_committed_strain, strain)
        end
    end

    if con.is_cohesive
        con.max_committed_strain = max(con.max_committed_strain, strain)

        if con.max_committed_strain >= 1.0
            con.is_broken = true
            con.damage = 1.0
            con.k_tension = @SVector zeros(3)
        else
            # Linear softening (or replace with your specific curve)
            numer = (d_n / dnc)^2 + (d_s / dsc)^2
            denom = (d_n / dn0)^2 + (d_s / ds0)^2
            curr_lam_cr = (denom > 1e-12) ? sqrt(numer / denom) : (dn0 / dnc)

            if con.max_committed_strain > curr_lam_cr
                d = (con.max_committed_strain - curr_lam_cr) / (1.0 - curr_lam_cr)
                con.damage = clamp(d, 0.0, 1.0)
            end
        end
    end

end

##### ---------- End Bond Constraint ---------- #####

##### ---------- Helper functions ---------- #####

#TODO Gemini provided math - check for accuracy - renamed for now to be not used
function get_delta_twist(b::Body, from_pos::Vec3, from_quat::Quat)
    # 1. Translational Delta
    dx = b.pos - from_pos

    # 2. Rotational Delta (Exact Log Map)
    # Ensure shortest path (Hemisphere check)
    # Matches Python: qs = self._closest_hemisphere(q, qs) 
    if dot(b.quat, from_quat) < 0.0
        from_quat = -from_quat
    end

    # Difference quaternion: q_diff = q_current * inv(q_inertial)
    q_rel = quat_mul(b.quat, quat_inv(from_quat))

    # Map to rotation vector (axis * angle)
    # Matches Python: dth = quat_log(qerr) 
    d_th = quat_to_rotvec(q_rel)

    return vcat(dx, d_th)
end

end # module end
