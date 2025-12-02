module PhysicsCore

using LinearAlgebra
using StaticArrays
using SparseArrays

# Include sibling modules
include("maths.jl")
include("collisions.jl")
include("constraints.jl")

using .Maths
import .Collisions: Body, update_rotation!, get_collisions
using .Constraints

export SimulationState, init_simulation, step_simulation!

# --- Simulation Container ---
mutable struct SimulationState
    bodies::Vector{Body}
    manifold::Manifold
    bonds::Vector{FaceBond}

    # Solver Tunables
    dt::Float64
    gravity::Float64
    iterations::Int

    # Temp Storage for Cholesky Solver
    # We use a dense matrix for small-medium systems, or sparse for larger ones.
    # For high performance with <1000 bodies, Dense is often faster due to SIMD.
    # We will use a flat array and reshape it to avoid allocations.
    H::Matrix{Float64}
    f::Vector{Float64}
    delta::Vector{Float64}

    function SimulationState(bodies, bonds, dt, grav, iters)
        n = length(bodies)
        # 6 DOFs per body
        dof = n * 6
        H = zeros(Float64, dof, dof)
        f = zeros(Float64, dof)
        delta = zeros(Float64, dof)

        new(bodies, Manifold(), bonds, dt, grav, iters, H, f, delta)
    end
end

# --- Initialization ---
function init_simulation(
    pos_flat::Matrix{Float64},
    vel_flat::Matrix{Float64},
    masses::Vector{Float64},
    # Bond data: [idxA, idxB, pA(3), pB(3), n(3), kn, kt, area, tensile, Gc]
    bond_data::Matrix{Float64},
    dt::Float64, gravity::Float64, iterations::Int;
    sizes::Union{Matrix{Float64},Nothing}=nothing,
    assembly_ids::Union{Vector{Int},Nothing}=nothing
)
    n_bodies = length(masses)
    bodies = Vector{Body}(undef, n_bodies)

    # 1. Build Bodies
    for i in 1:n_bodies
        # Unpack flat arrays
        p = Vec3(pos_flat[i, 1], pos_flat[i, 2], pos_flat[i, 3])
        q_raw = Quat(pos_flat[i, 4], pos_flat[i, 5], pos_flat[i, 6], pos_flat[i, 7])
        q = normalize(q_raw)
        size_v = sizes === nothing ? Vec3(1.0, 1.0, 1.0) : Vec3(sizes[i, 1], sizes[i, 2], sizes[i, 3])

        v_lin = Vec3(vel_flat[i, 1], vel_flat[i, 2], vel_flat[i, 3])
        v_ang = Vec3(vel_flat[i, 4], vel_flat[i, 5], vel_flat[i, 6])

        mass = masses[i]
        is_static = (mass == Inf)
        asm_id = assembly_ids === nothing ? -1 : assembly_ids[i]

        bodies[i] = Body(i - 1, asm_id, is_static, p, q, size_v, mass; vel=v_lin, ang_vel=v_ang)
    end

    # 2. Build Bonds
    bonds = Vector{FaceBond}()
    n_bonds = size(bond_data, 1)
    if n_bonds > 0
        for i in 1:n_bonds
            row = @view bond_data[i, :]
            idxA = Int(row[1])
            idxB = Int(row[2])
            pA = Vec3(row[3], row[4], row[5])
            pB = Vec3(row[6], row[7], row[8])
            n_w = Vec3(row[9], row[10], row[11])

            # Params: kn, kt, area, tensile, Gc
            # Note: make_bond expects params as a tuple or array
            params = (row[12], row[13], row[14], row[15], row[16])

            # Constraints.jl helper
            # Note: Julia is 1-based, Python ID is 0-based.
            b = Constraints.make_bond(idxA, idxB, pA, pB, n_w, params, bodies[idxA+1])
            push!(bonds, b)
        end
    end

    return SimulationState(bodies, bonds, dt, gravity, iterations)
end

# --- The Hot Loop Helpers ---

function integrate_inertial!(sim::SimulationState, dt::Float64)
    # y = x + v * dt + 0.5 * g * dt^2
    # But we want the solver to solve for delta from x_current.
    # Standard Position Based Dynamics / Implicit Euler setup:
    # Guess position y = x_n + dt * v_n + dt^2 * M^-1 * f_ext

    # Gravity applied along Y (matches Python solver convention).
    g_vec = Vec3(0.0, sim.gravity * dt * dt, 0.0)

    for b in sim.bodies
        if b.is_static
            b.pos_prev = b.pos
            b.quat_prev = b.quat
            b.pos_inertial = b.pos
            b.quat_inertial = b.quat
            continue
        end

        b.pos_prev = b.pos
        b.quat_prev = b.quat

        # 1. Linear Prediction
        # For now, just explicit Euler prediction for the guess
        # b.vel is not stored in Body struct in Collisions.jl?
        # We need to ensure Body has velocity or we track it.
        # Assuming we added vel to Body, or we infer it.
        # Let's assume for this snippet that `delta` calculation handles the step,
        # and `pos_inertial` is purely x + v*dt.

        # NOTE: You need to add `vel::Vec3` and `ang_vel::Vec3` to your Body struct in collisions.jl!
        # I will assume they exist.

        b.pos_inertial = b.pos + b.vel * dt + g_vec

        # 2. Angular Prediction (Gyro)
        # q_new = q + 0.5 * w * q * dt
        dq = Quat(0.0, b.ang_vel[1], b.ang_vel[2], b.ang_vel[3])
        q_update = quat_mul(dq, b.quat) * (0.5 * dt)
        b.quat_inertial = normalize(b.quat + q_update)
    end
end

function add_block_to_H!(H::Matrix{Float64}, idxA::Int, idxB::Int, data::Matrix{Float64}, bodies::Vector{Body})
    # Add 6x6 blocks to the global H matrix
    # idxA, idxB are 0-based body IDs. 
    # Global H indices: 1-based.

    staticA = bodies[idxA+1].is_static
    staticB = bodies[idxB+1].is_static

    rA = idxA * 6 + 1
    rB = idxB * 6 + 1

    # H_AA (Only if A is dynamic)
    if !staticA
        for c in 1:6, r in 1:6
            H[rA+r-1, rA+c-1] += data[r, c]
        end
    end

    # H_BB (Only if B is dynamic)
    if !staticB
        for c in 1:6, r in 1:6
            H[rB+r-1, rB+c-1] += data[r+6, c+6]
        end
    end

    # H_AB and H_BA (Only if BOTH are dynamic)
    # If one is static, the delta is 0, so the cross term vanishes from the equation of motion
    # of the dynamic body.
    if !staticA && !staticB
        for c in 1:6, r in 1:6
            val = data[r, c+6]
            H[rA+r-1, rB+c-1] += val
            H[rB+c-1, rA+r-1] += val # Symmetry
        end
    end
end

function add_force!(f::Vector{Float64}, idxA::Int, idxB::Int, f_local::Vector{Float64}, bodies::Vector{Body})
    staticA = bodies[idxA+1].is_static
    staticB = bodies[idxB+1].is_static

    rA = idxA * 6 + 1
    rB = idxB * 6 + 1

    if !staticA
        for i in 1:6
            f[rA+i-1] += f_local[i]
        end
    end

    if !staticB
        for i in 1:6
            f[rB+i-1] += f_local[i+6]
        end
    end
end

function build_system!(sim::SimulationState, alpha::Float64)
    fill!(sim.H, 0.0)
    fill!(sim.f, 0.0)

    dt = sim.dt
    inv_dt2 = 1.0 / (dt^2)

    # 1. Mass Matrix (Diagonal)
    for i in 1:length(sim.bodies)
        b = sim.bodies[i]
        if b.is_static
            continue
        end

        idx = (i - 1) * 6 + 1

        # Linear Mass (M/dt^2)
        m_val = b.mass * inv_dt2
        sim.H[idx, idx] += m_val
        sim.H[idx+1, idx+1] += m_val
        sim.H[idx+2, idx+2] += m_val

        # Angular Inertia (I_world / dt^2)
        # Note: Correct way is J^T * M * J, but for small steps diagonal approx is often used.
        # Let's use the full I_world.
        # I_world = R * I_body * R^T
        I_body = b.inertia # SMatrix
        I_world = b.rot_mat * I_body * transpose(b.rot_mat)
        I_val = I_world * inv_dt2

        for r in 1:3, c in 1:3
            sim.H[idx+3+r-1, idx+3+c-1] += I_val[r, c]
        end

        # Force: M/dt^2 * (p_inertial - p_curr)
        # This pulls the body toward the inertial guess
        diff_lin = b.pos_inertial - b.pos

        # Calculate rotation from current to inertial: q_diff * q_curr = q_inertial
        # q_diff = q_inertial * q_curr^-1
        q_diff = quat_mul(b.quat_inertial, quat_inv(b.quat))
        diff_ang = quat_to_rotvec(q_diff)

        # Linear Forces
        sim.f[idx] += m_val * diff_lin[1]
        sim.f[idx+1] += m_val * diff_lin[2]
        sim.f[idx+2] += m_val * diff_lin[3]

        # Angular Forces (I_val * diff_ang)
        # We manually multiply the 3x3 block I_val by the 3-vector diff_ang
        f_ang = I_val * diff_ang
        sim.f[idx+3] += f_ang[1]
        sim.f[idx+4] += f_ang[2]
        sim.f[idx+5] += f_ang[3]
    end

    # 2. Bonds
    Constraints.prepare_bonds!(sim.bonds, sim.bodies)
    for bond in sim.bonds
        if bond.is_broken
            continue
        end
        H_el, f_el = Constraints.get_bond_system(bond, sim.bodies)
        # Pass bodies to check static flags
        add_block_to_H!(sim.H, bond.body_idx_a, bond.body_idx_b, H_el, sim.bodies)
        add_force!(sim.f, bond.body_idx_a, bond.body_idx_b, f_el, sim.bodies)
    end

    # 3. Contacts
    Constraints.prepare_contacts!(sim.manifold, sim.bodies, dt)
    for con in sim.manifold.contacts
        H_el, f_el = Constraints.get_contact_system(con, sim.bodies)
        add_block_to_H!(sim.H, con.body_idx_a, con.body_idx_b, H_el, sim.bodies)
        add_force!(sim.f, con.body_idx_a, con.body_idx_b, f_el, sim.bodies)
    end
end

function solve_and_update!(sim::SimulationState)
    # Solve H * delta = f
    # For stability add epsilon to diagonal
    n = size(sim.H, 1)
    for i in 1:n
        sim.H[i, i] += 1e-8
    end

    # Cholesky
    # In Julia, \ operator automatically picks the best solver.
    # For Positive Definite matrices, it uses Cholesky.
    try
        sim.delta = sim.H \ sim.f
    catch
        # Fallback for singular matrix
        sim.delta = (sim.H + I * 1e-3) \ sim.f
    end

    # Update Bodies
    for i in 1:length(sim.bodies)
        b = sim.bodies[i]
        if b.is_static
            continue
        end

        idx = (i - 1) * 6 + 1
        dx = Vec3(sim.delta[idx], sim.delta[idx+1], sim.delta[idx+2])
        dtheta = Vec3(sim.delta[idx+3], sim.delta[idx+4], sim.delta[idx+5])

        b.pos += dx

        # Update Quat: q_new = q + 0.5 * w * q
        dq = rotvec_to_quat(dtheta)
        b.quat = normalize(quat_mul(dq, b.quat))

        # Update cached rotation matrix
        update_rotation!(b)
    end
end

function update_velocity!(sim::SimulationState)
    inv_dt = 1.0 / sim.dt
    for b in sim.bodies
        if b.is_static
            continue
        end
        # Simple finite difference for linear velocity
        b.vel = (b.pos - b.pos_prev) * inv_dt
        # Angular velocity from quaternion delta
        dq = quat_mul(b.quat, quat_inv(b.quat_prev))
        rotvec = quat_to_rotvec(dq) * inv_dt
        b.ang_vel = rotvec
    end
end

function step_simulation!(sim::SimulationState)
    # 1. Prediction
    integrate_inertial!(sim, sim.dt)

    # 2. Broad/Narrow Phase
    new_contacts = Collisions.get_collisions(sim.bodies)
    Constraints.update_manifold!(sim.manifold, new_contacts, 0.5) # Friction 0.5

    # 3. Solve Loop
    for i in 1:sim.iterations
        # Alpha can vary (e.g. for spectral methods), 1.0 for standard Newton
        build_system!(sim, 1.0)
        solve_and_update!(sim)
    end

    # 4. Finalize
    update_velocity!(sim)
    Constraints.commit_bond_damage!(sim.bonds)
end

end # module
