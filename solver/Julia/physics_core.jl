module PhysicsCore

using LinearAlgebra
using StaticArrays
using Base.Threads

include("maths.jl")
include("collisions.jl")
include("constraints.jl")
include("manifold.jl")

using .Maths
import .Collisions
import .Collisions: Body, Contact, BroadPhaseState, update_rotation!, update_inv_inertia_world!, build_manifold!, sat_check, broad_phase!
using .Constraints
using .ManifoldHandling

export SimulationState, init_simulation, step_simulation!

const CONTACT_PAD = 16

mutable struct SimulationState
    bodies::Vector{Body}
    manifolds::Dict{Tuple{Int,Int},Manifold}
    broadphase::BroadPhaseState
    contacts::Vector{Contact}
    contact_count::Int
    contact_buffer::Vector{ContactConstraint}
    active_contact_count::Int
    bond_buffer::Vector{FaceBond}

    dt::Float64
    gravity::Float64
    friction::Float64
    iterations::Int
end

function SimulationState(bodies, bonds, dt, grav, fric, iters)
    n = length(bodies)
    contact_cap = max(4 * n, CONTACT_PAD)
    contacts = Vector{Contact}(undef, contact_cap)
    constraint_buffer = Vector{ContactConstraint}(undef, contact_cap)
    for i in 1:contact_cap
        constraint_buffer[i] = ContactConstraint()
    end
    broad = BroadPhaseState(n)
    manifolds = Dict{Tuple{Int,Int},Manifold}()
    return SimulationState(bodies, manifolds, broad, contacts, 0, constraint_buffer, 0, bonds, dt, grav, fric, iters)
end

# --- Helpers ---

function ensure_contact_capacity!(sim::SimulationState, needed::Int)
    current_contacts = length(sim.contacts)
    current_constraints = length(sim.contact_buffer)
    target = max(needed, current_contacts, current_constraints)
    if target <= current_contacts && target <= current_constraints
        return
    end

    new_len = max(target + 8, current_contacts * 2, current_constraints * 2)
    resize!(sim.contacts, new_len)
    old_len = length(sim.contact_buffer)
    resize!(sim.contact_buffer, new_len)
    @inbounds for i in (old_len+1):new_len
        sim.contact_buffer[i] = ContactConstraint()
    end
end

@inline function get_manifold!(sim::SimulationState, a_id::Int, b_id::Int)
    key = a_id <= b_id ? (a_id, b_id) : (b_id, a_id)
    return get!(sim.manifolds, key) do
        ManifoldHandling.init_manifold(key[1], key[2])
    end
end

function predict_bodies!(sim::SimulationState, dt::Float64, gravity::Float64)
    g_vec = Vec3(0.0, gravity * dt, 0.0)

    @threads for i in eachindex(sim.bodies)
        b = sim.bodies[i]
        b.pos_prev = b.pos
        b.quat_prev = b.quat

        if b.is_static
            update_rotation!(b)
            update_inv_inertia_world!(b)
            continue
        end

        b.vel += g_vec
        b.pos += b.vel * dt

        w = b.ang_vel
        dq = Quat(0.0, w[1], w[2], w[3])
        b.quat = normalize(b.quat + quat_mul(dq, b.quat) * (0.5 * dt))

        update_rotation!(b)
        update_inv_inertia_world!(b)
    end
end

function detect_collisions!(sim::SimulationState, friction::Float64)
    sim.contact_count = 0
    sim.active_contact_count = 0

    pair_count = broad_phase!(sim.broadphase, sim.bodies)
    contact_count = 0
    active_count = 0

    for i in 1:pair_count
        idxA, idxB = sim.broadphase.potential_pairs[i]
        bA = sim.bodies[idxA]
        bB = sim.bodies[idxB]
        if bA.is_static && bB.is_static
            continue
        end

        axis, overlap = sat_check(bA, bB)
        if axis === nothing
            continue
        end

        start_idx = contact_count
        contact_count = build_manifold!(bA, bB, axis, overlap, sim.contacts, contact_count)
        added = contact_count - start_idx
        if added == 0
            continue
        end

        ensure_contact_capacity!(sim, contact_count)
        if active_count + added > length(sim.contact_buffer)
            ensure_contact_capacity!(sim, active_count + added)
        end

        manifold = get_manifold!(sim, bA.id, bB.id)
        active_count = ManifoldHandling.update_manifold!(manifold, sim.contacts, start_idx + 1, added,
            friction, sim.contact_buffer, active_count)
    end

    sim.contact_count = contact_count
    sim.active_contact_count = active_count
end

function prepare_contacts!(sim::SimulationState, dt::Float64)
    @inbounds for i in 1:sim.active_contact_count
        con = sim.contact_buffer[i]
        bA = sim.bodies[con.body_idx_a+1]
        bB = sim.bodies[con.body_idx_b+1]
        prepare_contact!(con, bA, bB, dt)
    end
end

function solve_contacts!(sim::SimulationState)
    @inbounds for i in 1:sim.active_contact_count
        con = sim.contact_buffer[i]
        bA = sim.bodies[con.body_idx_a+1]
        bB = sim.bodies[con.body_idx_b+1]
        solve_contact!(con, bA, bB)
    end
end

function solve_bonds!(sim::SimulationState, dt::Float64)
    @inbounds for bond in sim.bond_buffer
        bA = sim.bodies[bond.body_idx_a+1]
        bB = sim.bodies[bond.body_idx_b+1]
        solve_bond!(bond, bA, bB, dt)
    end
end

function finalize_integration!(sim::SimulationState, dt::Float64)
    @threads for i in eachindex(sim.bodies)
        b = sim.bodies[i]
        if b.is_static
            update_rotation!(b)
            update_inv_inertia_world!(b)
            continue
        end
        b.pos = b.pos_prev + b.vel * dt
        w = b.ang_vel
        dq = Quat(0.0, w[1], w[2], w[3])
        b.quat = normalize(b.quat_prev + quat_mul(dq, b.quat_prev) * (0.5 * dt))
        update_rotation!(b)
        update_inv_inertia_world!(b)
    end
end

# --- Initialization ---

function init_simulation(
    pos_flat::Matrix{Float64},
    vel_flat::Matrix{Float64},
    masses::Vector{Float64},
    bond_data::Matrix{Float64},
    dt::Float64, gravity::Float64, iterations::Int;
    friction::Float64=0.5,
    sizes::Union{Matrix{Float64},Nothing}=nothing,
    assembly_ids::Union{Vector{Int},Nothing}=nothing
)
    n_bodies = length(masses)
    bodies = Vector{Body}(undef, n_bodies)

    for i in 1:n_bodies
        p = Vec3(pos_flat[i, 1], pos_flat[i, 2], pos_flat[i, 3])
        q_raw = Quat(pos_flat[i, 4], pos_flat[i, 5], pos_flat[i, 6], pos_flat[i, 7])
        q = normalize(q_raw)
        size_v = sizes === nothing ? Vec3(1.0, 1.0, 1.0) : Vec3(sizes[i, 1], sizes[i, 2], sizes[i, 3])

        v_lin = Vec3(vel_flat[i, 1], vel_flat[i, 2], vel_flat[i, 3])
        v_ang = Vec3(vel_flat[i, 4], vel_flat[i, 5], vel_flat[i, 6])

        mass = masses[i]
        is_static = (mass == Inf) && (norm(v_lin) == 0.0) && (norm(v_ang) == 0.0)
        asm_id = assembly_ids === nothing ? -1 : assembly_ids[i]

        bodies[i] = Body(i - 1, asm_id, is_static, p, q, size_v, mass; vel=v_lin, ang_vel=v_ang)
    end

    bonds = Vector{FaceBond}()
    n_bonds = size(bond_data, 1)
    if n_bonds > 0
        sizehint!(bonds, n_bonds)
        for i in 1:n_bonds
            row = @view bond_data[i, :]
            idxA = Int(row[1])
            idxB = Int(row[2])
            pA = Vec3(row[3], row[4], row[5])
            pB = Vec3(row[6], row[7], row[8])
            n_w = Vec3(row[9], row[10], row[11])
            params = (row[12], row[13], row[14], row[15], row[16])
            b = Constraints.make_bond(idxA, idxB, pA, pB, n_w, params, bodies[idxA+1])
            push!(bonds, b)
        end
    end

    return SimulationState(bodies, bonds, dt, gravity, friction, iterations)
end

# --- Simulation Step ---

function step_simulation!(sim::SimulationState)
    predict_bodies!(sim, sim.dt, sim.gravity)

    detect_collisions!(sim, sim.friction)
    prepare_bonds!(sim.bond_buffer, sim.bodies, sim.dt)
    prepare_contacts!(sim, sim.dt)

    for _ in 1:sim.iterations
        solve_bonds!(sim, sim.dt)
        solve_contacts!(sim)
    end

    finalize_integration!(sim, sim.dt)
    commit_bond_damage!(sim.bond_buffer, sim.bodies)
end

end # module
