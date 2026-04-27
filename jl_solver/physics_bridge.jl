"""
Julia bridge functions exposed to Python.

This module translates NumPy-compatible arrays into the solver state, advances
the AVBD simulation, and writes the binary/CSV exports consumed by the thesis
visualization and post-processing scripts.
"""
module PhysicsBridge

include("maths.jl")
include("collisions.jl")
include("avbd_constraints.jl")
include("manifold.jl")
include("energy.jl")
include("avbd_core.jl")

using .AVBDCore
using .Maths

const FRAME_MAGIC = "AVB2"
const BOND_META_MAGIC = "ABM2"

# --- Interface Functions ---

"""Initialize the Julia `SimulationState` from Python-owned arrays."""
function init_system(
    pos::Matrix{FLOAT},   # (N, 7)
    vel::Matrix{FLOAT},   # (N, 6)
    masses::Vector{FLOAT}, # (N,)
    bond_data::Matrix{FLOAT}, # (M, 17)
    dt::FLOAT,
    gravity::FLOAT,
    iters::Int;
    friction::FLOAT=0.5,
    beta::FLOAT=10.0,
    gamma::FLOAT=0.99,
    alpha::FLOAT=0.95,
    stabilize::Bool=true,
    sizes::Union{Matrix{FLOAT},Nothing}=nothing,
    assembly_ids::Union{Vector{Int},Nothing}=nothing,
    active=nothing,
    valid_mask=nothing,
    can_refine=nothing,
    level=nothing,
    parent_list=nothing,
    children_start=nothing,
    children_count=nothing,
    neighbor_map=nothing,
    max_ref_level=nothing,
    max_ref_level_per_body=nothing,
    criteria_refine_stress_threshold=nothing,
    criteria_refine_stress_exclude_kinematic::Bool=true,
)
    sim = AVBDCore.init_simulation(pos, vel, masses, bond_data, dt, gravity, iters;
        friction=friction, beta=beta, gamma=gamma, alpha=alpha, stabilize=stabilize,
        sizes=sizes, assembly_ids=assembly_ids,
        active=active, valid_mask=valid_mask, can_refine=can_refine, level=level, parent_list=parent_list, children_start=children_start, children_count=children_count, neighbor_map=neighbor_map,
        max_ref_level=max_ref_level, max_ref_level_per_body=max_ref_level_per_body,
        criteria_refine_stress_threshold=criteria_refine_stress_threshold,
        criteria_refine_stress_exclude_kinematic=criteria_refine_stress_exclude_kinematic)
    return sim
end

"""Advance the simulation for `steps` solver steps."""
function step_batch!(sim::AVBDCore.SimulationState, steps::Int)
    for _ in 1:steps
        AVBDCore.step_simulation!(sim)
    end
end

# Python cannot access `!`-suffixed names directly; provide a stable alias.
step_batch(sim::AVBDCore.SimulationState, steps::Int) = step_batch!(sim, steps)

function step_timed(sim::AVBDCore.SimulationState)
    return AVBDCore.step_simulation_timed!(sim)
end

"""Return positions and quaternions for all bodies."""
function get_positions(sim::AVBDCore.SimulationState)
    n = length(sim.bodies)
    data = zeros(FLOAT, n, 7)

    for i in 1:n
        b = sim.bodies[i]
        data[i, 1] = b.pos[1]
        data[i, 2] = b.pos[2]
        data[i, 3] = b.pos[3]
        data[i, 4] = b.quat[1]
        data[i, 5] = b.quat[2]
        data[i, 6] = b.quat[3]
        data[i, 7] = b.quat[4]
    end
    return data
end

"""Write one binary visualization frame with active bodies and active bonds."""
function write_frame(sim::AVBDCore.SimulationState, filename::String)
    active_body_indices = sim.active_body_ids
    stress_data, bond_data = get_visualization_data(sim)
    n_bodies = length(active_body_indices)
    n_bonds = size(bond_data, 1)

    open(filename, "w") do io

        write(io, codeunits(FRAME_MAGIC))
        write(io, Int32(n_bodies))
        write(io, Int32(n_bonds))

        # Time
        write(io, Float32(sim.dt))

        # Body positional data + IDs + per-body stress (72 bytes per body)
        for (local_idx, body_idx) in enumerate(active_body_indices)
            b = sim.bodies[body_idx]
            # Position
            write(io, Float32(b.pos[1]))
            write(io, Float32(b.pos[2]))
            write(io, Float32(b.pos[3]))

            # rotation
            write(io, Float32(b.quat[1]))
            write(io, Float32(b.quat[2]))
            write(io, Float32(b.quat[3]))
            write(io, Float32(b.quat[4]))

            # For adaptive meshing - if voxels change size
            write(io, Float32(b.size[1]))
            write(io, Float32(b.size[2]))
            write(io, Float32(b.size[3]))

            # Persistent body ID
            write(io, Int32(b.id))

            # ID
            write(io, Int32(b.assembly_id))

            # Symmetric stress tensor (xx, yy, zz, xy, yz, zx)
            write(io, Float32(stress_data[local_idx, 1]))
            write(io, Float32(stress_data[local_idx, 2]))
            write(io, Float32(stress_data[local_idx, 3]))
            write(io, Float32(stress_data[local_idx, 4]))
            write(io, Float32(stress_data[local_idx, 5]))
            write(io, Float32(stress_data[local_idx, 6]))
        end

        # Raw bond state for post-processing
        for i in 1:n_bonds
            write(io, Int32(bond_data[i, 1]))
            write(io, Int32(bond_data[i, 2]))
            write(io, Int32(bond_data[i, 3]))
            write(io, Float32(bond_data[i, 4]))
            write(io, Float32(bond_data[i, 5]))
            write(io, Float32(bond_data[i, 6]))
            write(io, Float32(bond_data[i, 7]))
            write(io, Float32(bond_data[i, 8]))
            write(io, Float32(bond_data[i, 9]))
            write(io, Float32(bond_data[i, 10]))
            write(io, Float32(bond_data[i, 11]))
            write(io, Float32(bond_data[i, 12]))
            write(io, Float32(bond_data[i, 13]))
            write(io, UInt8(bond_data[i, 14] != 0.0))
            write(io, UInt8(bond_data[i, 15] != 0.0))
            write(io, UInt16(0))
        end
    end
    return (n_bodies, n_bonds)
end

"""Write the most recent energy sample as a single-row CSV."""
function write_energy_csv(sim::AVBDCore.SimulationState, filename::String, frame_idx::Int)
    log = sim.energy_log

    if frame_idx == 0 && isempty(log.kinetic)
        for bond in sim.bond_constraints
            if !bond.rest_initialized
                AVBDConstraints.initialize!(bond)
            end
        end
        AVBDCore.log_step!(log, sim.bodies, sim.bond_constraints, sim.contact_constraints, sim.alpha,
            sim.active_body_ids, sim.active_bond_ids)
    end

    if isempty(log.kinetic)
        return
    end

    idx = length(log.kinetic)
    t = (idx - 1) * sim.dt

    open(filename, "w") do io
        write(io, "frame,time,kinetic,bond_potential,contact_potential,fracture_work,viscous_work,mech_energy,accounted_energy\n")
        write(io, string(frame_idx, ",", t, ",", log.kinetic[idx], ",", log.bond_potential[idx], ",", log.contact_potential[idx], ",", log.fracture_work[idx], ",", log.viscous_work[idx], ",", log.mech_energy[idx], ",", log.accounted_energy[idx], "\n"))
    end
end

"""Write static bond metadata used by downstream post-processing."""
function write_bond_metadata(sim::AVBDCore.SimulationState, filename::String)
    open(filename, "w") do io
        write(io, codeunits(BOND_META_MAGIC))
        write(io, Int32(length(sim.bond_constraints)))
        for bond in sim.bond_constraints
            write(io, Int32(bond.id))
            write(io, Int32(bond.bodyA.id))
            write(io, Int32(bond.bodyB.id))
            write(io, Float32(bond.area))
            write(io, Float32(bond.f_min[1]))
            write(io, Float32(bond.f_min[2]))
            write(io, Float32(bond.f_min[3]))
            write(io, Float32(bond.f_max[1]))
            write(io, Float32(bond.f_max[2]))
            write(io, Float32(bond.f_max[3]))
        end
    end
end

"""Return coarse per-step solver metrics for progress and logging."""
function get_last_step_metrics(sim::AVBDCore.SimulationState)
    return (
        sim.step_count,
        sim.last_iters_used,
        sim.last_max_violation,
        length(sim.active_body_ids),
        length(sim.active_bond_ids),
        sim.last_contact_count,
    )
end

"""Return stress tensors and bond metadata for Python-side visualization."""
function get_visualization_data(sim::AVBDCore.SimulationState)
    # Active bodies are sourced from sim.active_body_ids (precomputed in solver)
    active_body_indices = sim.active_body_ids

    # Reuse the same solver-side stress accumulation that drives
    # stress-based AMR so the exported field matches the criterion input.
    stress_data, _ = AVBDCore.Criteria.compute_body_stress_data(sim)

    # Active bonds for metadata: include broken bonds so exported damage can reach 1.0.
    active_bond_count = 0
    for bond in sim.bond_constraints
        bond.is_active || continue
        a_idx = bond.bodyA.id + 1
        b_idx = bond.bodyB.id + 1
        if sim.active[a_idx] && sim.active[b_idx]
            active_bond_count += 1
        end
    end

    bond_data = zeros(FLOAT, active_bond_count, 15)

    # Reuse an active-bond pass to emit metadata (including broken bonds).
    bond_out_idx = 0
    for bond in sim.bond_constraints
        bond.is_active || continue
        a_idx = bond.bodyA.id + 1
        b_idx = bond.bodyB.id + 1
        if !(sim.active[a_idx] && sim.active[b_idx])
            continue
        end
        if !bond.rest_initialized
            AVBDConstraints.initialize!(bond)
        end
        bond_out_idx += 1
        bA = bond.bodyA
        bB = bond.bodyB

        # Raw bond state
        bond_data[bond_out_idx, 1] = Float64(bond.id)
        bond_data[bond_out_idx, 2] = Float64(bA.id)
        bond_data[bond_out_idx, 3] = Float64(bB.id)
        bond_data[bond_out_idx, 4] = bond.C[1]
        bond_data[bond_out_idx, 5] = bond.C[2]
        bond_data[bond_out_idx, 6] = bond.C[3]
        bond_data[bond_out_idx, 7] = bond.rest[1]
        bond_data[bond_out_idx, 8] = bond.rest[2]
        bond_data[bond_out_idx, 9] = bond.rest[3]
        bond_data[bond_out_idx, 10] = bond.penalty_k[1]
        bond_data[bond_out_idx, 11] = bond.penalty_k[2]
        bond_data[bond_out_idx, 12] = bond.penalty_k[3]
        bond_data[bond_out_idx, 13] = clamp(bond.damage, 0.0, 1.0)
        bond_data[bond_out_idx, 14] = bond.is_broken ? 1.0 : 0.0
        bond_data[bond_out_idx, 15] = bond.is_cohesive ? 1.0 : 0.0
    end
    return stress_data, bond_data
end

end # module
