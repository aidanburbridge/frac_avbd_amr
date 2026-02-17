module PhysicsBridge

using LinearAlgebra
using StaticArrays

# Load the Core
include("maths.jl")
include("collisions.jl")
include("avbd_constraints.jl")
include("manifold.jl")
include("energy.jl")
include("avbd_core.jl")

using .AVBDCore
using .Maths

# --- Interface Functions ---

function init_system(
    pos::Matrix{FLOAT},   # (N, 7)
    vel::Matrix{FLOAT},   # (N, 6)
    masses::Vector{FLOAT}, # (N,)
    bond_data::Matrix{FLOAT}, # (M, 17)
    dt::FLOAT,
    gravity::FLOAT,
    iters::Int;
    friction::FLOAT=0.5,
    sizes::Union{Matrix{FLOAT},Nothing}=nothing,
    assembly_ids::Union{Vector{Int},Nothing}=nothing,
    active=nothing, # TODO do I really need to pass these, could initialize active in julia solver based on level
    valid_mask=nothing,
    can_refine=nothing,
    level=nothing,
    parent_list=nothing,
    children_start=nothing,
    children_count=nothing,
    neighbor_map=nothing,
    max_ref_level=nothing,
)
    # Create the SimulationState object
    # This object stays in Julia memory
    sim = AVBDCore.init_simulation(pos, vel, masses, bond_data, dt, gravity, iters;
        friction=friction, sizes=sizes, assembly_ids=assembly_ids,
        active=active, valid_mask=valid_mask, can_refine=can_refine, level=level, parent_list=parent_list, children_start=children_start, children_count=children_count, neighbor_map=neighbor_map,
        max_ref_level=max_ref_level)
    return sim
end

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

function get_positions(sim::AVBDCore.SimulationState)
    # Extract positions to return to Python for rendering
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

function write_frame(sim::AVBDCore.SimulationState, filename::String)

    open(filename, "w") do io

        # Active bodies are sourced from sim.active_body_ids (precomputed in solver)
        active_body_indices = sim.active_body_ids

        # Visualization data (active-only, indices already remapped)
        stress_data, bond_data = get_visualization_data(sim)

        # Number of bodies and bonds (active only)
        n_bodies = length(active_body_indices)
        n_bonds = size(bond_data, 1)

        write(io, Int32(n_bodies))
        write(io, Int32(n_bonds))

        # Time
        write(io, Float32(sim.dt))

        # Body positional data + per-body stress (68 bytes per body)
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

        # Bond data (strain/damage/stiffness)
        for i in 1:n_bonds
            write(io, Int32(bond_data[i, 1]))
            write(io, Int32(bond_data[i, 2]))
            write(io, Float32(bond_data[i, 4]))
            write(io, Float32(bond_data[i, 3]))
            write(io, Float32(bond_data[i, 5]))
            write(io, Float32(bond_data[i, 6]))
            write(io, Float32(bond_data[i, 7]))
            write(io, Float32(bond_data[i, 8]))
        end
    end
end

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
        write(io, "frame,time,kinetic,bond_potential,contact_potential,fracture_work,mech_energy,accounted_energy\n")
        write(io, string(frame_idx, ",", t, ",", log.kinetic[idx], ",", log.bond_potential[idx], ",", log.contact_potential[idx], ",", log.fracture_work[idx], ",", log.mech_energy[idx], ",", log.accounted_energy[idx], "\n"))
    end
end

function get_visualization_data(sim::AVBDCore.SimulationState)
    # Active bodies are sourced from sim.active_body_ids (precomputed in solver)
    active_body_indices = sim.active_body_ids

    # Map global body id -> local active index (1-based) for stress and bond remapping
    id_to_local = fill(0, length(sim.bodies))
    for (local_idx, body_idx) in enumerate(active_body_indices)
        body_id = sim.bodies[body_idx].id
        id_to_local[body_id + 1] = local_idx
    end

    # Need to create stress tensor - symmetric
    stress_data = zeros(FLOAT, length(active_body_indices), 6)

    # Active bonds: unbroken, active, and both endpoints active
    active_bond_count = 0
    for bond in sim.bond_constraints
        bond.is_broken && continue
        bond.is_active || continue
        a_idx = bond.bodyA.id + 1
        b_idx = bond.bodyB.id + 1
        if sim.active[a_idx] && sim.active[b_idx]
            active_bond_count += 1
        end
    end

    bond_data = zeros(FLOAT, active_bond_count, 8)

    bond_out_idx = 0
    for bond in sim.bond_constraints
        bond.is_broken && continue
        bond.is_active || continue
        a_idx = bond.bodyA.id + 1
        b_idx = bond.bodyB.id + 1
        if !(sim.active[a_idx] && sim.active[b_idx])
            continue
        end
        bond_out_idx += 1
        bA = bond.bodyA
        bB = bond.bodyB

        # World geometry
        R_A = quat_to_rotmat(bA.quat)
        n = R_A * bond.n_local
        t1 = R_A * bond.t1_local
        t2 = R_A * bond.t2_local
        pA = transform_point(bond.pA_local, bA.pos, bA.quat)
        pB = transform_point(bond.pB_local, bB.pos, bB.quat)

        #dp = pA - pB

        #AVBDConstraints.eval_bond(bond)

        # Force calculation uses the same estimate as the solver (AL-style).
        f_local = MVector{3,Float64}(undef)
        for r in 1:3
            k_val = bond.penalty_k[r]
            lambda_base = isinf(bond.stiffness[r]) ? bond.lambda[r] : 0.0
            f_local[r] = clamp(k_val * bond.C[r] + lambda_base, bond.f_min[r], bond.f_max[r])
        end
        F_world = n * f_local[1] + t1 * f_local[2] + t2 * f_local[3]

        # stress calculation
        rA = pA - bA.pos
        volA = bA.size[1] * bA.size[2] * bA.size[3]
        local_a = id_to_local[bA.id + 1]
        _acc_stress!(stress_data, local_a, rA, F_world, volA)

        rB = pB - bB.pos
        volB = bB.size[1] * bB.size[2] * bB.size[3]
        local_b = id_to_local[bB.id + 1]
        _acc_stress!(stress_data, local_b, rB, -F_world, volB)

    end

    # Reuse the same active-bond pass to emit metadata
    bond_out_idx = 0
    for bond in sim.bond_constraints
        bond.is_broken && continue
        bond.is_active || continue
        a_idx = bond.bodyA.id + 1
        b_idx = bond.bodyB.id + 1
        if !(sim.active[a_idx] && sim.active[b_idx])
            continue
        end
        bond_out_idx += 1
        bA = bond.bodyA
        bB = bond.bodyB

        rest_len = max(abs(bond.rest[1]), 1e-12)
        strain_n = bond.C[1] / rest_len
        strain_t1 = bond.C[2] / rest_len
        strain_t2 = bond.C[3] / rest_len
        eff_strain = sqrt(strain_n^2 + strain_t1^2 + strain_t2^2)

        damage_val = bond.is_broken ? 1.0 : clamp(bond.damage, 0.0, 1.0)
        k_eff_n = bond.is_broken ? 0.0 : bond.k_eff[1]
        k_eff_t1 = bond.is_broken ? 0.0 : bond.k_eff[2]
        k_eff_t2 = bond.is_broken ? 0.0 : bond.k_eff[3]

        # Bond metadata
        bond_data[bond_out_idx, 1] = Float64(id_to_local[bA.id + 1] - 1)
        bond_data[bond_out_idx, 2] = Float64(id_to_local[bB.id + 1] - 1)
        bond_data[bond_out_idx, 3] = eff_strain
        bond_data[bond_out_idx, 4] = bond.max_eff_strain
        bond_data[bond_out_idx, 5] = damage_val
        bond_data[bond_out_idx, 6] = k_eff_n
        bond_data[bond_out_idx, 7] = k_eff_t1
        bond_data[bond_out_idx, 8] = k_eff_t2
    end
    return stress_data, bond_data
end

function _acc_stress!(buf, idx, r, f, vol)
    iv = 1.0 / max(vol, 1e-9)

    # volumetric 
    buf[idx, 1] += r[1] * f[1] * iv
    buf[idx, 2] += r[2] * f[2] * iv
    buf[idx, 3] += r[3] * f[3] * iv

    # deviatoric
    buf[idx, 4] += r[1] * f[2] * iv
    buf[idx, 5] += r[2] * f[3] * iv
    buf[idx, 6] += r[3] * f[1] * iv
end

end # module
