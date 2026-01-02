module PhysicsBridge

using LinearAlgebra
using StaticArrays

# Load the Core
include("physics_core.jl")
include("maths.jl")
using .PhysicsCore
using .Maths

# --- Interface Functions ---

function init_system(
    pos::Matrix{FLOAT},   # (N, 7)
    vel::Matrix{FLOAT},   # (N, 6)
    masses::Vector{FLOAT}, # (N,)
    bond_data::Matrix{FLOAT}, # (M, 16)
    dt::FLOAT,
    gravity::FLOAT,
    iters::Int;
    friction::FLOAT=0.5,
    sizes::Union{Matrix{FLOAT},Nothing}=nothing,
    assembly_ids::Union{Vector{Int},Nothing}=nothing
)
    # Create the SimulationState object
    # This object stays in Julia memory
    sim = PhysicsCore.init_simulation(pos, vel, masses, bond_data, dt, gravity, iters; friction=friction, sizes=sizes, assembly_ids=assembly_ids)
    return sim
end

function step_batch!(sim::PhysicsCore.SimulationState, steps::Int)
    for _ in 1:steps
        PhysicsCore.step_simulation!(sim)
    end
end

# Python cannot access `!`-suffixed names directly; provide a stable alias.
step_batch(sim::PhysicsCore.SimulationState, steps::Int) = step_batch!(sim, steps)

function get_positions(sim::PhysicsCore.SimulationState)
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

function write_frame(sim::PhysicsCore.SimulationState, filename::String)

    open(filename, "w") do io

        # Number of bodies and bonds
        n_bodies = length(sim.bodies)
        active_bonds = [b for b in sim.bond_buffer if !b.is_broken]
        n_bonds = length(active_bonds)

        write(io, Int32(n_bodies))
        write(io, Int32(n_bonds))

        # Time
        write(io, Float32(sim.dt))

        # Body positional data + per-body stress (68 bytes per body)
        stress_data, _ = get_visualization_data(sim)
        for (i, b) in enumerate(sim.bodies)
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
            write(io, Float32(stress_data[i, 1]))
            write(io, Float32(stress_data[i, 2]))
            write(io, Float32(stress_data[i, 3]))
            write(io, Float32(stress_data[i, 4]))
            write(io, Float32(stress_data[i, 5]))
            write(io, Float32(stress_data[i, 6]))
        end

        # Bond data (strain/stress)
        for b in active_bonds
            write(io, Int32(b.body_idx_a))
            write(io, Int32(b.body_idx_b))

            write(io, Float32(b.max_eff_strain))
            write(io, Float32(b.current_eff_strain))
        end
    end
end

function get_visualization_data(sim::PhysicsCore.SimulationState)
    n_bodies = length(sim.bodies)
    # Need to create stress tensor - symmetric
    stress_data = zeros(FLOAT, n_bodies, 6)

    active_bonds = [b for b in sim.bond_buffer if !b.is_broken]
    bond_data = zeros(FLOAT, length(active_bonds), 4)

    for (i, bond) in enumerate(active_bonds)
        bA = sim.bodies[bond.body_idx_a+1]
        bB = sim.bodies[bond.body_idx_b+1]

        # World geometry
        n, t1, t2 = PhysicsCore.Constraints.bond_basis_world(bond, bA)
        pA = transform_point(bond.pA_local, bA.pos, bA.quat)
        pB = transform_point(bond.pB_local, bB.pos, bB.quat)
        dp = pA - pB

        # Force calculation (F = -k*x)
        err = Vec3(dot(n, dp), dot(t1, dp), dot(t2, dp)) # basically x
        f_local = -bond.current_k .* err
        F_world = n * f_local[1] + t1 * f_local[2] + t2 * f_local[3]

        # stress calculation
        rA = pA - bA.pos
        volA = bA.size[1] * bA.size[2] * bA.size[3]
        _acc_stress!(stress_data, bond.body_idx_a + 1, rA, F_world, volA)

        rB = pB - bB.pos
        volB = bB.size[1] * bB.size[2] * bB.size[3]
        _acc_stress!(stress_data, bond.body_idx_b + 1, rB, -F_world, volB)

        # Bond metadata
        bond_data[i, 1] = Float64(bond.body_idx_a)
        bond_data[i, 2] = Float64(bond.body_idx_b)
        bond_data[i, 3] = bond.current_eff_strain
        bond_data[i, 4] = bond.max_eff_strain
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
