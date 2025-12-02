module PhysicsBridge

using LinearAlgebra
using StaticArrays

# Load the Core
include("physics_core.jl")
using .PhysicsCore

# --- Interface Functions ---

function init_system(
    pos::Matrix{Float64},   # (N, 7)
    vel::Matrix{Float64},   # (N, 6)
    masses::Vector{Float64}, # (N,)
    bond_data::Matrix{Float64}, # (M, 16)
    dt::Float64,
    gravity::Float64,
    iters::Int;
    sizes::Union{Matrix{Float64},Nothing}=nothing,
    assembly_ids::Union{Vector{Int},Nothing}=nothing
)
    # Create the SimulationState object
    # This object stays in Julia memory
    sim = PhysicsCore.init_simulation(pos, vel, masses, bond_data, dt, gravity, iters; sizes=sizes, assembly_ids=assembly_ids)
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
    data = zeros(Float64, n, 7)

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

end # module
