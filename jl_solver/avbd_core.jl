module AVBDCore

# TODO CHECK avbd_core.jl, avbd_constraints.jl, manifold.jl and physics_bridge.jl - one of these screws with the voxel size??? Still error voxelizing to resolution!!!

using LinearAlgebra
using StaticArrays
using Base.Threads
using ..Maths: Vec3, Quat, integrate_quat, quat_to_rotvec, quat_mul, quat_inv, rotvec_to_quat, delta_twist_from
import ..Collisions: Body, Contact, get_collisions, update_rotation!, update_inv_inertia_world!
using ..AVBDConstraints
using ..ManifoldHandling: Manifold, init_manifold, update_manifold_dynamic!

export SimulationState, init_simulation, step_simulation!

mutable struct SimulationState
    bodies::Vector{Body}
    manifolds::Dict{Tuple{Int,Int},Manifold}

    persistent_constraints::Vector{BondConstraint}
    active_constraints::Vector{AbstractConstraint}

    # Body ID -> list of constraints affecting body
    incidence_map::Vector{Vector{AbstractConstraint}} # Incident map of constraints on bodies - needed? - better way to do this?

    # Solver settings
    dt::Float64
    gravity::Float64
    friction::Float64
    iterations::Int

    # AVBD loop params
    beta::Float64
    gamma::Float64
    alpha::Float64
    stabilize::Bool

    #TODO where do these values get set? In HybridSolver? I am setting them as default values 
    function SimulationState(bodies, dt; grav=-9.81, iters=10, mu=0.6, b=10.0, g=0.99, al=0.95, stabil=true)
        per_cons = Vector{BondConstraint}()
        act_cons = Vector{AbstractConstraint}()
        inc_map = Vector{Vector{AbstractConstraint}}()
        manifolds = Dict{Tuple{Int,Int},Manifold}()

        new(bodies, manifolds, per_cons, act_cons, inc_map, dt, grav, mu, iters, Float64(b), Float64(g), Float64(al), stabil)

    end
end

function predict_inertia!(sim::SimulationState) # Why do I pass dt and gravity if they are in simulation state struct??
    dt = sim.dt
    gravity_vec = Vec3(0.0, sim.gravity, 0.0)

    for b in sim.bodies
        # Initial position
        b.pos0 = b.pos
        b.quat0 = b.quat

        if b.is_static
            b.pos_inertia = b.pos
            b.quat_inertia = b.quat
            update_rotation!(b)
            update_inv_inertia_world!(b)
            continue
        end

        # Apply gravity
        b.vel += gravity_vec * dt

        # Inertial guess
        b.pos_inertia = b.pos + b.vel * dt
        b.quat_inertia = integrate_quat(b.quat, b.ang_vel, dt)

        b.pos = b.pos_inertia
        b.quat = b.quat_inertia

        update_rotation!(b)
        update_inv_inertia_world!(b)

    end
end

function detect_collisions!(sim::SimulationState)
    empty!(sim.active_constraints)
    append!(sim.active_constraints, sim.persistent_constraints)

    raw_contacts = get_collisions(sim.bodies)

    pair_groups = Dict{Tuple{Int,Int},Vector{Contact}}()

    for c in raw_contacts
        idA, idB = c.body_idx_a, c.body_idx_b
        if idA > idB
            idA, idB = idB, idA
        end

        key = (idA, idB)
        if !haskey(pair_groups, key)
            pair_groups[key] = Contact[]
        end
        push!(pair_groups[key], c)
    end

    for (key, contacts) in pair_groups
        manifold = get!(sim.manifolds, key) do
            init_manifold(key[1], key[2])
        end
        update_manifold_dynamic!(manifold, contacts, sim.friction, sim.bodies)
        append!(sim.active_constraints, manifold.constraints[1:manifold.count])
    end
end

function warm_start!(sim::SimulationState)
    # Clear incidence map
    num_bodies = length(sim.bodies)
    if length(sim.incidence_map) < num_bodies
        resize!(sim.incidence_map, num_bodies)
    end

    for i in 1:num_bodies
        sim.incidence_map[i] = AbstractConstraint[]
    end

    # Loop over all constraints
    for con in sim.active_constraints
        initialize!(con)

        con.penalty_k = clamp.(con.penalty_k .* sim.gamma, con.k_min, con.k_max)
        if !sim.stabilize
            con.lambda = con.lambda .* (sim.alpha * sim.gamma)
        end
        pk = con.penalty_k
        for r in 1:3
            if !isinf(con.stiffness[r])
                pk = setindex(pk, min(pk[r], con.stiffness[r]), r)
            end
        end
        con.penalty_k = pk

        # Build incidence_map
        push!(sim.incidence_map[con.bodyA.id+1], con)
        push!(sim.incidence_map[con.bodyB.id+1], con)
    end
end

function primal_solve!(b::Body, constraints::Vector{AbstractConstraint}, invdt2::Float64, alpha::Float64)
    if !isfinite(b.mass) || b.mass <= 0.0 || !all(isfinite, b.inertia_diag)
        return
    end
    m_val = b.mass * invdt2
    I_val = b.inertia_diag * invdt2

    H = MMatrix{6,6,Float64}(undef)
    fill!(H, 0.0)
    H[1, 1] = H[2, 2] = H[3, 3] = m_val
    H[4, 4] = I_val[1]
    H[5, 5] = I_val[2]
    H[6, 6] = I_val[3]

    d_twist = delta_twist_from(b, b.pos_inertia, b.quat_inertia)
    F = H * (-d_twist)

    for con in constraints
        compute_constraint!(con, alpha)
        if isa(con, ContactConstraint)
            AVBDConstraints.update_bounds!(con)
        end

        isA = (con.bodyA == b)
        J_block = isA ? con.JA : con.JB

        for r in 1:3
            k_val = con.penalty_k[r]
            k_val <= 0.0 && continue

            J_row = J_block[r, :]

            if isinf(con.stiffness[r])
                f_mag = k_val * con.C[r] + con.lambda[r]
                f_mag = clamp(f_mag, con.f_min[r], con.f_max[r])
            else
                f_mag = k_val * con.C[r]
            end

            H += (J_row * J_row') * k_val
            F -= J_row * f_mag
        end
    end

    for i in 1:6
        H[i, i] += 1e-6
    end

    dx = H \ F
    b.pos += dx[1:3]

    dq = rotvec_to_quat(Vec3(dx[4], dx[5], dx[6]))
    b.quat = quat_mul(dq, b.quat)
    b.quat = normalize(b.quat)
end

function dual_update!(sim::SimulationState, alpha::Float64)

    #Threads.@threads 
    for con in sim.active_constraints
        compute_constraint!(con, alpha)

        pk = con.penalty_k
        lam = con.lambda
        for r in 1:3
            if isinf(con.stiffness[r])
                sigma = lam[r] + pk[r] * con.C[r]
                lam = setindex(lam, clamp(sigma, con.f_min[r], con.f_max[r]), r)

                if lam[r] > con.f_min[r] && lam[r] < con.f_max[r]
                    new_k = pk[r] + sim.beta * abs(con.C[r])
                    new_k = min(new_k, con.k_max[r])
                    pk = setindex(pk, new_k, r)
                end
            else
                new_k = pk[r] + sim.beta * abs(con.C[r])
                new_k = min(new_k, min(con.k_max[r], con.stiffness[r]))
                pk = setindex(pk, new_k, r)
            end
        end
        con.penalty_k = pk
        con.lambda = lam

        if isa(con, ContactConstraint)
            AVBDConstraints.update_bounds!(con)
        end
    end
end

function update_vel!(sim::SimulationState, invdt::Float64)
    for b in sim.bodies
        b.is_static && continue

        b.vel = (b.pos - b.pos0) * invdt

        q_rel = quat_mul(b.quat, quat_inv(b.quat0))
        d_theta = quat_to_rotvec(q_rel)
        b.ang_vel = d_theta * invdt

        update_rotation!(b)
        update_inv_inertia_world!(b)
    end
end

function init_simulation(
    pos_flat::Matrix{Float64},
    vel_flat::Matrix{Float64},
    masses::Vector{Float64},
    bond_data::Matrix{Float64},
    dt::Float64, gravity::Float64, iterations::Int;
    friction::Float64=0.5,
    #sizes::Union{Matrix{Float64},Nothing}=nothing,
    sizes::Union{AbstractMatrix,Nothing}=nothing,
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

    sim = SimulationState(bodies, dt; grav=gravity, iters=iterations, mu=friction)

    n_bonds = size(bond_data, 1)
    if n_bonds > 0
        sizehint!(sim.persistent_constraints, n_bonds)
        for i in 1:n_bonds
            row = @view bond_data[i, :]
            idxA = Int(row[1])
            idxB = Int(row[2])
            pA = Vec3(row[3], row[4], row[5])
            pB = Vec3(row[6], row[7], row[8])
            n_w = Vec3(row[9], row[10], row[11])
            kn = row[12]
            kt = row[13]
            area = row[14]
            tensile = row[15]
            Gc = row[16]
            bA = bodies[idxA+1]
            bB = bodies[idxB+1]
            bond = BondConstraint(bA, bB, pA, pB, n_w, kn, kt, area, tensile, Gc)
            push!(sim.persistent_constraints, bond)
        end
    end

    return sim
end

function damage_bonds!(sim::SimulationState)
    idx_to_remove = Int[]

    for (i, bond) in enumerate(sim.persistent_constraints)
        commit_bond_damage!(bond)
        if bond.is_broken
            push!(idx_to_remove, i)
        end
    end
    deleteat!(sim.persistent_constraints, idx_to_remove)
end

@inline function assert_finite_body!(b, tag)
    if !(all(isfinite, b.pos) && all(isfinite, b.vel) && all(isfinite, b.quat) && all(isfinite, b.ang_vel))
        error("Non-finite body $(b.id) at $tag: pos=$(b.pos) quat=$(b.quat)")
    end
end


function step_simulation!(sim::SimulationState)

    dt = sim.dt
    inv_dt = 1 / dt
    inv_dt2 = 1 / (dt * dt)

    # Inertial solve
    predict_inertia!(sim)
    for b in sim.bodies
        assert_finite_body!(b, "after predict_inertia")
    end

    # TODO CHECK time diffrence between running with collision detection and contact constraints vs. just bonds
    # Build contact constraints
    detect_collisions!(sim)

    # warm start constraints
    warm_start!(sim) # Python builds incident map here - inside the constraint loop 
    for b in sim.bodies
        assert_finite_body!(b, "after warm_start")
    end

    total_iters = sim.iterations + (sim.stabilize ? 1 : 0)

    for it = 1:total_iters
        alpha_eff = sim.stabilize ? (it <= sim.iterations ? 1.0 : 0.0) : sim.alpha
        # Loop over bodies aka primal loop
        #Threads.@threads 
        for body in sim.bodies
            # solve constriants not in loop but via linear algebra
            body.is_static && continue

            primal_solve!(body, sim.incidence_map[body.id+1], inv_dt2, alpha_eff)
            assert_finite_body!(body, "after primal_solve it=$it")
        end
        # dual update
        if it <= sim.iterations
            dual_update!(sim, alpha_eff)
            for b in sim.bodies
                assert_finite_body!(b, "after dual_update it=$it")
            end
        end
    end

    # Velocity derivation from new positions
    update_vel!(sim, inv_dt)

    # Update & break some bonds
    damage_bonds!(sim)
end

end # module end
