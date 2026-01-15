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

# TODO things to implement
# - have damage only apply to tension -> leave the stiffness of compresison untouched
# - ensure both compressive and tensile strain and stress are exported to the visualizer
# - need to somehow stabilize the fracture through 
# - early out for fast convergences

const EARLY_OUT_TOL = 1e-5 # TODO have this as a parameter as a fraction of h (voxel size) ~0.1% 
# Change iters to these values - lock them in as same 
# const INNER_ITERS = 5
# const MAX_DAMAGE_PASSES = 5

# TODO add damping value calculation - or have NN do this??
# damp_val = 2 * [0.05 - 0.3] * sqrt(k*mass)


mutable struct SimulationState
    bodies::Vector{Body}
    manifolds::Dict{Tuple{Int,Int},Manifold}

    bond_constraints::Vector{BondConstraint}
    contact_constraints::Vector{ContactConstraint}

    bond_incidence::Vector{Vector{BondConstraint}}
    contact_incidence::Vector{Vector{ContactConstraint}}

    # Body ID -> list of constraints affecting body
    #incidence_map::Vector{Vector{AbstractConstraint}} # Incident map of constraints on bodies - needed? - better way to do this?

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
        # Initialize constraint vectors
        bond_cons = Vector{BondConstraint}()
        contact_cons = Vector{ContactConstraint}()

        # Initialize constraint maps
        bond_map = Vector{Vector{BondConstraint}}()
        contact_map = Vector{Vector{ContactConstraint}}()

        manifolds = Dict{Tuple{Int,Int},Manifold}()

        new(bodies, manifolds, bond_cons, contact_cons, bond_map, contact_map,
            dt, grav, mu, iters, Float64(b), Float64(g), Float64(al), stabil)

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

        if b.inv_mass == 0.0
            b.pos_inertia = b.pos + b.vel * dt
            b.quat_inertia = integrate_quat(b.quat, b.ang_vel, dt)
            b.pos = b.pos_inertia
            b.quat = b.quat_inertia
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

    # Clear the contacts
    empty!(sim.contact_constraints)

    #append!(sim.active_constraints, sim.persistent_constraints)

    raw_contacts = get_collisions(sim.bodies)
    raw_count = length(raw_contacts)

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
        append!(sim.contact_constraints, manifold.constraints[1:manifold.count])
    end
    return raw_count
end

function warm_start!(sim::SimulationState)
    # Clear incidence map
    num_bodies = length(sim.bodies)

    current_len = length(sim.bond_incidence)
    if current_len < num_bodies
        resize!(sim.bond_incidence, num_bodies)
        resize!(sim.contact_incidence, num_bodies)
        # Initialize new
        for i in 1:num_bodies # TODO WHAT is the logic behind this? why not current_len?
            #for i in (current_len+1):num_bodies
            if !isassigned(sim.bond_incidence, i) # If not assigned, assign 
                sim.bond_incidence[i] = BondConstraint[]
                sim.contact_incidence[i] = ContactConstraint[]
            end
        end
    end

    for i in 1:num_bodies
        empty!(sim.bond_incidence[i])
        empty!(sim.contact_incidence[i])
    end

    # Loop bonds first
    for con in sim.bond_constraints

        # Skip broken bonds
        if con.is_broken
            continue
        end

        # TODO split warm start between compression and tension
        initialize!(con)

        # Decay both tension and compression
        con.k_tension = clamp.(con.k_tension .* sim.gamma, con.k_min, con.k_max)
        con.k_compression = clamp.(con.k_compression .* sim.gamma, con.k_min, con.k_max)

        # Build incidence_map
        push!(sim.bond_incidence[con.bodyA.id+1], con)
        push!(sim.bond_incidence[con.bodyB.id+1], con)

    end

    # Loop contacts
    for con in sim.contact_constraints

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
        push!(sim.contact_incidence[con.bodyA.id+1], con)
        push!(sim.contact_incidence[con.bodyB.id+1], con)
    end
end

function primal_solve!(b::Body, bonds::Vector{BondConstraint}, contacts::Vector{ContactConstraint}, dt::Float64, invdt2::Float64, alpha::Float64)
    if b.inv_mass == 0.0
        return
    end

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

    for con in bonds

        C_loc, JA, JB, k_limit = eval_bond(con, dt)
        # TODO Do I want damping only in tension or also in compression???

        isA = (con.bodyA.id == b.id)
        J_block = isA ? JA : JB

        # Check if in tension or compression
        in_tension = C_loc[1] > 0
        current_k = in_tension ? con.k_tension : con.k_compression

        # Should not have penalty_k anymore but select based on which state it is in, tension or compression
        for r in 1:3
            k_val = min(current_k[r], k_limit[r])
            k_val <= 0.0 && continue

            J_row = J_block[r, :]
            f_mag = k_val * C_loc[r]

            H += (J_row * J_row') * k_val
            F -= J_row * f_mag
        end
    end

    for con in contacts
        compute_constraint!(con, alpha)
        AVBDConstraints.update_bounds!(con)

        isA = (con.bodyA.id == b.id)
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
    # Dual update will now return a max constraint violation 
    max_violation = 0.0

    for con in sim.bond_constraints

        if con.is_broken
            continue
        end

        C_loc, _, _, k_limit = eval_bond(con, sim.dt) # get effective stiffness from evaluating the damaged bond

        # Check maximum violation of bonds
        max_violation = max(max_violation, maximum(abs.(C_loc)))

        # Check load type
        in_tension = C_loc[1] > 0

        if in_tension
            pk = con.k_tension
            for r in 1:3
                new_k = pk[r] + sim.beta * abs(C_loc[r])
                new_k = min(new_k, min(con.k_max[r], k_limit[r]))
                pk = setindex(pk, new_k, r)
            end
            con.k_tension = pk
        else
            pk = con.k_compression
            for r in 1:3
                new_k = pk[r] + sim.beta * abs(C_loc[r])
                new_k = min(new_k, min(con.k_max[r], k_limit[r]))
                pk = setindex(pk, new_k, r)
            end
            con.k_compression = pk
        end
    end

    #Threads.@threads 
    for con in sim.contact_constraints

        compute_constraint!(con, alpha)

        # Check max violation contacts
        max_violation = max(max_violation, maximum(abs.(con.C)))

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

        AVBDConstraints.update_bounds!(con)
    end

    return max_violation
end

function update_vel!(sim::SimulationState, invdt::Float64)
    for b in sim.bodies
        b.is_static && continue

        # dont update kinematic bodies, let them keep moving @ set velocity
        if b.inv_mass == 0.0
            continue
        end

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

    #HACK adding weibull randomization to the material properties
    weibull_shape = 5.0

    n_bonds = size(bond_data, 1)
    if n_bonds > 0
        sizehint!(sim.bond_constraints, n_bonds)
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

            U = rand() # Uniform [0,1]
            random_factor = (-log(U))^(1.0 / weibull_shape)

            tensile = row[15] * random_factor
            Gc = row[16] * random_factor
            damp_val = size(bond_data, 2) >= 17 ? row[17] : 0.0
            bA = bodies[idxA+1]
            bB = bodies[idxB+1]


            # set bond break steps & avoid infinite mass issues 
            mass_bond = (bA.mass + bB.mass) / 2
            if !isfinite(mass_bond) || !isfinite(kt) || kt <= 0
                bond_break_steps = 10
            else
                t_wave = sqrt(mass_bond / kt)
                calc_steps = ceil(Int, (t_wave / dt) * 1.5)
                bond_break_steps = clamp(calc_steps, 3, 50)
            end


            bond = BondConstraint(bA, bB, pA, pB, n_w, kn, kt, area, tensile, Gc, damp_val)
            bond.max_break_steps = bond_break_steps
            push!(sim.bond_constraints, bond)
        end
    end

    return sim
end

function damage_bonds!(sim::SimulationState, dt::Float64)
    changed = false
    for bond in sim.bond_constraints
        if !bond.is_broken
            was_broken = bond.is_broken
            old_damage = bond.damage

            update_bond_state!(bond, dt)

            if bond.is_broken != was_broken || abs(bond.damage - old_damage) > 1e-4
                changed = true
            end
        end
    end
    return changed
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

    #total_iters = sim.iterations + (sim.stabilize ? 1 : 0)
    inner_passes = 15# sim.iterations * 5
    damage_passes = 2 #sim.iterations

    curr_max_violation = Inf

    applied_damage = false

    for out_it in 1:damage_passes

        for in_it in 1:inner_passes

            alpha_eff = sim.stabilize ? 1.0 : sim.alpha

            # Loop over bodies aka primal loop
            #Threads.@threads 
            for body in sim.bodies
                # solve constriants not in loop but via linear algebra
                body.is_static && continue

                # TODO include a L-scheme term here that adds numerical inertia
                primal_solve!(body,
                    sim.bond_incidence[body.id+1],
                    sim.contact_incidence[body.id+1],
                    dt, inv_dt2, alpha_eff)  # Uses eval_bond

                assert_finite_body!(body, "after primal_solve out_it=$out_it in_it=$in_it")
            end
            curr_max_violation = dual_update!(sim, alpha_eff)
        end
        topo_changed = false
        if !applied_damage
            topo_changed = damage_bonds!(sim, dt)
            applied_damage = true
        end

        # Early out - skip prescribed num of iterations if below tolerance
        if !topo_changed && curr_max_violation < EARLY_OUT_TOL
            break
        end

    end

    # Velocity derivation from new positions
    update_vel!(sim, inv_dt)

    # Stabilization pass
    if sim.stabilize
        alpha_stabil = 0.05
        for body in sim.bodies
            body.is_static && continue
            primal_solve!(body, sim.bond_incidence[body.id+1], sim.contact_incidence[body.id+1], dt, inv_dt2, alpha_stabil)
        end
        dual_update!(sim, alpha_stabil)
    end


end

end # module end
