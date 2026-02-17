module AVBDCore

# TODO CHECK avbd_core.jl, avbd_constraints.jl, manifold.jl and physics_bridge.jl - one of these screws with the voxel size??? Still error voxelizing to resolution!!!

using LinearAlgebra
using StaticArrays
using Base.Threads
using ..Maths: Vec3, Quat, integrate_quat, quat_to_rotvec, quat_mul, quat_inv, rotvec_to_quat, delta_twist_from, rotate_vec
import ..Collisions: Body, Contact, get_collisions_active, update_rotation!, update_inv_inertia_world!
using ..AVBDConstraints
using ..ManifoldHandling: Manifold, init_manifold, update_manifold_dynamic!
using ..Energy

export SimulationState, init_simulation, step_simulation!

# TODO things to implement
# - have damage only apply to tension -> leave the stiffness of compresison untouched
# - ensure both compressive and tensile strain and stress are exported to the visualizer
# - need to somehow stabilize the fracture through 
# - early out for fast convergences

const EARLY_OUT_TOL = 1e-5 # TODO have this as a parameter as a fraction of h (voxel size) ~0.1% 

# Debug: force refinement of all active bodies (set false to disable)
const DEBUG_REFINE_ALL = false
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

    bond_incidence::Vector{Vector{Int}} # active bond ids per body
    contact_incidence::Vector{Vector{Int}}
    bond_incidence_all::Vector{Vector{Int}} # all bond ids per body (static)

    energy_log::EnergyLog

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

    # AMR
    active_body_ids::Vector{Int}
    active_bond_ids::Vector{Int} # active bonds (rebuilt from active bodies)
    bond_mark::Vector{Int} # epoch marks for active bond rebuild
    bond_mark_epoch::Int

    # Hierarchy lists
    # TODO there must be a way to make this somehow more compact - do I really need all of these lists?
    active::BitVector # TODO make this the single source of truth
    valid_mask::BitVector
    can_refine::BitVector
    level::Vector{Int} # TODO - why not make level the single source of truth and use -1 if not active?
    parent_list::Vector{Int}
    children_start::Vector{Int}
    children_count::Vector{Int}

    neighbor_map::Matrix{Int}

    max_ref_level::Int

    function SimulationState(bodies, dt; grav=-9.81, iters=10, mu=0.6, b=10.0, g=0.99, al=0.95, stabil=true)
        # Initialize constraint vectors
        bond_cons = Vector{BondConstraint}()
        contact_cons = Vector{ContactConstraint}()

        # Initialize constraint maps
        bond_map = Vector{Vector{Int}}()
        contact_map = Vector{Vector{Int}}()
        bond_map_all = Vector{Vector{Int}}()

        manifolds = Dict{Tuple{Int,Int},Manifold}()

        e_log = EnergyLog()

        # AMR
        active = falses(length(bodies)) # default all inactive OR set later
        valid_mask = trues(length(bodies))
        can_refine = BitVector()
        active_body_ids = Int[]
        active_bond_ids = Int[]
        bond_mark = Int[]
        bond_mark_epoch = 0
        level = fill(0, length(bodies))
        parent_list = Int[]
        children_start = Int[]
        children_count = Int[]

        # Neighbor map for 2:1 refinement check
        neighbor_map = Matrix{Int}(undef, 0, 0)

        # TODO must pass max_level in here
        max_level = 1 # default 1 for now but should come from octree

        new(bodies, manifolds, bond_cons, contact_cons, bond_map, contact_map, bond_map_all, e_log,
            dt, grav, mu, iters, Float64(b), Float64(g), Float64(al), stabil,
            active_body_ids, active_bond_ids, bond_mark, bond_mark_epoch,
            active, valid_mask, can_refine, level, parent_list, children_start, children_count,
            neighbor_map, max_level)

    end
end

function predict_inertia!(sim::SimulationState) # Why do I pass dt and gravity if they are in simulation state struct??
    dt = sim.dt
    gravity_vec = Vec3(0.0, sim.gravity, 0.0)

    for b_id in sim.active_body_ids
        b = sim.bodies[b_id]

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

    raw_contacts = get_collisions_active(sim.bodies, sim.active_body_ids)
    raw_count = length(raw_contacts)

    pair_groups = Dict{Tuple{Int,Int},Vector{Contact}}()

    for c in raw_contacts
        idA, idB = c.body_idx_a, c.body_idx_b

        # TODO HACK -> replace so raw_contacts = get_collisions(sim.bodies, sim.active_body_ids)
        if !sim.active[idA+1] || !sim.active[idB+1]
            continue
        end

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
            if !isassigned(sim.bond_incidence, i)
                sim.bond_incidence[i] = Int[]
            end
            if !isassigned(sim.contact_incidence, i)
                sim.contact_incidence[i] = Int[]
            end
        end
    end

    for i in sim.active_body_ids
        empty!(sim.bond_incidence[i])
        empty!(sim.contact_incidence[i])
    end

    # Loop bonds first (active bodies only)
    empty!(sim.active_bond_ids)

    if length(sim.bond_mark) != length(sim.bond_constraints)
        sim.bond_mark = zeros(Int, length(sim.bond_constraints))
        sim.bond_mark_epoch = 0
    end

    sim.bond_mark_epoch += 1
    if sim.bond_mark_epoch == typemax(Int)
        fill!(sim.bond_mark, 0)
        sim.bond_mark_epoch = 1
    end

    for b_idx in sim.active_body_ids
        for con_id in sim.bond_incidence_all[b_idx]
            if sim.bond_mark[con_id] == sim.bond_mark_epoch
                continue
            end
            sim.bond_mark[con_id] = sim.bond_mark_epoch

            con = sim.bond_constraints[con_id]

            # Skip broken/inactive bonds
            con.is_broken && continue
            con.is_active || continue

            a_idx = con.bodyA.id + 1
            b_idx = con.bodyB.id + 1
            if !sim.active[a_idx] || !sim.active[b_idx]
                continue
            end

            initialize!(con)

            # MUST CLAMP TO Material stiffness!!

            con.penalty_k = clamp.(con.penalty_k .* sim.gamma, con.k_min, con.k_max)

            if !sim.stabilize
                con.lambda = con.lambda .* (sim.alpha * sim.gamma)
            end

            k_eff = get_effective_stiffness(con)
            pk = con.penalty_k
            for r in 1:3
                pk = setindex(pk, min(pk[r], k_eff[r]), r)
            end
            con.penalty_k = pk

            # Build incidence_map (endpoints already verified active)
            push!(sim.bond_incidence[a_idx], con_id)
            push!(sim.bond_incidence[b_idx], con_id)
            push!(sim.active_bond_ids, con_id)
        end
    end

    # Loop contacts
    for con_id in eachindex(sim.contact_constraints)

        con = sim.contact_constraints[con_id]

        a_idx = con.bodyA.id + 1
        b_idx = con.bodyB.id + 1
        if !sim.active[a_idx] || !sim.active[b_idx]
            continue
        end

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

        # Build incidence_map (endpoints already verified active)
        push!(sim.contact_incidence[a_idx], con_id)
        push!(sim.contact_incidence[b_idx], con_id)
    end
end

function primal_solve!(sim::SimulationState, b::Body, bonds::Vector{Int}, contacts::Vector{Int}, dt::Float64, invdt2::Float64, alpha::Float64)
    b.inv_mass == 0.0 && return

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

    for con_id in bonds

        con = sim.bond_constraints[con_id]
        con.is_broken && continue
        con.is_active || continue

        eval_bond(con)

        isA = (con.bodyA.id == b.id)
        J_block = isA ? con.JA : con.JB

        for r in 1:3
            k_val = con.penalty_k[r]
            k_val <= 0.0 && continue

            J_row = J_block[r, :]

            f_mag = clamp(k_val * con.C[r], con.f_min[r], con.f_max[r])

            H += (J_row * J_row') * k_val
            F -= J_row * f_mag
        end
    end

    for con_id in contacts
        con = sim.contact_constraints[con_id]

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

    # List of bodies to be refined 
    refinement_list = Int[]

    for con_id in sim.active_bond_ids
        con = sim.bond_constraints[con_id]

        con.is_broken && continue
        con.is_active || continue

        eval_bond(con)
        max_violation = max(max_violation, maximum(abs.(con.C)))

        pk = con.penalty_k
        lam = con.lambda

        # TODO need a way to get the level & check for refinement/fracture - this is probabaly a bad way
        a_idx = con.bodyA.id + 1
        b_idx = con.bodyB.id + 1

        if !sim.active[a_idx] || !sim.active[b_idx]
            continue
        end

        levelA = sim.level[a_idx]
        levelB = sim.level[b_idx]

        # TODO essentially need to check if over threshold -> refine, UNLESS already @ max_ref_level, then break/fracture

        for r in 1:3
            #lambda_base = isinf(con.stiffness[r]) ? lam[r] : 0.0
            sigma = pk[r] * con.C[r]# + lambda_base
            lam_r = clamp(sigma, con.f_min[r], con.f_max[r])
            lam = setindex(lam, lam_r, r)

            # TODO change the reference critical value later to something with more grounding
            ref_crit = con.fracture[r] * 0.8 #TODO replace with refine ratio!

            # TODO REFINEMENT CRITERIA
            if abs(lam_r) >= ref_crit
                # TODO mark or trigger refinement process
                if levelA < sim.max_ref_level
                    push!(refinement_list, a_idx)
                end
                if levelB < sim.max_ref_level
                    push!(refinement_list, b_idx)
                end
            end

            # TODO add refinement check before fracturing! -> do NOT allow fracture unless @ finest level
            # FRACTURE CRITERIA
            if abs(lam_r) >= (con.fracture[r] * 10)
                if (levelA >= sim.max_ref_level) && (levelB >= sim.max_ref_level)
                    # d = abs(sigma) - abs(lam_r)
                    # con.damage += d
                    con.is_broken = true
                    con.penalty_k = @SVector zeros(3)
                    con.lambda = @SVector zeros(3)
                    break
                else
                    # NOT @ finest level -> push to refine
                    if levelA < sim.max_ref_level
                        push!(refinement_list, a_idx)
                    end
                    if levelB < sim.max_ref_level
                        push!(refinement_list, b_idx)
                    end
                end
            end

            if lam_r > con.f_min[r] && lam_r < con.f_max[r]
                new_k = pk[r] + sim.beta * abs(con.C[r])
                new_k = min(new_k, con.k_max[r])
                if !isinf(con.stiffness[r])
                    new_k = min(new_k, con.stiffness[r])
                end
                pk = setindex(pk, new_k, r)
            end
        end

        if !con.is_broken
            con.penalty_k = pk
            con.lambda = lam
        end
    end

    #Threads.@threads 
    for con in sim.contact_constraints

        compute_constraint!(con, alpha)
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

    return max_violation, refinement_list
end

function update_vel!(sim::SimulationState, invdt::Float64)

    for b_id in sim.active_body_ids
        body = sim.bodies[b_id]

        body.is_static && continue
        body.inv_mass == 0.0 && continue

        body.vel = (body.pos - body.pos0) * invdt #* 0.1 -> velocity issue isolated

        q_rel = quat_mul(body.quat, quat_inv(body.quat0))
        d_theta = quat_to_rotvec(q_rel)
        body.ang_vel = d_theta * invdt# * 0.1

        update_rotation!(body)
        update_inv_inertia_world!(body)
    end
end

function init_simulation(
    pos_flat::Matrix{Float64},
    vel_flat::Matrix{Float64},
    masses::Vector{Float64},
    bond_data::Matrix{Float64},
    dt::Float64, gravity::Float64, iterations::Int;
    friction::Float64=0.5,
    sizes::Union{AbstractMatrix,Nothing}=nothing,
    assembly_ids::Union{Vector{Int},Nothing}=nothing,
    active::Union{AbstractVector{Bool},Nothing}=nothing,
    valid_mask::Union{AbstractVector{Bool},Nothing}=nothing,
    can_refine::Union{AbstractVector{Bool},Nothing}=nothing,
    level::Union{AbstractVector{<:Integer},Nothing}=nothing,
    parent_list::Union{AbstractVector{<:Integer},Nothing}=nothing,
    children_start::Union{AbstractVector{<:Integer},Nothing}=nothing,
    children_count::Union{AbstractVector{<:Integer},Nothing}=nothing,
    neighbor_map::Union{AbstractMatrix{<:Integer},Nothing}=nothing,
    max_ref_level::Union{Int,Nothing}=nothing,)

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

    if neighbor_map !== nothing
        sim.neighbor_map = Int.(neighbor_map)
        # DEBUG
        @show size(sim.neighbor_map)
    end


    # TODO cleaner way to do this?
    # Apply AMR arrays if provided
    if active !== nothing
        sim.active = BitVector(active)
    else
        sim.active = falses(n_bodies)
    end
    if valid_mask !== nothing
        sim.valid_mask = BitVector(valid_mask)
    else
        sim.valid_mask = trues(n_bodies)
    end
    if can_refine !== nothing
        sim.can_refine = BitVector(can_refine)
    else
        sim.can_refine = BitVector()
    end
    rebuild_active_body_ids!(sim)

    if level !== nothing
        sim.level = Int.(level)
    end
    if parent_list !== nothing
        sim.parent_list = Int.(parent_list)
    end
    if children_start !== nothing
        sim.children_start = Int.(children_start)
    end
    if children_count !== nothing
        sim.children_count = Int.(children_count)
    end
    if neighbor_map !== nothing
        sim.neighbor_map = Int.(neighbor_map)
    end
    if max_ref_level !== nothing
        sim.max_ref_level = max_ref_level
    end

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

            # Kelvin-Voigt
            N = 25
            tau = N * dt
            bond.viscosity = kn * tau

            bond.is_active = true

            push!(sim.bond_constraints, bond)
        end
    end

    # Build static bond incidence (all bonds per body) and mark buffer
    resize!(sim.bond_incidence_all, n_bodies)
    for i in 1:n_bodies
        if !isassigned(sim.bond_incidence_all, i)
            sim.bond_incidence_all[i] = Int[]
        else
            empty!(sim.bond_incidence_all[i])
        end
    end
    for (con_id, con) in enumerate(sim.bond_constraints)
        a_idx = con.bodyA.id + 1
        b_idx = con.bodyB.id + 1
        push!(sim.bond_incidence_all[a_idx], con_id)
        push!(sim.bond_incidence_all[b_idx], con_id)
    end
    sim.bond_mark = zeros(Int, length(sim.bond_constraints))
    sim.bond_mark_epoch = 0

    return sim
end

function damage_bonds!(sim::SimulationState, dt::Float64)
    changed = false

    function bond_energy(b::BondConstraint)
        b.is_broken && return 0.0

        # Recalculate C based on current positions (update_bond_state! does this, 
        # but we need it before update to capture 'Pre-Damage' state)
        rA = AVBDConstraints.rotate_vec(b.pA_local, b.bodyA.quat)
        rB = AVBDConstraints.rotate_vec(b.pB_local, b.bodyB.quat)
        dp = (b.bodyA.pos + rA) - (b.bodyB.pos + rB)

        rotA = AVBDConstraints.quat_to_rotmat(b.bodyA.quat)
        c_n = dot(rotA * b.n_local, dp) - b.rest[1]
        c_t1 = dot(rotA * b.t1_local, dp) - b.rest[2]
        c_t2 = dot(rotA * b.t2_local, dp) - b.rest[3]
        C_local = @SVector [c_n, c_t1, c_t2]

        k_eff = AVBDConstraints.get_effective_stiffness(b)

        en = 0.5 * (k_eff[1] * C_local[1]^2 + k_eff[2] * C_local[2]^2 + k_eff[3] * C_local[3]^2)
        return en
    end

    for bond in sim.bond_constraints
        if !bond.is_broken
            was_broken = bond.is_broken
            old_damage = bond.damage

            # Calculate energy before updating bond
            E_pre = bond_energy(bond)

            # Calculate energy after update
            E_post = bond_energy(bond)

            diff = E_pre - E_post
            if diff > 0
                record_fracture_work!(sim.energy_log, diff)
            end

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

    # TODO CHECK time diffrence between running with collision detection and contact constraints vs. just bonds
    # Build contact constraints
    detect_collisions!(sim)

    # warm start constraints
    warm_start!(sim) # Python builds incident map here - inside the constraint loop 

    #total_iters = sim.iterations + (sim.stabilize ? 1 : 0)
    inner_passes = sim.iterations

    curr_max_violation = Inf
    refine_list = Int[]

    for in_it in 1:inner_passes

        alpha_eff = sim.stabilize ? 1.0 : sim.alpha

        # Loop over bodies aka primal loop
        for b_id in sim.active_body_ids
            body = sim.bodies[b_id]
            # solve constriants not in loop but via linear algebra
            body.is_static && continue

            # TODO include a L-scheme term here that adds numerical inertia
            primal_solve!(sim, body,
                sim.bond_incidence[body.id+1],
                sim.contact_incidence[body.id+1],
                dt, inv_dt2, alpha_eff)  # Uses eval_bond

            #assert_finite_body!(body, "after primal_solve in_it=$in_it")
        end

        # TODO should dual update output a list of voxels to refine OR should it happen in dual update? -> probably shouldn't
        curr_max_violation, refine_list = dual_update!(sim, alpha_eff)

        # Early out - skip prescribed num of iterations if below tolerance
        curr_max_violation < EARLY_OUT_TOL && break

    end

    if DEBUG_REFINE_ALL
        refine_list = copy(sim.active_body_ids)
    end

    # # TODO do refinement step here
    if !isempty(refine_list)
        refine_list = unique(refine_list)
        if size(sim.neighbor_map, 2) == 6
            refine_list = enforce_2to1!(sim, refine_list)
        end
        for v in refine_list
            refine_voxel!(sim, v)
        end
        rebuild_active_body_ids!(sim)
    end

    # Velocity derivation from new positions
    update_vel!(sim, inv_dt)

    alpha_log = sim.alpha

    # Stabilization pass
    if sim.stabilize
        alpha_stabil = 0.05
        for b_id in sim.active_body_ids
            body = sim.bodies[b_id]
            body.is_static && continue
            primal_solve!(sim, body, sim.bond_incidence[body.id+1], sim.contact_incidence[body.id+1], dt, inv_dt2, alpha_stabil)
        end
        dual_update!(sim, alpha_stabil)
        alpha_log = alpha_stabil
    end

    log_step!(sim.energy_log, sim.bodies, sim.bond_constraints, sim.contact_constraints, alpha_log,
        sim.active_body_ids, sim.active_bond_ids)

end

# ---------- AMR HELPER FUNCITONS ---------- #

@inline function resolve_active_neighbor(sim::SimulationState, nb_node::Int)
    nb_node < 0 && return 0
    nb_idx = nb_node + 1

    if sim.active[nb_idx]
        return nb_idx
    end

    # Walk up to active parent (coarser)
    p_node = sim.parent_list[nb_idx]
    while p_node >= 0
        p_idx = p_node + 1
        if sim.active[p_idx]
            return p_idx
        end
        p_node = sim.parent_list[p_idx]
    end

    return 0
end

@inline function has_valid_child(sim::SimulationState, body_idx::Int)
    if length(sim.can_refine) == length(sim.bodies)
        return sim.can_refine[body_idx]
    end
    node_id = sim.bodies[body_idx].id
    start0 = sim.children_start[node_id+1]
    count = sim.children_count[node_id+1]
    if start0 < 0 || count <= 0
        return false
    end
    for child_node in start0:(start0+count-1)
        child_idx = child_node + 1
        if sim.valid_mask[child_idx]
            return true
        end
    end
    return false
end

function enforce_2to1!(sim::SimulationState, refine_list::Vector{Int})
    marked = falses(length(sim.bodies))
    blocked = falses(length(sim.bodies))
    q = Int[]
    out = Int[]
    for id in refine_list
        if !marked[id]
            marked[id] = true
            push!(q, id)
            push!(out, id)
        end
    end

    while !isempty(q)
        v = pop!(q)
        sim.active[v] || continue
        blocked[v] && continue

        L = sim.level[v]
        target_level = L + 1
        if target_level > sim.max_ref_level
            blocked[v] = true
            continue
        end
        node_id = sim.bodies[v].id
        for dir in 1:6
            nb_node = sim.neighbor_map[node_id+1, dir]
            nb_idx = resolve_active_neighbor(sim, nb_node)
            nb_idx > 0 || continue

            if sim.level[nb_idx] < target_level - 1
                if has_valid_child(sim, nb_idx)
                    if !marked[nb_idx]
                        push!(q, nb_idx)
                        marked[nb_idx] = true
                        push!(out, nb_idx)
                    end
                else
                    # Neighbor cannot refine (no valid children) -> block this refine.
                    blocked[v] = true
                    break
                end
            end
        end
    end

    return [id for id in out if !blocked[id]]
end

function refine_voxel!(sim::SimulationState, parent_idx::Int, params=nothing)
    # Refines given voxel to children voxels and transfers kinematics.
    # parent_idx is a 1-based index into sim.bodies (Body.id == parent_idx - 1).

    if !sim.active[parent_idx]
        return
    end

    # TODO here active_body_ids is changed -> nowhere else 
    parent = sim.bodies[parent_idx]

    node_id = parent.id # 0-based id to index AMR lists
    start0 = sim.children_start[node_id+1]
    count = sim.children_count[node_id+1]

    if start0 < 0 || count <= 0
        return
    end

    has_valid_child = false
    for child_slot in 1:count
        child_node = start0 + (child_slot - 1)
        child_idx = child_node + 1
        if sim.valid_mask[child_idx]
            has_valid_child = true
            break
        end
    end
    if !has_valid_child
        return
    end

    # Deactivate parent
    set_active!(sim, parent_idx, false)

    activated_children = Int[]
    for child_slot in 1:count
        child_node = start0 + (child_slot - 1)  # child node id (0-based)
        child_idx = child_node + 1              # body index (1-based)
        child = sim.bodies[child_idx]

        if !sim.valid_mask[child_idx]
            continue
        end

        set_active!(sim, child_idx, true)
        push!(activated_children, child_idx)

        # Transfer kinematics (fallback uses octree fill order)
        transfer_kinematics!(parent, child, sim.dt; slot=child_slot)
    end

    if !isempty(activated_children)
        seed_child_lambdas!(sim, parent_idx, activated_children)
    end
end

function seed_child_lambdas!(sim::SimulationState, parent_idx::Int, child_indices::Vector{Int})
    # Distribute parent's bond lambda to new child bonds (per active neighbor).
    lambda_sum = Dict{Int,Vec3}()
    for con_id in sim.bond_incidence_all[parent_idx]
        con = sim.bond_constraints[con_id]
        con.is_broken && continue

        a_idx = con.bodyA.id + 1
        b_idx = con.bodyB.id + 1
        other_idx = (a_idx == parent_idx) ? b_idx : (b_idx == parent_idx ? a_idx : 0)
        other_idx == 0 && continue
        sim.active[other_idx] || continue

        if haskey(lambda_sum, other_idx)
            lambda_sum[other_idx] = lambda_sum[other_idx] + con.lambda
        else
            lambda_sum[other_idx] = con.lambda
        end
    end

    isempty(lambda_sum) && return

    child_bonds = Dict{Int,Vector{Int}}()
    for child_idx in child_indices
        for con_id in sim.bond_incidence_all[child_idx]
            con = sim.bond_constraints[con_id]
            con.is_broken && continue

            a_idx = con.bodyA.id + 1
            b_idx = con.bodyB.id + 1
            other_idx = (a_idx == child_idx) ? b_idx : (b_idx == child_idx ? a_idx : 0)
            other_idx == 0 && continue
            sim.active[other_idx] || continue
            haskey(lambda_sum, other_idx) || continue

            ids = get!(child_bonds, other_idx, Int[])
            push!(ids, con_id)
        end
    end

    for (other_idx, lam) in lambda_sum
        ids = get(child_bonds, other_idx, nothing)
        ids === nothing && continue
        n = length(ids)
        n == 0 && continue
        lam_share = lam / n
        for con_id in ids
            sim.bond_constraints[con_id].lambda = lam_share
        end
    end
end

@inline function set_active!(sim::SimulationState, body_idx::Int, on::Bool)
    # Pass true or false to on to set on or off
    sim.active[body_idx] = on
    sim.bodies[body_idx].is_active = on # for collisions
end

function rebuild_active_body_ids!(sim::SimulationState)
    # TODO should probably build this list ONCE and then empty and fill -- aka no resizing
    empty!(sim.active_body_ids)
    for i in eachindex(sim.active)
        if sim.active[i]
            push!(sim.active_body_ids, i)
        end
    end
end

@inline function _local_offset(parent::Body, child::Body)
    # Get child local offset in parent's local frame from prev proses 
    r_world = child.pos_prev - parent.pos_prev
    if all(isfinite, r_world) && !iszero(r_world)
        return rotate_vec(r_world, quat_inv(parent.quat_prev))
    end
    return nothing
end

# Helper for _local_offset
@inline iszero(v::Vec3) = (v[1] == 0.0 && v[2] == 0.0 && v[3] == 0.0)

@inline function _octant_local_offset(parent::Body, slot::Int)
    slot0 = slot - 1

    # Octree order from geometry/octree.py: di major, dj mid, dk minor (dk fastest)
    di = (slot0 >> 2) & 1
    dj = (slot0 >> 1) & 1
    dk = slot0 & 1
    half = parent.size * 0.25
    ox = di == 0 ? -half[1] : half[1]
    oy = dj == 0 ? -half[2] : half[2]
    oz = dk == 0 ? -half[3] : half[3]
    return Vec3(ox, oy, oz)
end

@inline function transfer_kinematics!(parent::Body, child::Body, dt::Float64; slot::Union{Int,Nothing}=nothing)
    # Compute child local offset in parent's rest frame (robust to missing children).
    # r_local = _local_offset(parent, child)
    # if r_local === nothing
    if slot === nothing
        r_local = Vec3(0.0, 0.0, 0.0)
    else
        r_local = _octant_local_offset(parent, slot)
    end
    # end

    # r_world = rotate_vec(r_local, quat_inv(parent.quat))
    r_world = rotate_vec(r_local, parent.quat)

    # Rigid transfer
    child.pos = parent.pos + r_world
    child.quat = parent.quat
    child.vel = parent.vel + cross(parent.ang_vel, r_world)
    child.ang_vel = parent.ang_vel

    # Keep integration state consistent if refinement happens mid-step.
    child.pos_inertia = child.pos
    child.quat_inertia = child.quat
    child.pos0 = child.pos - child.vel * dt
    delta_q = rotvec_to_quat(child.ang_vel * dt)
    child.quat0 = quat_mul(quat_inv(delta_q), child.quat)

    update_rotation!(child)
    update_inv_inertia_world!(child)
end

end # module end
