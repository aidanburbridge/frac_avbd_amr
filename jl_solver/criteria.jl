module Criteria

using StaticArrays

using ...Maths: Vec3, Quat
import ...Collisions: update_rotation!, update_inv_inertia_world!
import ...AVBDConstraints: BondConstraint, get_effective_stiffness

export CriteriaConfig, RefineSpec, FractureSpec, RefineKind, FractureKind, LogicOperation,
    REFINE_LAMBDA, REFINE_ENERGY, FRAC_LAMBDA, FRAC_STRETCH, ANY, ALL,
    R13_REFINE_TOP_FRAC, MAX_REFINE_EVENTS_PER_STEP, MAX_BREAK_EVENTS_PER_STEP,
    REFINE_COOLDOWN_STEPS, FRACTURE_COOLDOWN_STEPS, CUTBACK_RETRY_ENABLED,
    CUTBACK_RETRY_MAX, CUTBACK_FACTOR, CUTBACK_EVENT_FRACTION,
    CUTBACK_VIOLATION_FACTOR, PENALTY_RAMP_MAX_DELTA,
    SimpleRefine, SimpleFracture,
    should_refine, should_fracture, endpoint_at_fracture_cap,
    endpoint_can_schedule_refine, apply_refine_budget,
    has_valid_child, local_max_ref_level, penalty_increment,
    decay_cooldowns!, cap_events, prepare_step_events,
    should_retry_step, can_retry_attempt, attempt_dt,
    StepSnapshot, snapshot_step_state, restore_step_state!

# ---------- Enums and Logic Params ---------- #

@enum LogicOperation ANY ALL

@enum RefineKind REFINE_LAMBDA REFINE_ENERGY
@enum FractureKind FRAC_LAMBDA FRAC_STRETCH

const R13_REFINE_TOP_FRAC = 0.005
const MAX_REFINE_EVENTS_PER_STEP = 128
const MAX_BREAK_EVENTS_PER_STEP = 128
const REFINE_COOLDOWN_STEPS = 2
const FRACTURE_COOLDOWN_STEPS = 2
const CUTBACK_RETRY_ENABLED = true
const CUTBACK_RETRY_MAX = 1
const CUTBACK_FACTOR = 0.5
const CUTBACK_EVENT_FRACTION = 0.05
const CUTBACK_VIOLATION_FACTOR = 10.0
const PENALTY_RAMP_MAX_DELTA = 1e6

# ---------- Structs ---------- #

struct RefineSpec
    kind::RefineKind
    params::SVector{2,Float64} # [threshold, unused]
end

struct FractureSpec
    kind::FractureKind
    params::SVector{2,Float64}
end

struct CriteriaConfig
    refine_specs::Vector{RefineSpec}
    refine_logic::LogicOperation
    fracture_specs::Vector{FractureSpec}
    fracture_logic::LogicOperation
end

# -------------------------------------------------- #
#                   Refinement
# -------------------------------------------------- #

@inline function _check_refinement_criteria(spec::RefineSpec, con, lam_r, r)::Bool
    # Simple lambda refinement check
    if spec.kind == REFINE_LAMBDA
        # Refine if tension is ever over lambda threshold: simple/dumb rule
        _bond_in_tension(con) || return false
        thresh_mul = spec.params[1]
        cap = max(con.fracture[r], eps(Float64))
        return abs(lam_r) >= thresh_mul * cap

    elseif spec.kind == REFINE_ENERGY
        # Refine if 
        r == 1 || return false
        _bond_in_tension(con) || return false

        k_eff = get_effective_stiffness(con)
        e_bond = 0.5 * (k_eff[1] * con.C[1]^2 + k_eff[2] * con.C[2]^2 + k_eff[3] * con.C[3]^2)

        # Use fracture energy as refinement cap
        cap1 = max(con.fracture[1], eps(Float64))
        cap2 = max(con.fracture[2], eps(Float64))
        cap3 = max(con.fracture[3], eps(Float64))
        k1 = max(con.stiffness[1], eps(Float64))
        k2 = max(con.stiffness[2], eps(Float64))
        k3 = max(con.stiffness[3], eps(Float64))
        e_cap = 0.5 * ((cap1^2) / k1 + (cap2^2) / k2 + (cap3^2) / k3)

        thresh_mul = spec.params[1]
        return e_bond >= thresh_mul * e_cap
    end
    return false
end

@inline function should_refine(config::CriteriaConfig, sim, con, lam_r, r)::Bool
    if config.refine_logic == ANY
        for spec in config.refine_specs
            # If any true, return true
            _check_refinement_criteria(spec, con, lam_r, r) && return true
        end
        return false
    else
        for spec in config.refine_specs
            # If any not true, return false
            _check_refinement_criteria(spec, con, lam_r, r) || return false
        end
        return true
    end
end

# -------------------------------------------------- #
#                   Fracture
# -------------------------------------------------- #

@inline function _check_fracture_criteria(spec::FractureSpec, con, lam_r, r)::Bool
    # Simple lambda fracture check
    if spec.kind == FRAC_LAMBDA
        # Fracture if over certain lambda (w/ ratio): dumb check
        _bond_in_tension(con) || return false
        thresh_mul = spec.params[1]
        cap = max(con.fracture[r], eps(Float64))
        return abs(lam_r) >= thresh_mul * cap

    elseif spec.kind == FRAC_STRETCH
        # Fracture if stretch is 
        r == 1 || return false
        _bond_in_tension(con) || return false

        # Rest length w/ safety
        len_0 = max(abs(con.rest[1]), eps(Float64))
        s = abs(con.C[1]) / len_0

        # Critical stretch derived from existing force cap + stiffness
        cap = max(con.fracture[1], eps(Float64))
        k_n = max(con.stiffness[1], eps(Float64))
        s_cap = cap / (k_n * len_0)

        thresh_mul = spec.params[1]
        return s >= thresh_mul * s_cap
    end

    return false
end

@inline function should_fracture(config::CriteriaConfig, sim, con, lam_r, r)::Bool
    if config.fracture_logic == ANY
        for spec in config.fracture_specs
            # If any true, return true
            _check_fracture_criteria(spec, con, lam_r, r) && return true
        end
        return false
    else
        for spec in config.fracture_specs
            # If any not true, return false
            _check_fracture_criteria(spec, con, lam_r, r) || return false
        end
        return true
    end
end

# ---------- Helper Functions ---------- #

@inline function _bond_in_tension(con)::Bool
    # Returns true only if bond is in tension
    sigma_n = con.penalty_k[1] * con.C[1]
    lam_n = clamp(sigma_n, con.f_min[1], con.f_max[1])
    return lam_n < 0.0
end

@inline function _has_valid_child(sim, body_idx::Int)::Bool
    if length(sim.can_refine) == length(sim.bodies)
        return sim.can_refine[body_idx]
    end
    node_id = sim.bodies[body_idx].id
    start0 = sim.children_start[node_id+1]
    count = sim.children_count[node_id+1]
    if start0 < 0 || count <= 0
        return false
    end
    @inbounds for child_node in start0:(start0 + count - 1)
        child_idx = child_node + 1
        if sim.valid_mask[child_idx]
            return true
        end
    end
    return false
end

@inline function _local_max_ref_level(sim, body_idx::Int)
    if length(sim.max_ref_level_per_body) == length(sim.bodies)
        return sim.max_ref_level_per_body[body_idx]
    end
    return sim.max_ref_level
end

@inline has_valid_child(sim, body_idx::Int) = _has_valid_child(sim, body_idx)
@inline local_max_ref_level(sim, body_idx::Int) = _local_max_ref_level(sim, body_idx)

@inline function endpoint_at_fracture_cap(sim, body_idx::Int)::Bool
    level = sim.level[body_idx]
    return (level >= _local_max_ref_level(sim, body_idx)) || !_has_valid_child(sim, body_idx)
end

@inline function endpoint_can_schedule_refine(sim, body_idx::Int, cap::Bool)::Bool
    cap && return false
    return sim.body_refine_cooldown[body_idx] == 0
end

@inline function apply_refine_budget(sim, refinement_list::Vector{Int}, refine_votes::Vector{Int})
    isempty(refinement_list) && return refinement_list

    candidates = unique(refinement_list)
    if R13_REFINE_TOP_FRAC >= 1.0
        return candidates
    end

    n_active = length(sim.active_body_ids)
    n_keep = ceil(Int, R13_REFINE_TOP_FRAC * n_active)
    n_keep = clamp(n_keep, 1, length(candidates))

    sort!(candidates, by=i -> refine_votes[i], rev=true)
    return candidates[1:n_keep]
end

@inline function penalty_increment(beta::Float64, C_r::Float64)::Float64
    return min(beta * abs(C_r), PENALTY_RAMP_MAX_DELTA)
end

@inline function decay_cooldowns!(sim)
    @inbounds for i in eachindex(sim.body_refine_cooldown)
        sim.body_refine_cooldown[i] > 0 && (sim.body_refine_cooldown[i] -= 1)
    end
    @inbounds for i in eachindex(sim.bond_fracture_cooldown)
        sim.bond_fracture_cooldown[i] > 0 && (sim.bond_fracture_cooldown[i] -= 1)
    end
end

@inline function cap_events(refine_ids::Vector{Int}, break_ids::Vector{Int})
    if MAX_REFINE_EVENTS_PER_STEP > 0 && length(refine_ids) > MAX_REFINE_EVENTS_PER_STEP
        refine_ids = refine_ids[1:MAX_REFINE_EVENTS_PER_STEP]
    end
    if MAX_BREAK_EVENTS_PER_STEP > 0 && length(break_ids) > MAX_BREAK_EVENTS_PER_STEP
        break_ids = break_ids[1:MAX_BREAK_EVENTS_PER_STEP]
    end
    return refine_ids, break_ids
end

@inline function should_retry_step(sim, max_violation::Float64, refine_count::Int, break_count::Int, early_out_tol::Float64)::Bool
    CUTBACK_RETRY_ENABLED || return false
    n_active = max(length(sim.active_body_ids), 1)
    burst_cap = ceil(Int, CUTBACK_EVENT_FRACTION * n_active)
    event_burst = (refine_count + break_count) > burst_cap
    poor_convergence = max_violation > (CUTBACK_VIOLATION_FACTOR * early_out_tol)
    return event_burst || poor_convergence
end

@inline function prepare_step_events(refine_list::Vector{Int}, fracture_list::Vector{Int})
    refine_ids = unique(refine_list)
    fracture_ids = unique(fracture_list)
    raw_refine_count = length(refine_ids)
    raw_fracture_count = length(fracture_ids)
    refine_ids, fracture_ids = cap_events(refine_ids, fracture_ids)
    return refine_ids, fracture_ids, raw_refine_count, raw_fracture_count
end

@inline function can_retry_attempt(attempt::Int)::Bool
    return CUTBACK_RETRY_ENABLED && (attempt < CUTBACK_RETRY_MAX)
end

@inline function attempt_dt(dt_base::Float64, attempt::Int)::Float64
    return dt_base * (CUTBACK_FACTOR^attempt)
end

struct StepSnapshot
    body_pos::Vector{Vec3}
    body_quat::Vector{Quat}
    body_vel::Vector{Vec3}
    body_ang_vel::Vector{Vec3}
    body_pos0::Vector{Vec3}
    body_quat0::Vector{Quat}
    body_pos_inertia::Vector{Vec3}
    body_quat_inertia::Vector{Quat}
    bond_lambda::Vector{Vec3}
    bond_penalty_k::Vector{Vec3}
end

@inline function snapshot_step_state(sim)
    return StepSnapshot(
        [b.pos for b in sim.bodies],
        [b.quat for b in sim.bodies],
        [b.vel for b in sim.bodies],
        [b.ang_vel for b in sim.bodies],
        [b.pos0 for b in sim.bodies],
        [b.quat0 for b in sim.bodies],
        [b.pos_inertia for b in sim.bodies],
        [b.quat_inertia for b in sim.bodies],
        [c.lambda for c in sim.bond_constraints],
        [c.penalty_k for c in sim.bond_constraints],
    )
end

@inline function restore_step_state!(sim, snap::StepSnapshot)
    @inbounds for i in eachindex(sim.bodies)
        b = sim.bodies[i]
        b.pos = snap.body_pos[i]
        b.quat = snap.body_quat[i]
        b.vel = snap.body_vel[i]
        b.ang_vel = snap.body_ang_vel[i]
        b.pos0 = snap.body_pos0[i]
        b.quat0 = snap.body_quat0[i]
        b.pos_inertia = snap.body_pos_inertia[i]
        b.quat_inertia = snap.body_quat_inertia[i]
        update_rotation!(b)
        update_inv_inertia_world!(b)
    end
    @inbounds for i in eachindex(sim.bond_constraints)
        c = sim.bond_constraints[i]
        c.lambda = snap.bond_lambda[i]
        c.penalty_k = snap.bond_penalty_k[i]
    end
end


end
