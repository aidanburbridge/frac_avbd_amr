module Criteria

using StaticArrays

using ...Maths: Vec3, Quat
import ...Collisions: update_rotation!, update_inv_inertia_world!
import ...AVBDConstraints: BondConstraint, get_effective_stiffness

export CriteriaConfig, RefineSpec, FractureSpec, RefineKind, FractureKind, LogicOperation,
    REFINE_LAMBDA, REFINE_ENERGY,
    REFINE_R2_DAMAGE_BAND, REFINE_R3_NEAR_INIT, REFINE_R4_GRADIENT,
    REFINE_R8_STRETCH, REFINE_R9_KAPPA,
    FRAC_LAMBDA, FRAC_STRETCH, FRAC_ENERGY, FRAC_CZM, ANY, ALL,
    R13_REFINE_TOP_FRAC, MAX_REFINE_EVENTS_PER_STEP, MAX_BREAK_EVENTS_PER_STEP,
    REFINE_COOLDOWN_STEPS, FRACTURE_COOLDOWN_STEPS, CUTBACK_RETRY_ENABLED,
    CUTBACK_RETRY_MAX, CUTBACK_FACTOR, CUTBACK_EVENT_FRACTION,
    CUTBACK_VIOLATION_FACTOR, PENALTY_RAMP_MAX_DELTA,
    KELVIN_VOIGT_ENABLED, KAPPA_NONLOCAL_BLEND,
    ENABLE_NONLOCAL_KAPPA, ENABLE_PERSISTENCE, ENABLE_HYSTERESIS, ENABLE_RATE_DAMAGE,
    REF_PERSIST_STEPS, FRAC_PERSIST_STEPS,
    REF_HYST_ON, REF_HYST_OFF, FRAC_HYST_ON, FRAC_HYST_OFF,
    DAMAGE_TAU, DAMAGE_SOFT_BAND, CRACK_BAND_REF_LEN,
    ENERGY_GUARD_ENABLED, ENERGY_GUARD_REL_JUMP,
    SimpleRefine, SimpleFracture,
    should_refine, should_fracture, endpoint_at_fracture_cap,
    endpoint_can_schedule_refine, apply_refine_budget,
    has_valid_child, local_max_ref_level, penalty_increment,
    bond_sigma, bond_viscous_dissipation, commit_bond_history!,
    decay_cooldowns!, cap_events, prepare_step_events,
    should_retry_step, can_retry_attempt, attempt_dt, energy_guard_trip,
    StepSnapshot, snapshot_step_state, restore_step_state!

# ---------- Enums and Logic Params ---------- #

@enum LogicOperation ANY ALL

@enum RefineKind REFINE_LAMBDA REFINE_ENERGY REFINE_R2_DAMAGE_BAND REFINE_R3_NEAR_INIT REFINE_R4_GRADIENT REFINE_R8_STRETCH REFINE_R9_KAPPA
@enum FractureKind FRAC_LAMBDA FRAC_STRETCH FRAC_ENERGY FRAC_CZM

const R13_REFINE_TOP_FRAC = 0.002
const MAX_REFINE_EVENTS_PER_STEP = 32
const MAX_BREAK_EVENTS_PER_STEP = 16
const REFINE_COOLDOWN_STEPS = 2
const FRACTURE_COOLDOWN_STEPS = 2
const CUTBACK_RETRY_ENABLED = false
const CUTBACK_RETRY_MAX = 1
const CUTBACK_FACTOR = 0.5
const CUTBACK_EVENT_FRACTION = 0.05
const CUTBACK_VIOLATION_FACTOR = 10.0
const PENALTY_RAMP_MAX_DELTA = 1e6
const KELVIN_VOIGT_ENABLED = false
const KAPPA_NONLOCAL_BLEND = 0.35
const ENABLE_NONLOCAL_KAPPA = false
const ENABLE_PERSISTENCE = false
const ENABLE_HYSTERESIS = false
const ENABLE_RATE_DAMAGE = false
const REF_PERSIST_STEPS = 2
const FRAC_PERSIST_STEPS = 2
const REF_HYST_ON = 0.85
const REF_HYST_OFF = 0.75
const FRAC_HYST_ON = 1.0
const FRAC_HYST_OFF = 0.90
const DAMAGE_TAU = 5.0e-4
const DAMAGE_SOFT_BAND = 0.25
const CRACK_BAND_REF_LEN = 0.0  # <=0 disables scaling
const ENERGY_GUARD_ENABLED = false
const ENERGY_GUARD_REL_JUMP = 5.0

const _REFINE_PERSIST = Dict{UInt64,Int}()
const _FRACTURE_PERSIST = Dict{UInt64,Int}()
const _REFINE_LATCH = Dict{UInt64,Bool}()
const _FRACTURE_LATCH = Dict{UInt64,Bool}()
const _ENERGY_KAPPA = Dict{UInt64,Float64}()

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
        # Energy-consistent process-zone refinement:
        # params[1] = lower kappa threshold (on), params[2] = optional upper threshold.
        # If params[2] <= 0, upper cap is disabled.
        r == 1 || return false

        kappa_e = _update_energy_kappa!(con)
        k_on = spec.params[1]
        k_upper = spec.params[2] > k_on ? spec.params[2] : Inf
        return (kappa_e >= k_on) && (kappa_e < k_upper)

    elseif spec.kind == REFINE_R2_DAMAGE_BAND
        # R2: process-zone indicator from partial damage (max at d=0.5)
        r == 1 || return false
        d = clamp(con.damage, 0.0, 1.0)
        indicator = d * (1.0 - d)
        thresh = spec.params[1]
        return indicator >= thresh

    elseif spec.kind == REFINE_R3_NEAR_INIT
        # R3: refine near mixed-mode traction initiation
        r == 1 || return false
        _bond_in_tension(con) || return false
        phi = _traction_utilization(con)
        thresh_mul = spec.params[1]
        return phi >= (thresh_mul^2)

    elseif spec.kind == REFINE_R4_GRADIENT
        # R4: local gradient proxy using change in bond deformation
        r == 1 || return false
        _bond_in_tension(con) || return false
        grad_proxy = _deformation_gradient_proxy(con)
        thresh = spec.params[1]
        return grad_proxy >= thresh

    elseif spec.kind == REFINE_R8_STRETCH
        # R8: effective mixed-mode stretch proximity
        r == 1 || return false
        _bond_in_tension(con) || return false
        s_eff = _effective_stretch(con)
        s_cap = _effective_stretch_cap(con)
        thresh_mul = spec.params[1]
        return s_eff >= thresh_mul * s_cap

    elseif spec.kind == REFINE_R9_KAPPA
        # R9: irreversible history variable (kappa) based refinement
        r == 1 || return false
        _bond_in_tension(con) || return false
        s_eff = _effective_stretch(con)
        con.max_committed_strain = max(con.max_committed_strain, s_eff)
        s_cap = _effective_stretch_cap(con)
        thresh_mul = spec.params[1]
        return con.max_committed_strain >= thresh_mul * s_cap
    end
    return false
end

@inline function should_refine(config::CriteriaConfig, sim, con, lam_r, r)::Bool
    r == 1 || return false

    raw = if config.refine_logic == ANY
        any(spec -> _check_refinement_criteria(spec, con, lam_r, r), config.refine_specs)
    else
        all(spec -> _check_refinement_criteria(spec, con, lam_r, r), config.refine_specs)
    end

    # Optional stabilization layers (disabled for simple debug mode).
    key = _bond_key(con)
    kappa = ENABLE_HYSTERESIS ? _update_kappa_nonlocal!(sim, con) : 0.0
    persist_ok = ENABLE_PERSISTENCE ? _update_persistence!(_REFINE_PERSIST, key, raw, REF_PERSIST_STEPS) : raw
    hyst_ok = ENABLE_HYSTERESIS ? _update_hysteresis!(_REFINE_LATCH, key, kappa, REF_HYST_ON, REF_HYST_OFF) : true
    return raw && persist_ok && hyst_ok
end

# -------------------------------------------------- #
#                   Fracture
# -------------------------------------------------- #

@inline function _check_fracture_criteria(spec::FractureSpec, sim, con, lam_r, r)::Bool
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

    elseif spec.kind == FRAC_ENERGY
        # Energy-based fracture with irreversible history variable.
        r == 1 || return false
        _bond_in_tension(con) || return false

        kappa_e = _update_energy_kappa!(con)
        k_frac = spec.params[1]
        if ENABLE_RATE_DAMAGE
            dt = max(sim.dt, eps(Float64))
            d_eq = clamp((kappa_e - k_frac) / max(DAMAGE_SOFT_BAND, eps(Float64)), 0.0, 1.0)
            d_trial = con.damage + (dt / max(DAMAGE_TAU, eps(Float64))) * (d_eq - con.damage)
            con.damage = clamp(max(con.damage, d_trial), 0.0, 1.0)
            return con.damage >= 1.0
        end
        return kappa_e >= k_frac

    elseif spec.kind == FRAC_CZM
        # Bundle 1 F12: simple mixed-mode CZM-style separation check
        r == 1 || return false
        _bond_in_tension(con) || return false

        # Opening-only normal separation under current sign convention
        dn = max(0.0, -con.C[1])
        dt1 = abs(con.C[2])
        dt2 = abs(con.C[3])

        # Convert existing force caps to separation caps via stiffness
        k1 = max(con.stiffness[1], eps(Float64))
        k2 = max(con.stiffness[2], eps(Float64))
        k3 = max(con.stiffness[3], eps(Float64))
        dn_cap = max(con.fracture[1], eps(Float64)) / k1
        dt1_cap = max(con.fracture[2], eps(Float64)) / k2
        dt2_cap = max(con.fracture[3], eps(Float64)) / k3

        # Quadratic mixed-mode interaction, trigger at >= 1
        phi = (dn / max(dn_cap, eps(Float64)))^2 +
              (dt1 / max(dt1_cap, eps(Float64)))^2 +
              (dt2 / max(dt2_cap, eps(Float64)))^2

        # Bundle-1 stabilization #15 and #3: rate-regularized, irreversible damage
        drive = sqrt(max(phi, 0.0))
        thresh_mul = spec.params[1]
        if ENABLE_RATE_DAMAGE
            dt = max(sim.dt, eps(Float64))
            d_eq = clamp((drive - 1.0) / max(DAMAGE_SOFT_BAND, eps(Float64)), 0.0, 1.0)
            d_trial = con.damage + (dt / max(DAMAGE_TAU, eps(Float64))) * (d_eq - con.damage)
            con.damage = clamp(max(con.damage, d_trial), 0.0, 1.0) # no healing
            return con.damage >= thresh_mul
        end
        return drive >= thresh_mul
    end

    return false
end

@inline function should_fracture(config::CriteriaConfig, sim, con, lam_r, r)::Bool
    r == 1 || return false

    raw = if config.fracture_logic == ANY
        any(spec -> _check_fracture_criteria(spec, sim, con, lam_r, r), config.fracture_specs)
    else
        all(spec -> _check_fracture_criteria(spec, sim, con, lam_r, r), config.fracture_specs)
    end

    # Optional stabilization layers (disabled for simple debug mode).
    key = _bond_key(con)
    kappa = ENABLE_HYSTERESIS ? _update_kappa_nonlocal!(sim, con) : 0.0
    persist_ok = ENABLE_PERSISTENCE ? _update_persistence!(_FRACTURE_PERSIST, key, raw, FRAC_PERSIST_STEPS) : raw
    hyst_ok = ENABLE_HYSTERESIS ? _update_hysteresis!(_FRACTURE_LATCH, key, kappa, FRAC_HYST_ON, FRAC_HYST_OFF) : true
    return raw && persist_ok && hyst_ok
end

# ---------- Helper Functions ---------- #

@inline function _bond_in_tension(con)::Bool
    # Returns true only if bond is in opening tension.
    # Keep sign convention consistent with get_effective_stiffness in avbd_constraints.jl,
    # where C[1] > 0 is treated as tensile opening.
    sigma_n = con.penalty_k[1] * con.C[1]
    lam_n = clamp(sigma_n, con.f_min[1], con.f_max[1])
    return lam_n > 0.0
end

@inline function _bond_strain_energy(con)::Float64
    k_eff = get_effective_stiffness(con)
    return 0.5 * (k_eff[1] * con.C[1]^2 + k_eff[2] * con.C[2]^2 + k_eff[3] * con.C[3]^2)
end

@inline function _energy_drive(con)::Float64
    e_cap = _fracture_energy_cap(con)
    e_cap = max(e_cap, eps(Float64))
    return _bond_strain_energy(con) / e_cap
end

@inline function _update_energy_kappa!(con)::Float64
    key = _bond_key(con)
    drive = _energy_drive(con)
    k_prev = get(_ENERGY_KAPPA, key, 0.0)
    k_new = max(k_prev, drive)
    _ENERGY_KAPPA[key] = k_new
    return k_new
end

@inline function _fracture_energy_cap(con)::Float64
    cap1 = max(con.fracture[1], eps(Float64))
    cap2 = max(con.fracture[2], eps(Float64))
    cap3 = max(con.fracture[3], eps(Float64))
    k1 = max(con.stiffness[1], eps(Float64))
    k2 = max(con.stiffness[2], eps(Float64))
    k3 = max(con.stiffness[3], eps(Float64))
    base = 0.5 * ((cap1^2) / k1 + (cap2^2) / k2 + (cap3^2) / k3)
    if CRACK_BAND_REF_LEN > 0.0
        h = _char_length(con)
        return base * (h / CRACK_BAND_REF_LEN)
    end
    return base
end

@inline function _char_length(con)::Float64
    return max(abs(con.rest[1]), eps(Float64))
end

@inline function _traction_utilization(con)::Float64
    sigma = con.penalty_k .* con.C
    lam = clamp.(sigma, con.f_min, con.f_max)

    cap1 = max(con.fracture[1], eps(Float64))
    cap2 = max(con.fracture[2], eps(Float64))
    cap3 = max(con.fracture[3], eps(Float64))

    lam_n_open = max(0.0, -lam[1])
    lam_t1 = abs(lam[2])
    lam_t2 = abs(lam[3])

    return (lam_n_open / cap1)^2 + (lam_t1 / cap2)^2 + (lam_t2 / cap3)^2
end

@inline function _effective_stretch(con)::Float64
    len0 = _char_length(con)
    dn = max(0.0, -con.C[1])
    dt = sqrt(con.C[2]^2 + con.C[3]^2)
    return sqrt(dn^2 + dt^2) / len0
end

@inline function _effective_stretch_cap(con)::Float64
    len0 = _char_length(con)
    k1 = max(con.stiffness[1], eps(Float64))
    k2 = max(con.stiffness[2], eps(Float64))
    k3 = max(con.stiffness[3], eps(Float64))

    dn_cap = max(con.fracture[1], eps(Float64)) / k1
    dt1_cap = max(con.fracture[2], eps(Float64)) / k2
    dt2_cap = max(con.fracture[3], eps(Float64)) / k3
    dt_cap = sqrt(dt1_cap^2 + dt2_cap^2)

    return sqrt(dn_cap^2 + dt_cap^2) / len0
end

@inline function _deformation_gradient_proxy(con)::Float64
    # Avoid a first-step artifact: C_prev is zero at initialization.
    if (abs(con.C_prev[1]) + abs(con.C_prev[2]) + abs(con.C_prev[3])) < 1e-14
        return 0.0
    end
    len0 = _char_length(con)
    d1 = con.C[1] - con.C_prev[1]
    d2 = con.C[2] - con.C_prev[2]
    d3 = con.C[3] - con.C_prev[3]
    return sqrt(d1^2 + d2^2 + d3^2) / len0
end

@inline function _bond_key(con)::UInt64
    return UInt64(objectid(con))
end

@inline function _update_persistence!(table::Dict{UInt64,Int}, key::UInt64, raw::Bool, n_req::Int)::Bool
    n_req <= 1 && return raw
    c = get(table, key, 0)
    if raw
        c = min(c + 1, n_req)
    else
        c = max(c - 1, 0)
    end
    table[key] = c
    return c >= n_req
end

@inline function _update_hysteresis!(table::Dict{UInt64,Bool}, key::UInt64, drive::Float64, on::Float64, off::Float64)::Bool
    is_on = get(table, key, false)
    if is_on
        is_on = drive >= off
    else
        is_on = drive >= on
    end
    table[key] = is_on
    return is_on
end

@inline function _local_kappa_drive(con)::Float64
    s_eff = _effective_stretch(con)
    s_cap = max(_effective_stretch_cap(con), eps(Float64))
    return s_eff / s_cap
end

@inline function _neighbor_nonlocal_drive(sim, con)::Float64
    a_idx = con.bodyA.id + 1
    b_idx = con.bodyB.id + 1
    total = 0.0
    count = 0
    for idx in (a_idx, b_idx)
        if idx < 1 || idx > length(sim.bond_incidence_all)
            continue
        end
        for con_id in sim.bond_incidence_all[idx]
            nb = sim.bond_constraints[con_id]
            nb.is_broken && continue
            nb.is_active || continue
            total += _local_kappa_drive(nb)
            count += 1
        end
    end
    if count == 0
        return _local_kappa_drive(con)
    end
    return total / count
end

@inline function _update_kappa_nonlocal!(sim, con)::Float64
    local_drive = _local_kappa_drive(con)
    ybar = if ENABLE_NONLOCAL_KAPPA
        nl_drive = _neighbor_nonlocal_drive(sim, con)
        (1.0 - KAPPA_NONLOCAL_BLEND) * local_drive + KAPPA_NONLOCAL_BLEND * nl_drive
    else
        local_drive
    end
    con.current_eff_strain = ybar
    con.max_eff_strain = max(con.max_eff_strain, ybar)
    return con.max_eff_strain
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

@inline function _refine_cap_distance(sim, body_idx::Int)::Int
    return max(_local_max_ref_level(sim, body_idx) - sim.level[body_idx], 0)
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

    # Gate refinement toward fracture-cap closure:
    # prioritize bodies one split away from local cap, then by drive votes.
    sort!(candidates, by=i -> (
        _refine_cap_distance(sim, i) <= 1 ? 1 : 0,
        refine_votes[i],
        sim.level[i]
    ), rev=true)
    return candidates[1:n_keep]
end

@inline function penalty_increment(beta::Float64, C_r::Float64)::Float64
    return min(beta * abs(C_r), PENALTY_RAMP_MAX_DELTA)
end

@inline function bond_sigma(sim, con, pk_r::Float64, r::Int)::Float64
    sigma = pk_r * con.C[r]
    KELVIN_VOIGT_ENABLED || return sigma

    eta = max(con.viscosity, 0.0)
    eta <= 0.0 && return sigma

    dt = max(sim.dt, eps(Float64))
    c_dot = (con.C[r] - con.C_prev[r]) / dt
    return sigma + eta * c_dot
end

@inline function bond_viscous_dissipation(sim, con)::Float64
    KELVIN_VOIGT_ENABLED || return 0.0
    eta = max(con.viscosity, 0.0)
    eta <= 0.0 && return 0.0
    dt = max(sim.dt, eps(Float64))
    c_dot = (con.C - con.C_prev) / dt
    return dt * eta * (c_dot[1]^2 + c_dot[2]^2 + c_dot[3]^2)
end

@inline function commit_bond_history!(con)
    con.C_prev = con.C
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

@inline function energy_guard_trip(sim, trial_accounted::Union{Nothing,Float64})::Bool
    ENERGY_GUARD_ENABLED || return false
    trial_accounted === nothing && return false
    isfinite(trial_accounted) || return true

    log = sim.energy_log
    isempty(log.accounted_energy) && return false

    prev = log.accounted_energy[end]
    abs(prev) < 1e-9 && return false
    denom = max(abs(prev), 1e-12)
    rel_jump = abs(trial_accounted - prev) / denom
    return rel_jump > ENERGY_GUARD_REL_JUMP
end

@inline function should_retry_step(sim, max_violation::Float64, refine_count::Int, break_count::Int, early_out_tol::Float64;
    trial_accounted::Union{Nothing,Float64}=nothing)::Bool
    CUTBACK_RETRY_ENABLED || return false
    n_active = max(length(sim.active_body_ids), 1)
    burst_cap = ceil(Int, CUTBACK_EVENT_FRACTION * n_active)
    event_burst = (refine_count + break_count) > burst_cap
    poor_convergence = max_violation > (CUTBACK_VIOLATION_FACTOR * early_out_tol)
    bad_energy = energy_guard_trip(sim, trial_accounted)
    return event_burst || poor_convergence || bad_energy
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
    accumulated_fracture_work::Float64
    accumulated_viscous_work::Float64
    refine_persist::Dict{UInt64,Int}
    fracture_persist::Dict{UInt64,Int}
    refine_latch::Dict{UInt64,Bool}
    fracture_latch::Dict{UInt64,Bool}
    energy_kappa::Dict{UInt64,Float64}
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
        sim.energy_log.accumulated_fracture_work,
        sim.energy_log.accumulated_viscous_work,
        copy(_REFINE_PERSIST),
        copy(_FRACTURE_PERSIST),
        copy(_REFINE_LATCH),
        copy(_FRACTURE_LATCH),
        copy(_ENERGY_KAPPA),
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
    sim.energy_log.accumulated_fracture_work = snap.accumulated_fracture_work
    sim.energy_log.accumulated_viscous_work = snap.accumulated_viscous_work

    empty!(_REFINE_PERSIST); merge!(_REFINE_PERSIST, snap.refine_persist)
    empty!(_FRACTURE_PERSIST); merge!(_FRACTURE_PERSIST, snap.fracture_persist)
    empty!(_REFINE_LATCH); merge!(_REFINE_LATCH, snap.refine_latch)
    empty!(_FRACTURE_LATCH); merge!(_FRACTURE_LATCH, snap.fracture_latch)
    empty!(_ENERGY_KAPPA); merge!(_ENERGY_KAPPA, snap.energy_kappa)
end


end
