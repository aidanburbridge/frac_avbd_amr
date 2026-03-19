module Criteria

using LinearAlgebra
using StaticArrays

import ...AVBDConstraints: BondConstraint, get_effective_stiffness
import ...Maths: FLOAT, quat_to_rotmat, transform_point

export CriteriaConfig, RefineSpec, FractureSpec, RefineKind, FractureKind, LogicOperation,
    REFINE_LAMBDA, REFINE_ENERGY, REFINE_STRESS, FRAC_LAMBDA, FRAC_STRETCH, FRAC_ENERGY, ANY, ALL,
    SimpleRefine, StressRefine, SimpleFracture, default_criteria_config, build_criteria_config,
    compute_body_stress_data, collect_stress_refinement_candidates, should_refine, should_fracture

# ---------- Enums and Logic Params ---------- #

@enum LogicOperation ANY ALL

@enum RefineKind REFINE_LAMBDA REFINE_ENERGY REFINE_STRESS
@enum FractureKind FRAC_LAMBDA FRAC_STRETCH FRAC_ENERGY

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

SimpleRefine(threshold::Real=0.8) = RefineSpec(REFINE_LAMBDA, SVector(Float64(threshold), 0.0))
StressRefine(threshold::Real; exclude_kinematic::Bool=true) = RefineSpec(
    REFINE_STRESS,
    SVector(Float64(threshold), exclude_kinematic ? 1.0 : 0.0),
)
SimpleFracture(threshold::Real=1.0) = FractureSpec(FRAC_LAMBDA, SVector(Float64(threshold), 0.0))

function default_criteria_config()::CriteriaConfig
    ref_specs = [
        RefineSpec(REFINE_ENERGY, SVector(0.50, 1.0)),
    ]
    frac_specs = [
        FractureSpec(FRAC_ENERGY, SVector(1.00, 0.0)),
    ]
    return CriteriaConfig(ref_specs, ANY, frac_specs, ANY)
end

function build_criteria_config(;
    refine_stress_threshold::Union{Nothing,Real}=nothing,
    refine_stress_exclude_kinematic::Bool=true,
)::CriteriaConfig
    cfg = default_criteria_config()

    if refine_stress_threshold !== nothing && refine_stress_threshold > 0.0
        refine_specs = copy(cfg.refine_specs)
        push!(
            refine_specs,
            StressRefine(
                refine_stress_threshold;
                exclude_kinematic=refine_stress_exclude_kinematic,
            ),
        )
        return CriteriaConfig(refine_specs, cfg.refine_logic, copy(cfg.fracture_specs), cfg.fracture_logic)
    end

    return cfg
end

# -------------------------------------------------- #
#                   Refinement
# -------------------------------------------------- #

@inline function _check_refinement_criteria(spec::RefineSpec, con, lam_r, r)::Bool
    # Simple lambda refinement check
    if spec.kind == REFINE_LAMBDA
        # Mixed-mode traction utilization:
        # opening normal traction + both tangential shear components.
        # Compressive normal traction does not contribute.
        r == 1 || return false
        thresh_mul = spec.params[1]
        return _bond_traction_drive(con) >= thresh_mul

    elseif spec.kind == REFINE_ENERGY
        # Refine on a bond-normalized initiation energy scale.
        # eta = 1 corresponds to cohesive onset for this specific bond,
        # so the thresholds remain tied to the current model/discretization.
        r == 1 || return false

        eta = _bond_onset_measure(con)
        eta_lo = spec.params[1]
        eta_hi = spec.params[2] > eta_lo ? spec.params[2] : 1.0
        return eta >= eta_lo && eta < eta_hi

    elseif spec.kind == REFINE_STRESS
        # Stress-driven refinement is evaluated in a body pass after the
        # converged solve. Keep the bond path inactive for this criterion.
        return false
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
#             Stress-Driven Refinement
# -------------------------------------------------- #

function compute_body_stress_data(sim)
    active_body_indices = sim.active_body_ids
    id_to_local = fill(0, length(sim.bodies))

    for (local_idx, body_idx) in enumerate(active_body_indices)
        body_id = sim.bodies[body_idx].id
        id_to_local[body_id + 1] = local_idx
    end

    stress_data = zeros(FLOAT, length(active_body_indices), 6)

    for bond in sim.bond_constraints
        bond.is_broken && continue
        bond.is_active || continue

        a_idx = bond.bodyA.id + 1
        b_idx = bond.bodyB.id + 1
        if !(sim.active[a_idx] && sim.active[b_idx])
            continue
        end

        bA = bond.bodyA
        bB = bond.bodyB

        R_A = quat_to_rotmat(bA.quat)
        n = R_A * bond.n_local
        t1 = R_A * bond.t1_local
        t2 = R_A * bond.t2_local
        pA = transform_point(bond.pA_local, bA.pos, bA.quat)
        pB = transform_point(bond.pB_local, bB.pos, bB.quat)

        f_local = MVector{3,Float64}(undef)
        for r in 1:3
            k_val = bond.penalty_k[r]
            lambda_base = isinf(bond.stiffness[r]) ? bond.lambda[r] : 0.0
            f_local[r] = clamp(k_val * bond.C[r] + lambda_base, bond.f_min[r], bond.f_max[r])
        end
        F_world = n * f_local[1] + t1 * f_local[2] + t2 * f_local[3]

        rA = pA - bA.pos
        volA = bA.size[1] * bA.size[2] * bA.size[3]
        local_a = id_to_local[bA.id + 1]
        _acc_stress!(stress_data, local_a, rA, F_world, volA)

        rB = pB - bB.pos
        volB = bB.size[1] * bB.size[2] * bB.size[3]
        local_b = id_to_local[bB.id + 1]
        _acc_stress!(stress_data, local_b, rB, -F_world, volB)
    end

    return stress_data, id_to_local
end

function collect_stress_refinement_candidates(config::CriteriaConfig, sim)::Vector{Int}
    stress_specs = RefineSpec[spec for spec in config.refine_specs if spec.kind == REFINE_STRESS]
    isempty(stress_specs) && return Int[]

    stress_data, id_to_local = compute_body_stress_data(sim)
    principal_tension = zeros(Float64, length(sim.bodies))
    onset_vals = zeros(Float64, length(sim.bodies))
    candidates = Int[]
    onset_threshold = _stress_refine_onset_threshold(config)
    local_peak_ratio = 0.95

    for body_idx in sim.active_body_ids
        body = sim.bodies[body_idx]
        local_idx = id_to_local[body.id + 1]
        local_idx == 0 && continue

        principal_tension[body_idx] = _stress_max_principal_tension(stress_data, local_idx)
        onset_vals[body_idx] = _max_incident_bond_onset(sim, body_idx)
    end

    for body_idx in sim.active_body_ids
        body = sim.bodies[body_idx]
        stress_val = principal_tension[body_idx]
        onset_val = onset_vals[body_idx]
        neighbor_peak = _max_neighbor_stress(sim, body_idx, principal_tension)

        if config.refine_logic == ANY
            for spec in stress_specs
                _check_stress_refinement_criteria(
                    spec,
                    sim,
                    body_idx,
                    stress_val,
                    onset_val,
                    onset_threshold,
                    neighbor_peak,
                    local_peak_ratio,
                ) || continue
                push!(candidates, body_idx)
                break
            end
        else
            allow = true
            for spec in stress_specs
                _check_stress_refinement_criteria(
                    spec,
                    sim,
                    body_idx,
                    stress_val,
                    onset_val,
                    onset_threshold,
                    neighbor_peak,
                    local_peak_ratio,
                ) || (allow = false; break)
            end
            if allow
                push!(candidates, body_idx)
            end
        end
    end

    return candidates
end

# -------------------------------------------------- #
#                   Fracture
# -------------------------------------------------- #

@inline function _check_fracture_criteria(spec::FractureSpec, con, lam_r, r)::Bool
    # Simple lambda fracture check
    if spec.kind == FRAC_LAMBDA
        # Mixed-mode traction utilization:
        # opening normal traction + both tangential shear components.
        # Compressive normal traction does not contribute.
        r == 1 || return false
        thresh_mul = spec.params[1]
        return _bond_traction_drive(con) >= thresh_mul

    elseif spec.kind == FRAC_STRETCH
        # Fracture if stretch is 
        r == 1 || return false
        _bond_in_tension(con) || return false

        # Rest length w/ safety
        len_0 = max(abs(con.rest[1]), eps(Float64))
        s = _opening_disp(con) / len_0

        # Critical stretch derived from existing force cap + stiffness
        cap = max(con.fracture[1], eps(Float64))
        k_n = max(con.stiffness[1], eps(Float64))
        s_cap = cap / (k_n * len_0)

        thresh_mul = spec.params[1]
        return s >= thresh_mul * s_cap

    elseif spec.kind == FRAC_ENERGY
        r == 1 || return false
        # Energy-based fracture trigger uses the same bond-normalized
        # initiation measure. Once onset is reached, capped bonds enter the
        # cohesive damage law in commit_cohesive_damage!; uncapped bonds refine.
        return _bond_onset_measure(con) >= spec.params[1]
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

@inline function _acc_stress!(buf, idx, r, f, vol)
    idx <= 0 && return

    iv = 1.0 / max(vol, 1e-9)

    buf[idx, 1] += r[1] * f[1] * iv
    buf[idx, 2] += r[2] * f[2] * iv
    buf[idx, 3] += r[3] * f[3] * iv

    buf[idx, 4] += r[1] * f[2] * iv
    buf[idx, 5] += r[2] * f[3] * iv
    buf[idx, 6] += r[3] * f[1] * iv
end

@inline function _stress_tensor_magnitude(stress_data, idx)::Float64
    xx = stress_data[idx, 1]
    yy = stress_data[idx, 2]
    zz = stress_data[idx, 3]
    xy = stress_data[idx, 4]
    yz = stress_data[idx, 5]
    zx = stress_data[idx, 6]
    mag_sq = xx^2 + yy^2 + zz^2 + 2.0 * (xy^2 + yz^2 + zx^2)
    return sqrt(max(mag_sq, 0.0))
end

@inline function _stress_refine_excludes_kinematic(spec::RefineSpec)::Bool
    return spec.params[2] > 0.5
end

@inline function _check_stress_refinement_criteria(
    spec::RefineSpec,
    sim,
    body_idx::Int,
    stress_val::Float64,
    onset_val::Float64,
    onset_threshold::Float64,
    neighbor_peak::Float64,
    local_peak_ratio::Float64,
)::Bool
    spec.kind == REFINE_STRESS || return false

    if _stress_refine_excludes_kinematic(spec) && _body_in_prescribed_ring(sim, body_idx)
        return false
    end

    stress_val >= spec.params[1] || return false
    onset_val >= onset_threshold || return false
    return stress_val >= local_peak_ratio * neighbor_peak
end

@inline function _resolve_active_neighbor(sim, nb_node::Int)::Int
    nb_node < 0 && return 0
    return getfield(parentmodule(@__MODULE__), :resolve_active_neighbor)(sim, nb_node)
end

@inline function _stress_refine_onset_threshold(config::CriteriaConfig)::Float64
    eta = 0.0
    found = false
    for spec in config.refine_specs
        spec.kind == REFINE_ENERGY || continue
        eta_lo = spec.params[1]
        if !found || eta_lo < eta
            eta = eta_lo
            found = true
        end
    end
    return eta
end

@inline function _stress_max_principal_tension(stress_data, idx)::Float64
    xx = stress_data[idx, 1]
    yy = stress_data[idx, 2]
    zz = stress_data[idx, 3]
    xy = stress_data[idx, 4]
    yz = stress_data[idx, 5]
    zx = stress_data[idx, 6]

    sigma = @SMatrix [
        xx xy zx
        xy yy yz
        zx yz zz
    ]
    return max(eigmax(Symmetric(sigma)), 0.0)
end

@inline function _body_has_prescribed_motion(body)::Bool
    return body.inv_mass == 0.0
end

function _body_in_prescribed_ring(sim, body_idx::Int)::Bool
    body = sim.bodies[body_idx]
    _body_has_prescribed_motion(body) && return true

    size(sim.neighbor_map, 2) == 6 || return false
    node_id = body.id
    for dir in 1:6
        nb_node = sim.neighbor_map[node_id + 1, dir]
        nb_idx = _resolve_active_neighbor(sim, nb_node)
        nb_idx <= 0 && continue
        nb_idx == body_idx && continue
        _body_has_prescribed_motion(sim.bodies[nb_idx]) && return true
    end
    return false
end

function _max_neighbor_stress(sim, body_idx::Int, stress_vals::Vector{Float64})::Float64
    size(sim.neighbor_map, 2) == 6 || return 0.0

    peak = 0.0
    node_id = sim.bodies[body_idx].id
    for dir in 1:6
        nb_node = sim.neighbor_map[node_id + 1, dir]
        nb_idx = _resolve_active_neighbor(sim, nb_node)
        nb_idx <= 0 && continue
        nb_idx == body_idx && continue
        peak = max(peak, stress_vals[nb_idx])
    end
    return peak
end

function _max_incident_bond_onset(sim, body_idx::Int)::Float64
    peak = 0.0
    for con_id in sim.bond_incidence_all[body_idx]
        con = sim.bond_constraints[con_id]
        con.is_broken && continue
        con.is_active || continue

        a_idx = con.bodyA.id + 1
        b_idx = con.bodyB.id + 1
        (sim.active[a_idx] && sim.active[b_idx]) || continue

        peak = max(peak, _bond_onset_measure(con))
    end
    return peak
end

@inline function _normal_sign(con)::Float64
    rest_n = con.rest[1]
    return abs(rest_n) <= eps(Float64) ? 1.0 : sign(rest_n)
end

@inline function _opening_disp(con)::Float64
    return max(0.0, _normal_sign(con) * con.C[1])
end

@inline function _shear_disp(con)::Float64
    return sqrt(con.C[2]^2 + con.C[3]^2)
end

@inline function _bond_energy_components(con)::Tuple{Float64,Float64}
    dn = _opening_disp(con)
    en = 0.5 * con.stiffness[1] * dn^2
    et = 0.5 * (con.stiffness[2] * con.C[2]^2 + con.stiffness[3] * con.C[3]^2)
    return en, et
end

@inline function _bond_traction_components(con)::Tuple{Float64,Float64,Float64}
    sigma_n = clamp(con.penalty_k[1] * con.C[1], con.f_min[1], con.f_max[1])
    sigma_t1 = clamp(con.penalty_k[2] * con.C[2], con.f_min[2], con.f_max[2])
    sigma_t2 = clamp(con.penalty_k[3] * con.C[3], con.f_min[3], con.f_max[3])

    fn_open = max(0.0, _normal_sign(con) * sigma_n)
    ft1 = abs(sigma_t1)
    ft2 = abs(sigma_t2)
    return fn_open, ft1, ft2
end

@inline function _bond_traction_drive(con)::Float64
    fn_open, ft1, ft2 = _bond_traction_components(con)
    cap_n = max(con.fracture[1], eps(Float64))
    cap_t1 = max(con.fracture[2], eps(Float64))
    cap_t2 = max(con.fracture[3], eps(Float64))
    return sqrt((fn_open / cap_n)^2 + (ft1 / cap_t1)^2 + (ft2 / cap_t2)^2)
end

@inline function _bond_energy_budget(con)::Float64
    return max(0.5 * con.fracture[1] * con.limits[3], eps(Float64))
end

@inline function _bond_energy_drive(con)::Float64
    en, et = _bond_energy_components(con)
    return (en + et) / _bond_energy_budget(con)
end

@inline function _bond_onset_measure(con)::Float64
    dn = _opening_disp(con)
    ds = _shear_disp(con)
    dn0 = max(con.limits[1], eps(Float64))
    ds0 = max(con.limits[2], eps(Float64))
    return (dn / dn0)^2 + (ds / ds0)^2
end

@inline function _bond_final_measure(con)::Float64
    dn = _opening_disp(con)
    ds = _shear_disp(con)
    dnc = max(con.limits[3], eps(Float64))
    dsc = max(con.limits[4], eps(Float64))
    return (dn / dnc)^2 + (ds / dsc)^2
end

@inline function _bond_in_tension(con)::Bool
    return _opening_disp(con) > 0.0
end

function commit_cohesive_damage!(con::BondConstraint)::Tuple{Bool,Float64}
    con.is_broken && return (false, 0.0)

    function stored_energy()
        k_eff = get_effective_stiffness(con)
        return 0.5 * (k_eff[1] * con.C[1]^2 + k_eff[2] * con.C[2]^2 + k_eff[3] * con.C[3]^2)
    end

    was_broken = con.is_broken
    old_damage = con.damage
    e_pre = stored_energy()

    dn = _opening_disp(con)
    ds = _shear_disp(con)
    psi_on = _bond_onset_measure(con)
    psi_fin = _bond_final_measure(con)

    con.current_eff_strain = sqrt(psi_fin)

    if !con.is_cohesive && psi_on >= 1.0
        con.is_cohesive = true
        con.max_eff_strain = sqrt(psi_on)
    end

    if con.is_cohesive
        con.max_eff_strain = max(con.max_eff_strain, con.current_eff_strain)

        if con.current_eff_strain >= 1.0
            con.damage = 1.0
            con.is_broken = true
        else
            dnc = max(con.limits[3], eps(Float64))
            dsc = max(con.limits[4], eps(Float64))
            dn0 = max(con.limits[1], eps(Float64))
            ds0 = max(con.limits[2], eps(Float64))

            strain_cr = dn0 / dnc
            if ds > 1e-9
                denom = (dn / dn0)^2 + (ds / ds0)^2
                numer = (dn / dnc)^2 + (ds / dsc)^2
                strain_cr = denom > eps(Float64) ? sqrt(numer / denom) : dn0 / dnc
            end

            if con.max_eff_strain > strain_cr
                con.damage = clamp((con.max_eff_strain - strain_cr) / max(1.0 - strain_cr, eps(Float64)), 0.0, 1.0)
            else
                con.damage = 0.0
            end
        end
    else
        con.damage = 0.0
    end

    e_post = stored_energy()
    con.k_eff = get_effective_stiffness(con)

    changed = (con.is_broken != was_broken) || (abs(con.damage - old_damage) > 1e-12)
    released = max(e_pre - e_post, 0.0)
    return changed, released
end


end
