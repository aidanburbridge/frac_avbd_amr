module Criteria

using StaticArrays

import ...AVBDConstraints: BondConstraint, get_effective_stiffness

export CriteriaConfig, RefineSpec, FractureSpec, RefineKind, FractureKind, LogicOperation,
    REFINE_LAMBDA, REFINE_ENERGY, FRAC_LAMBDA, FRAC_STRETCH, FRAC_ENERGY, ANY, ALL,
    SimpleRefine, SimpleFracture, should_refine, should_fracture

# ---------- Enums and Logic Params ---------- #

@enum LogicOperation ANY ALL

@enum RefineKind REFINE_LAMBDA REFINE_ENERGY
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
SimpleFracture(threshold::Real=1.0) = FractureSpec(FRAC_LAMBDA, SVector(Float64(threshold), 0.0))

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
        # Refine on a bond-normalized initiation energy scale.
        # eta = 1 corresponds to cohesive onset for this specific bond,
        # so the thresholds remain tied to the current model/discretization.
        r == 1 || return false

        eta = _bond_onset_measure(con)
        eta_lo = spec.params[1]
        eta_hi = spec.params[2] > eta_lo ? spec.params[2] : 1.0
        return eta >= eta_lo && eta < eta_hi
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
