module Criteria

using StaticArrays

import ...AVBDConstraints: BondConstraint, get_effective_stiffness

export CriteriaConfig, RefineSpec, FractureSpec, RefineKind, FractureKind, LogicOperation,
    REFINE_LAMBDA, REFINE_ENERGY, FRAC_LAMBDA, FRAC_STRETCH, ANY, ALL,
    SimpleRefine, SimpleFracture, should_refine, should_fracture

# ---------- Enums and Logic Params ---------- #

@enum LogicOperation ANY ALL

@enum RefineKind REFINE_LAMBDA REFINE_ENERGY
@enum FractureKind FRAC_LAMBDA FRAC_STRETCH

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


end
