module Criteria

using StaticArrays

import ...AVBDConstraints: BondConstraint, get_effective_stiffness

export CriteriaConfig, RefineSpec, FractureSpec, RefineKind, FractureKind, LogicOperation,
       REFINE_LAMBDA, FRAC_LAMBDA, ANY, ALL,
       SimpleRefine, SimpleFracture, should_refine, should_fracture

# ---------- Enums and Logic Params ---------- #

@enum LogicOperation ANY ALL

@enum RefineKind REFINE_LAMBDA
@enum FractureKind FRAC_LAMBDA

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
        # Allow any mode (normal/shear), but only while bond is in opening tension.
        sigma_n = con.penalty_k[1] * con.C[1]
        lam_n = clamp(sigma_n, con.f_min[1], con.f_max[1])
        lam_n < 0.0 || return false
        thresh_mul = spec.params[1]
        cap = max(con.fracture[r], eps(Float64))
        return abs(lam_r) >= thresh_mul * cap
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
        # Allow any mode (normal/shear), but only while bond is in opening tension.
        sigma_n = con.penalty_k[1] * con.C[1]
        lam_n = clamp(sigma_n, con.f_min[1], con.f_max[1])
        lam_n < 0.0 || return false
        thresh_mul = spec.params[1]
        cap = max(con.fracture[r], eps(Float64))
        return abs(lam_r) >= thresh_mul * cap
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


# @kwdef struct RefineGradient <: AbstractRefineCriteria

# end

# @kwdef struct RefineResidual <: AbstractRefineCriteria

# end

# @inline function should_refine(cfg::CriteriaConfig, sim, con, lam_r, r)::Bool
#     return abs(lam_r) >= cfg.refine.force_threshold
# end

# function RefineTopPct()
#     # Refine only where the material is storing lots of strain energy - top active_constraints
#     # Using top-K or top percent so I get clusters
#     # W = []
#     # for v in voxels:
#     #     W[i] = 0
#     #     for b in bonds:

# end

# function RefineGradient()
#     # Refine where lambda gradient is steep
# end

# function RefineResidual()
#     # Refine where the discritized fails equilibrium (?)

# end

# function RefineCZM()
#     # Refine only near crack until enough cells to process crack-tip zone
#     # Small enough to resolve crack tip
#     # l_cz  = (E*G)/(stress_t^2) -> refine unitl h >= l_cz/N (N = ~3-5)
# end

# @kwdef struct SimpleFracture
#     force_threshold::Float64
# end

# struct FractureCZM <: AbstractFractureCriteria

# end


# @inline function should_fracture(cfg::CriteriaConfig, sim, con, lam_r, r)::Bool
#     return abs(lam_r) >= cfg.fracture.force_threshold
# end

# function FractureCZM()
#     # Uses mixed mode criteria

# end

# function FractureGriffith()
#     # Crack grows if 

# end

# function FractureSIF()
#     # Crack gorws when local tip field exceeds fracture toughness K_IC
#     # LEFM

# end

# function FractureMohrCoul()
#     # t_s >= c-t_n * tan(phi)
#     # for b in bonds:
#     #     t_n = 
# end

end
