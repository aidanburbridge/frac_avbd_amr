"""
Energy accounting utilities for the AVBD solver.

This module tracks kinetic, elastic, contact, fracture, and viscous work terms
so the Julia solver and the export bridge can write reproducible energy logs.
"""
module Energy

using LinearAlgebra
using StaticArrays
using ..Maths: rotate_vec, quat_to_rotmat
using ..Collisions: Body
using ..AVBDConstraints: BondConstraint, ContactConstraint, compute_constraint!, get_effective_stiffness

export EnergyLog, log_step!, record_fracture_work!, record_viscous_work!, preview_accounted_energy

mutable struct EnergyLog
    kinetic::Vector{Float64}
    bond_potential::Vector{Float64}
    contact_potential::Vector{Float64}
    fracture_work::Vector{Float64}
    viscous_work::Vector{Float64}
    mech_energy::Vector{Float64}
    accounted_energy::Vector{Float64}

    accumulated_fracture_work::Float64
    accumulated_viscous_work::Float64

    function EnergyLog()
        new(Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], 0.0, 0.0)
    end
end

function compute_kinetic(bodies::Vector{Body})
    T = 0.0
    for b in bodies
        # Skip static bodies
        if b.is_static || b.inv_mass == 0.0
            continue
        end

        # Translational
        v_sq = dot(b.vel, b.vel)
        T += 0.5 * b.mass * v_sq

        # Rotational
        T += 0.5 * dot(b.ang_vel, b.inertia_diag .* b.ang_vel)
    end
    return T
end

function compute_kinetic(bodies::Vector{Body}, active_ids::Vector{Int})
    T = 0.0
    for idx in active_ids
        b = bodies[idx]
        # Skip static bodies
        if b.is_static || b.inv_mass == 0.0
            continue
        end

        # Translational
        v_sq = dot(b.vel, b.vel)
        T += 0.5 * b.mass * v_sq

        # Rotational
        T += 0.5 * dot(b.ang_vel, b.inertia_diag .* b.ang_vel)
    end
    return T
end

function compute_bond_potential(bonds::Vector{BondConstraint})
    U = 0.0
    for bond in bonds
        if bond.is_broken
            continue
        end

        rA = rotate_vec(bond.pA_local, bond.bodyA.quat)
        rB = rotate_vec(bond.pB_local, bond.bodyB.quat)
        dp = (bond.bodyA.pos + rA) - (bond.bodyB.pos + rB)

        rotA = quat_to_rotmat(bond.bodyA.quat)
        c_n = dot(rotA * bond.n_local, dp) - bond.rest[1]
        c_t1 = dot(rotA * bond.t1_local, dp) - bond.rest[2]
        c_t2 = dot(rotA * bond.t2_local, dp) - bond.rest[3]

        C_local = @SVector [c_n, c_t1, c_t2]

        k_eff = get_effective_stiffness(bond)

        U += 0.5 * (k_eff[1] * C_local[1]^2 + k_eff[2] * C_local[2]^2 + k_eff[3] * C_local[3]^2)
    end
    return U
end

function compute_bond_potential(bonds::Vector{BondConstraint}, active_bond_ids::Vector{Int})
    U = 0.0
    for con_id in active_bond_ids
        bond = bonds[con_id]
        if bond.is_broken
            continue
        end

        rA = rotate_vec(bond.pA_local, bond.bodyA.quat)
        rB = rotate_vec(bond.pB_local, bond.bodyB.quat)
        dp = (bond.bodyA.pos + rA) - (bond.bodyB.pos + rB)

        rotA = quat_to_rotmat(bond.bodyA.quat)
        c_n = dot(rotA * bond.n_local, dp) - bond.rest[1]
        c_t1 = dot(rotA * bond.t1_local, dp) - bond.rest[2]
        c_t2 = dot(rotA * bond.t2_local, dp) - bond.rest[3]

        C_local = @SVector [c_n, c_t1, c_t2]

        k_eff = get_effective_stiffness(bond)

        U += 0.5 * (k_eff[1] * C_local[1]^2 + k_eff[2] * C_local[2]^2 + k_eff[3] * C_local[3]^2)
    end
    return U
end

function compute_contact_potential(contacts::Vector{ContactConstraint}, alpha::Float64)
    U = 0.0
    for con in contacts
        compute_constraint!(con, alpha)
        for i in 1:3
            term = 0.5 * con.penalty_k[i] * con.C[i]^2
            U += term
        end
    end
    return U
end

function record_fracture_work!(energy_log::EnergyLog, work::Float64)
    energy_log.accumulated_fracture_work += work
end

function record_viscous_work!(energy_log::EnergyLog, work::Float64)
    if isfinite(work) && work > 0.0
        energy_log.accumulated_viscous_work += work
    end
end

"""Preview the accounted energy using the current accumulated fracture and viscous work."""
function preview_accounted_energy(energy_log::EnergyLog, bodies::Vector{Body}, bonds::Vector{BondConstraint}, contacts::Vector{ContactConstraint},
    alpha::Float64, active_body_ids::Vector{Int}, active_bond_ids::Vector{Int})
    T = compute_kinetic(bodies, active_body_ids)
    U_bond = compute_bond_potential(bonds, active_bond_ids)
    U_contact = compute_contact_potential(contacts, alpha)
    W_frac = energy_log.accumulated_fracture_work
    W_visc = energy_log.accumulated_viscous_work
    E_mech = T + U_bond + U_contact
    return E_mech + W_frac + W_visc
end

"""Append one global energy sample using all bodies and bonds."""
function log_step!(energy_log::EnergyLog, bodies::Vector{Body}, bonds::Vector{BondConstraint}, contacts::Vector{ContactConstraint}, alpha::Float64)

    T = compute_kinetic(bodies)
    U_bond = compute_bond_potential(bonds)
    U_contact = compute_contact_potential(contacts, alpha)
    W_frac = energy_log.accumulated_fracture_work
    W_visc = energy_log.accumulated_viscous_work

    E_mech = T + U_bond + U_contact
    E_accounted = E_mech + W_frac + W_visc

    push!(energy_log.kinetic, T)
    push!(energy_log.bond_potential, U_bond)
    push!(energy_log.contact_potential, U_contact)
    push!(energy_log.fracture_work, W_frac)
    push!(energy_log.viscous_work, W_visc)
    push!(energy_log.mech_energy, E_mech)
    push!(energy_log.accounted_energy, E_accounted)

end

"""Append one energy sample restricted to the currently active bodies and bonds."""
function log_step!(energy_log::EnergyLog, bodies::Vector{Body}, bonds::Vector{BondConstraint}, contacts::Vector{ContactConstraint},
    alpha::Float64, active_body_ids::Vector{Int}, active_bond_ids::Vector{Int})

    T = compute_kinetic(bodies, active_body_ids)
    U_bond = compute_bond_potential(bonds, active_bond_ids)
    U_contact = compute_contact_potential(contacts, alpha)
    W_frac = energy_log.accumulated_fracture_work
    W_visc = energy_log.accumulated_viscous_work

    E_mech = T + U_bond + U_contact
    E_accounted = E_mech + W_frac + W_visc

    push!(energy_log.kinetic, T)
    push!(energy_log.bond_potential, U_bond)
    push!(energy_log.contact_potential, U_contact)
    push!(energy_log.fracture_work, W_frac)
    push!(energy_log.viscous_work, W_visc)
    push!(energy_log.mech_energy, E_mech)
    push!(energy_log.accounted_energy, E_accounted)
end

end # module
