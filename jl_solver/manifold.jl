# manifold.jl
module ManifoldHandling

using LinearAlgebra
using StaticArrays

using ..Collisions: Contact, Body
using ..AVBDConstraints: ContactConstraint

export Manifold, init_manifold, update_manifold_dynamic!

const MAX_CONTACTS = 4
const DIST_EPS = 2e-3
const COS_EPS = 0.985

mutable struct Manifold
    bodyA::Int
    bodyB::Int
    constraints::Vector{ContactConstraint}
    count::Int
end

function init_manifold(a::Int, b::Int)
    return Manifold(a, b, ContactConstraint[], 0)
end

@inline function proximity(c::Contact, old_con::ContactConstraint)
    dist_sq = sum(abs2, c.point - old_con.point)
    if dist_sq > DIST_EPS^2
        return false
    end
    return dot(c.normal, old_con.normal) > COS_EPS
end

function update_manifold_dynamic!(manifold::Manifold, contacts::Vector{Contact}, friction::Float64, bodies::Vector{Body})
    used_old = MVector{MAX_CONTACTS,Bool}(false, false, false, false)

    contact_count = min(length(contacts), MAX_CONTACTS)
    new_constraints = Vector{ContactConstraint}(undef, contact_count)
    new_count = 0

    old_count = min(manifold.count, MAX_CONTACTS)
    for i in 1:contact_count
        c = contacts[i]
        new_count += 1

        # Find match in previous frame's manifold data
        match_idx = 0
        for j in 1:old_count
            if !used_old[j]
                old_con = manifold.constraints[j]
                if c.feature_id == old_con.feature_id || proximity(c, old_con)
                    match_idx = j
                    used_old[j] = true
                    break
                end
            end
        end

        # Create new constraint
        bA = bodies[c.body_idx_a + 1]
        bB = bodies[c.body_idx_b + 1]

        con = ContactConstraint(bA, bB, c.point, c.normal, c.depth, friction; feature_id=c.feature_id)

        # Copy old values over
        if match_idx != 0
            old = manifold.constraints[match_idx]
            con.lambda = old.lambda
            con.penalty_k = old.penalty_k
        end

        new_constraints[new_count] = con
    end

    resize!(manifold.constraints, new_count)

    for i in 1:new_count
        manifold.constraints[i] = new_constraints[i]
    end

    manifold.count = new_count
end

end # module
