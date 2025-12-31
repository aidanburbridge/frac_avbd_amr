# manifold.jl
module ManifoldHandling

using LinearAlgebra
using StaticArrays

using ..Collisions: Contact
using ..Constraints: ContactConstraint, reset_contact!

export Manifold, update_manifold!, init_manifold

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
    buf = Vector{ContactConstraint}(undef, MAX_CONTACTS)
    for i in 1:MAX_CONTACTS
        buf[i] = ContactConstraint()
    end
    return Manifold(a, b, buf, 0)
end

@inline function proximity(c::Contact, old_con::ContactConstraint)
    dist_sq = sum(abs2, c.point - old_con.point)
    if dist_sq > DIST_EPS^2
        return false
    end
    return dot(c.normal, old_con.normal) > COS_EPS
end

function update_manifold!(manifold::Manifold, contacts::Vector{Contact}, start_idx::Int, count::Int,
    friction::Float64, active_buffer::Vector{ContactConstraint}, active_offset::Int)
    old_count = manifold.count
    used = MVector{MAX_CONTACTS,Bool}(false, false, false, false)

    new_count = 0
    out_idx = active_offset

    for i in 0:(count - 1)
        c = contacts[start_idx + i]
        match_idx = 0

        for j in 1:old_count
            if used[j]
                continue
            end
            old = manifold.constraints[j]
            if c.feature_id == old.feature_id || proximity(c, old)
                match_idx = j
                used[j] = true
                break
            end
        end

        new_count += 1
        target = manifold.constraints[new_count]

        if match_idx == 0
            target.lambda_n = 0.0
            target.lambda_t1 = 0.0
            target.lambda_t2 = 0.0
        else
            matched = manifold.constraints[match_idx]
            target.lambda_n = matched.lambda_n
            target.lambda_t1 = matched.lambda_t1
            target.lambda_t2 = matched.lambda_t2
        end

        reset_contact!(target, c, friction)
        out_idx += 1
        active_buffer[out_idx] = target
    end

    manifold.count = new_count
    return out_idx
end

end # module
