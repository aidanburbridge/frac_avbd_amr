# manifold.jl
module ManifoldHandling

using LinearAlgebra
using StaticArrays

# Imports from your other modules
# Assuming Body is in Collisions and ContactConstraint is in Constraints
using ..Collisions: Body, Contact, get_aabb
using ..Constraints: ContactConstraint

export Manifold, update_manifold!, is_empty

# --- Constants ---
const MAX_SEP_FRAMES = 3

# --- Helper Functions (Private) ---

"""
Generates a spatial key for a contact based on its local position and normal.
Matches Python: _key_from_contact / _contact_key
"""
function _contact_key(point::SVector{3,Float64}, normal::SVector{3,Float64}; p_dec::Int=3, n_dec::Int=2)
    # Quantize point and normal for hash stability
    p_key = round.(point, digits=p_dec)
    n_key = round.(normal, digits=n_dec)
    # We return a tuple that Julia can hash automatically
    return (p_key, n_key)
end

function _proximity_test(c::Contact, old_con::ContactConstraint; dist_eps=2e-3, cos_eps=0.985)
    # Check distance squared
    dist_sq = sum(abs2, c.point - old_con.point)
    if dist_sq > dist_eps^2
        return false
    end

    # Check normal alignment
    if dot(c.normal, old_con.normal) < cos_eps
        return false
    end

    return true
end

function _aabb_overlap(A::Body, B::Body)
    # Conservative test using AABBs
    aabbA = get_aabb(A)
    aabbB = get_aabb(B)

    # Check for separation on any axis
    if aabbA.max[1] < aabbB.min[1] || aabbB.max[1] < aabbA.min[1]
        return false
    end
    if aabbA.max[2] < aabbB.min[2] || aabbB.max[2] < aabbA.min[2]
        return false
    end
    if aabbA.max[3] < aabbB.min[3] || aabbB.max[3] < aabbA.min[3]
        return false
    end

    return true
end

# --- Manifold Struct ---

mutable struct Manifold
    bodyA::Body
    bodyB::Body

    # Stores constraints mapped by spatial key
    # Dict{ KeyTuple, ContactConstraint }
    constraint_dict::Dict{Any,ContactConstraint}

    # Stores age of separation for keys
    # Dict{ KeyTuple, Int }
    separation_frames::Dict{Any,Int}

    # The active list exposed to the solver
    constraints::Vector{ContactConstraint}

    function Manifold(A::Body, B::Body)
        new(A, B,
            Dict{Any,ContactConstraint}(),
            Dict{Any,Int}(),
            ContactConstraint[])
    end
end

# --- Main Logic ---

"""
Updates the manifold with new contacts, handling warm-starting and persistence.
Matches Python: Manifold.update_from_contacts
"""
function update_manifold!(manifold::Manifold, new_contacts::Vector{Contact}, friction::Float64; dist_eps=0.1, cos_eps=0.95)
    old_dict = manifold.constraint_dict

    new_dict = Dict{Any,ContactConstraint}()
    used_old_keys = Set{Any}()

    # 1. Match New Contacts to Old
    for c in new_contacts
        # Create fresh constraint (stateless)
        con = ContactConstraint(c, friction)

        matched_key = nothing

        # Try to find a match in old constraints
        # Note: Ideally we use spatial hashing lookup, but for exact matching "proximity" logic
        # we might iterate if the exact key drifted. 
        # The Python script iterates `old.items()` to check proximity if exact key logic isn't strict.
        # Here we follow Python's O(N*M) loop inside _proximity_test logic for robustness.

        for (k_old, con_old) in old_dict
            if k_old in used_old_keys
                continue
            end

            if _proximity_test(c, con_old, dist_eps=dist_eps, cos_eps=cos_eps)
                matched_key = k_old

                # Warm Start: Copy lambda and penalty state
                con.lambda_n = con_old.lambda_n
                con.lambda_t1 = con_old.lambda_t1
                con.lambda_t2 = con_old.lambda_t2
                # If you have penalty_k, copy that too

                push!(used_old_keys, k_old)

                # Reset separation counter
                manifold.separation_frames[k_old] = 0
                break
            end
        end

        # Determine Key
        if matched_key !== nothing
            key = matched_key
        else
            # Generate new key
            key = _contact_key(c.point, c.normal)
        end

        new_dict[key] = con

        # Init separation counter if new
        if !haskey(manifold.separation_frames, key)
            manifold.separation_frames[key] = 0
        end
    end

    # 2. Persistence (Keep stale contacts)
    overlap = _aabb_overlap(manifold.bodyA, manifold.bodyB)

    for (k_old, con_old) in old_dict
        if k_old in used_old_keys
            continue
        end

        # Increment age
        sep_age = get(manifold.separation_frames, k_old, 0) + 1

        # If bodies clearly separate, drop immediately (optimization)
        if !overlap
            sep_age = MAX_SEP_FRAMES + 1
        end

        if sep_age <= MAX_SEP_FRAMES
            # Keep the old constraint alive
            new_dict[k_old] = con_old
            manifold.separation_frames[k_old] = sep_age
        else
            # Too old, let it expire (remove from separation tracking)
            delete!(manifold.separation_frames, k_old)
        end
    end

    # 3. Update State
    manifold.constraint_dict = new_dict
    manifold.constraints = collect(values(new_dict))
end

function is_empty(m::Manifold)
    return isempty(m.constraint_dict)
end

end # module