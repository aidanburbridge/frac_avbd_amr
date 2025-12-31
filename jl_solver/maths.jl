module Maths

using LinearAlgebra
import LinearAlgebra: normalize
using StaticArrays

export quat_to_rotmat, quat_mul, quat_inv, quat_to_rotvec, rotvec_to_quat, delta_twist_from
export rotate_vec, transform_point, inv_transform_point, orthonormal_basis
export Vec3, Mat3, Quat, FLOAT

# Type aliases
const FLOAT = Float64
const Vec3 = SVector{3,FLOAT}
const Mat3 = SMatrix{3,3,FLOAT,9}
const Quat = SVector{4,FLOAT}

@inline function normalize(q::Quat)
    n = norm(q)
    if n == 0.0 || !isfinite(n)
        return Quat(1.0, 0.0, 0.0, 0.0)
    end
    return q / n
end


function delta_twist_from(b, from_pos::Vec3, from_quat::Quat)
    dx = b.pos - from_pos

    q_rel = quat_mul(b.quat, quat_inv(from_quat))

    d_th = @SVector [q_rel[2] * 2.0, q_rel[3] * 2.0, q_rel[4] * 2.0]

    # TODO maybe I should make this better - check python primitivies.py

    return vcat(dx, d_th)

end

# ---------- Quaternion Operations ---------- #
@inline function quat_to_rotmat(q::Quat)
    w, x, y, z = normalize(q)
    xx, yy, zz = x^2, y^2, z^2
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return Mat3(
        1 - 2(yy + zz), 2(xy + wz), 2(xz - wy),
        2(xy - wz), 1 - 2(xx + zz), 2(yz + wx),
        2(xz + wy), 2(yz - wx), 1 - 2(xx + yy)
    )
end

@inline function quat_mul(q1::Quat, q2::Quat)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return Quat(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    )
end

@inline function quat_inv(q::Quat)
    return Quat(q[1], -q[2], -q[3], -q[4])
end

@inline function rotvec_to_quat(v::Vec3)
    angle = norm(v)
    if angle < 1e-8
        return Quat(1.0, 0.0, 0.0, 0.0)
    end
    axis = v / angle
    half_angle = angle * 0.5
    s = sin(half_angle)
    c = cos(half_angle)
    return Quat(c, axis[1] * s, axis[2] * s, axis[3] * s)
end

@inline function integrate_quat(q::Quat, ang_vel::Vec3, dt)
    dq = Quat(0.0, ang_vel[1], ang_vel[2], ang_vel[3])
    return normalize(q + quat_mul(dq, q) * (0.5 * dt))
end


# ---------- Orthonormal basis helper ----------
@inline function orthonormal_basis(n::Vec3)
    # Stable Gram-Schmidt to produce two tangents orthogonal to n.
    if abs(n[1]) < 0.9
        a = Vec3(1, 0, 0)
    else
        a = Vec3(0, 1, 0)
    end
    t1 = normalize(cross(n, a))
    t2 = normalize(cross(n, t1))
    return t1, t2
end

@inline function quat_to_rotvec(q::Quat)
    # Map a quaternion to its axis-angle vector (Rodrigues vector).
    # Assumes q = (w, x, y, z).
    qn = normalize(q)
    w = clamp(qn[1], -1.0, 1.0)
    angle = 2.0 * acos(w)
    s = sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8 || angle < 1e-8
        return Vec3(0.0, 0.0, 0.0)
    end
    axis = Vec3(qn[2], qn[3], qn[4]) / s
    return axis * angle
end

# ---------- Vector Operations ---------- #
@inline function rotate_vec(v::Vec3, q::Quat)
    # Optimized rotation v' = q * v * q_inv
    # Standard formula: v + 2 * cross(q_xyz, cross(q_xyz, v) + q_w * v)
    u = Vec3(q[2], q[3], q[4])
    s = q[1]
    return v + 2.0 * cross(u, cross(u, v) + s * v)
end

@inline function transform_point(p::Vec3, pos::Vec3, rot::Quat)
    return pos + rotate_vec(p, rot)
end

@inline function inv_transform_point(p_world::Vec3, pos::Vec3, rot::Quat)
    # Transform world point back to local space
    diff = p_world - pos
    return rotate_vec(diff, quat_inv(rot))
end

end # module end
