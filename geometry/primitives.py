# PRIMITIVES
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union

# -------------------- Data Structures -------------------- #
@dataclass
class AABB_ND:
    """ An Axis Aligned Bounding Box (AABB) in D dimensions. """
    mins: np.ndarray # shape (D,)
    maxs: np.ndarray # shape (D,)

    @property
    def dim(self) -> int:
        """ Number of dimensions for this box. """
        return int(self.mins.shape[0])
    
@dataclass
class AABB2:
    """ 2D AABB. """
    min_x: float; max_x: float
    min_y: float; max_y: float

@dataclass
class AABB3:
    """ 3D AABB. """
    min_x: float; max_x: float
    min_y: float; max_y: float
    min_z: float; max_z: float

# -------------------- Linear Algebra Helper Functions -------------------- #

def skew(v: np.ndarray) -> np.ndarray:
    """
    Turn a 3D vector into a "cross-product matrix".
    Ex: skew(v) @ w = v x w
    """
    x, y, z = np.asarray(v, dtype=float)
    return np.array([
        [0 , -z, y],
        [ z, 0, -x],
        [-y, x, 0]
        ], dtype=float)


# -------------------- Quaternion Helper (3D) -------------------- #
"""
Quaternion being a way to represent rotation in 3D space - with 4 numbers hence quat.
    q = (w,x,y,z)
    Where:
        - w: scalar part 
        - (x, y, x): vector part, uses imaginary numbers
"""

def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize the quaternion, aka make it have unit length.
    This allows for "safe" rotations, so it does not promote or propagate skewness from compounding floating point number rounding.
    """
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q if n == 0 else (q/n)

def quat_mult(q1: np.ndarray, q2: np.array) -> np.ndarray:
    """
    Combine two rotations into one quaternion.
    Ex: Combining a raw rotation with a pitch rotation in one quaternion.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Make a quaternion that rotates by a set angle around a given axis.
    """
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    half = 0.5 * float(angle)
    s = np.sin(half) / n
    return quat_normalize(np.array([np.cos(half), axis[0]*s, axis[1]*s, axis[2]*s], dtype=float))

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Turns a quaternion into a normal 3x3 matrix so vectors and corners can be rotated.
    """
    w, x, y, z = quat_normalize(q)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy+zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx+zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),   1 - 2*(xx+yy)]
    ], dtype=float)

def quat_from_angular_velocity(omega: np.ndarray, dt: float) -> np.ndarray:
    """
    Convert an angular velocity to a quaternion rotation over a set timestep.
    """
    w = np.asarray(omega, dtype=float)
    theta = np.linalg.norm(w) * float(dt)
    if theta == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = w / (np.linalg.norm(w) + 1e-12)
    return quat_from_axis_angle(axis, theta)


# -------------------- Body base classes -------------------- #

class Body:
    def __init__(self, position, velocity, density:float, stiffness:float, static:bool=False):

        # Kinematics
        self.position = np.array(position, dtype=float).copy() # 2D:[x, y, theta]
        self.velocity = np.array(velocity, dtype=float).copy() # 2D:[vx, vy, omega]
        self.static = bool(static)

        # Material
        self.density = float(density)
        self.k = float(stiffness)

        # DOFs 
        self.dof = len(self.position)

        # Mass/inertia defaults (also for statics)
        self.mass = float('inf')
        self.inv_mass = 0.0
        self.inertia: Union[float, np.ndarray] = float('inf')
        self.inv_inertia: Union[float, np.ndarray] = 0.0

        # Solver related 
        self.initial_pos = self.position.copy()
        self.prev_vel = self.velocity.copy()

        # Inertial or "y" in paper
        self.inertial_pos = np.zeros_like(self.position)


class CollidableShape(Body, ABC):
    """
    Abstract base class for any Body that has collisions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_corners(self) -> np.array:
        """ Must return a list of world-space corner vertices. """
        pass
    
    @abstractmethod
    def get_axes(self) -> np.array:
        """ Must return a list of world-space normal axes for SAT. """
        pass

    @abstractmethod
    def get_aabb(self) -> Union[AABB2, AABB3, AABB_ND]:
        """ Must return a list of world-space Axis-Aligned Bounding Box. """
        pass

# -------------------- 2D rectangle -------------------- #
    
class rect_2D(CollidableShape):
    def __init__(self, position, velocity, density:float, stiffness:float, size, static: bool=False):
        super().__init__(position, velocity, density, stiffness, static)

        self.size = np.array(size, dtype=float)
        width, height = self.size
        area = float(width * height)

        if not self.static:
            self.mass = area * self.density
            self.inv_mass = 1.0 / self.mass if self.mass >  0 else 0
            I = self.mass * (width**2 + height**2) / 12.0
            self.inertia = I
            self.inv_inertia = 1.0 / I if I > 0.0 else 0.0
        else:
            self.mass = float('inf')
            self.inv_mass = 0.0
            self.inertia = float('inf')
            self.inv_inertia = 0.0
    
    def get_corners(self) -> np.array:
        """
        Returns the world coordinates of the rectangle corners.
        """

        width, height = self.size
        _, _, theta = self.position

        c, s = np.cos(theta), np.sin(theta)

        # Form rotation matrix for 2D world
        rot_matrix = np.array([[c, -s], [s, c]], dtype=float)

        # Define the "half-extents" like center to edge
        half_extents = np.array([width/2, height/2], dtype=float)

        # Define the local corner coordinates
        corner_pattern = np.array([
            [1,1], #top right
            [-1,1], #top left
            [-1,-1], #bottom left
            [1,-1] #bottom right
            ]) 
        
        corners_local = half_extents * corner_pattern

        # Rotate the local corners 
        # @ is matrix multiplication
        # so we rotate the corners to the world and translate them to the rectangles global position
        world_corners = (rot_matrix @ corners_local.T).T + self.position[:2]
        return world_corners # (4,2)
    
    def get_aabb(self) -> AABB2:
        """
        Computes and returns the Axis-Aligned Bounding Box for the body.
        Aka the min and max of x and y over the 4 corners aligned with world axis.
        """

        # Get the 2D rectangle corners
        corners = self.get_corners()

        # Form the AABB
        min_x = float(np.min(corners[:, 0]))
        max_x = float(np.max(corners[:, 0]))
        min_y = float(np.min(corners[:, 1]))
        max_y = float(np.max(corners[:, 1]))

        return AABB2(min_x, max_x, min_y, max_y)
    
    def get_axes(self) -> np.array:
        """
        Returns unit vectors of body rotation, relative to world space.
        Needed for the Separating Axis Test (SAT).
        """

        _, _, theta = self.position
        c, s = np.cos(theta), np.sin(theta)

        x_axis = np.array([c, s], dtype=float)
        y_axis = np.array([-s, c], dtype=float)
        return np.stack([x_axis, y_axis], axis=0)
    
# -------------------- 3D box -------------------- #

class box_3D(CollidableShape):
    """
    3D rigid body with center position, quaternion orientation, linear and angular velocities.
    """
    def __init__(
            self,
            pos: tuple[float, float, float],
            quat: tuple[float, float, float, float],
            linear_vel: tuple[float, float, float],
            ang_vel: tuple[float, float, float],
            density: float,
            penalty_gain: float,
            size: tuple[float, float, float],
            static: bool = False
            ):
        # Combine spatial position with quaternion orientation
        packed_pos = np.concatenate([np.asarray(pos, float), np.asarray(quat, float)])
        packed_vel = np.concatenate([np.asarray(linear_vel, float), np.asarray(ang_vel, float)])
        super().__init__(packed_pos, packed_vel, density, penalty_gain, static)

        # Unpack for convience 
        self.pos        = np.asarray(pos, dtype=float).copy()    # Position of the center of the box
        self.quat       = quat_normalize(np.asarray(quat, dtype=float)).copy()
        self.linear_vel = np.asarray(linear_vel, dtype=float).copy()
        self.ang_vel    = np.asarray(ang_vel, dtype=float).copy()

        # Geometry
        self.size = np.asarray(size, dtype=float)  # (w,h,d)
        w, h, d   = self.size
        volume    = float(w * h * d)

        # Mass properties
        if not self.static:
            self.mass = volume * self.density
            self.inv_mass = 0.0 if self.mass <= 0 else 1.0 / self.mass
            Ixx = (1.0/12.0) * self.mass * (h*h + d*d)
            Iyy = (1.0/12.0) * self.mass * (w*w + d*d)
            Izz = (1.0/12.0) * self.mass * (w*w + h*h)
            self.inertia = np.diag([Ixx, Iyy, Izz])
            self.inv_inertia = np.diag([
                0.0 if Ixx <= 0 else 1.0/Ixx,
                0.0 if Iyy <= 0 else 1.0/Iyy,
                0.0 if Izz <= 0 else 1.0/Izz
            ])
        else:
            self.mass = float('inf')
            self.inv_mass = 0.0
            self.inertia = np.diag([np.inf, np.inf, np.inf])
            self.inv_inertia = np.zeros((3,3), dtype=float)
        
        # Match the packed layout used by the solver
        self._sync_packed()
    
    def rotmat(self) -> np.ndarray:
        """ Return 3x3 rotation matrix from the box's quaternion. """
        return quat_to_rotmat(self.quat)
    
    def world_axes(self) -> np.ndarray:
        """ World/global space principal axes of the box. """
        return self.rotmat()
    
    def get_axes(self) -> np.ndarray:
        """ Returns the rotmat in rows/axes for the SAT function reusability. """
        return self.rotmat().T
    
    def get_corners(self) -> np.ndarray:
        """ Compute and return the 8 corners via half-extents + cube center. """

        w, h, d = self.size
        half_extents = np.array([w/2, h/2, d/2], dtype=float)
        corner_pattern = np.array([
            [ 1, 1, 1],
            [-1, 1, 1],
            [-1,-1, 1],
            [ 1,-1, 1],
            [ 1, 1,-1],
            [-1, 1,-1],
            [-1,-1,-1],
            [ 1,-1,-1]
        ], dtype=float)

        corners_local = half_extents * corner_pattern
        rot_matrix = self.rotmat()

        world_corners = (rot_matrix @ corners_local.T).T + self.pos

        return world_corners
    
    def get_aabb(self) -> AABB3:
        """ Computes and returns the Axis-Aligned Bounding Box for the 3D body. """

        corners = self.get_corners()
        return AABB3(float(np.min(corners[:,0])), float(np.max(corners[:,0])),
                     float(np.min(corners[:,1])), float(np.max(corners[:,1])),
                     float(np.min(corners[:,2])), float(np.max(corners[:,2])))
    
    # -------------------- Integration & packing helpers -------------------- #

    def integrate_orientation(self, dt: float):
        """ First-order quaternion update using current angular velocity and dt. """

        dq = quat_from_angular_velocity(self.ang_vel, dt)
        self.quat = quat_normalize(quat_mult(dq, self.quat))
        self._sync_packed()

    def _sync_packed(self):
        """ 
        Synchronize the 7 position variables. 
        Basically make sure the packed position agrees with/stores the correct pose (translation + rotation).
        Ensures after computation involving position/quaternion the packed array is updated.
        Also refresh the packed velocity to reflect correct linear velocity and angular velocity.
        
        position = [x, y, z, q_w, q_x, q_y, q_z]
            - Global spatial: [x,y,z]
            - Rotational quaternion: [q_w, q_x, q_y, q_z]
        velocity = [vx, vy, vz, wx, wy, wz]
            - Linear velocity: [vx, vy, vz]
            - Angular velocty: [wx, wy, wz]
        """
        self.position[:3] = self.pos
        if self.position.shape[0] >= 7:
            self.position[3:7] = self.quat
        else:
            self.position = np.concatenate([self.pos, self.quat], axis=0)
        self.velocity[:3] = self.linear_vel
        if self.velocity.shape[0] >= 6:
            self.velocity[3:6] = self.ang_vel
        else:
            self.velocity = np.concatenate([self.linear_vel, self.ang_vel], axis=0)


# -------------------- Utility: 3D faces -------------------- #

def box_face_vectors(b: box_3D) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    For a 3D box return...
        - center: center position of the face
        - face_vectors: the normal and tangent vectors of each face
    """
    # Box's 3x3 rotation matrix
    rot_mat = b.rotmat()

    # Unit vectors in global space coordinates that lie along the box's local x, y, z axes
    ex = rot_mat[:,0] # local x-axis unit vector in global coords
    ey = rot_mat[:,1] # local y-axis unit vector in global coords
    ez = rot_mat[:,2] # local z-axis unit vector in global coords

    # Half extents / how far each face is from the center of the box
    hx, hy, hz = 0.5 * b.size

    # Face centers = box center + (axis direction x half-extent)
    centers = [
        b.pos +  hx*ex,  # +x
        b.pos + -hx*ex,  # -x
        b.pos +  hy*ey,  # +y
        b.pos + -hy*ey,  # -y
        b.pos +  hz*ez,  # +z
        b.pos + -hz*ez,  # -z
    ]

    # Frame / face vectors 
    # column description: [normal, tangent in y, tangent in z]
    face_vectors = [
        np.column_stack([ ex,  ey,  ez]),  # +x
        np.column_stack([-ex,  ey,  ez]),  # -x
        np.column_stack([ ey,  ez,  ex]),  # +y
        np.column_stack([-ey,  ez,  ex]),  # -y
        np.column_stack([ ez,  ex,  ey]),  # +z
        np.column_stack([-ez,  ex,  ey]),  # -z
    ]
    return [(centers[i], face_vectors[i]) for i in range(6)]

# -------------------- ND Scaffolding (unused for now) -------------------- #

class RigidND_Scaffold(Body):
    """
    Plain: Minimal ND rigid placeholder (pos in R^D, linear velocity); orientation is identity.
    Scientific: Prototype for SE(D): stores translation and an identity rotation matrix R∈SO(D).
    """
    def __init__(self, D: int, position, velocity, density: float, stiffness: float, size, static: bool=False):
        pos = np.asarray(position, float).reshape(D)
        vel = np.asarray(velocity, float).reshape(D)
        super().__init__(pos, vel, density, stiffness, static)
        self.D = int(D)
        self.R = np.eye(self.D, dtype=float)              # placeholder orientation
        self.size = np.asarray(size, dtype=float).reshape(self.D)

        volume = float(np.prod(self.size))
        if not self.static:
            self.mass = volume * self.density
            self.inv_mass = 0.0 if self.mass <= 0 else 1.0 / self.mass
        else:
            self.mass = float('inf')
            self.inv_mass = 0.0

    def get_axes(self) -> np.ndarray:
        """
        Plain: Rows of the body’s ND rotation (identity here).
        Scientific: Returns R^T; basis of the body frame in world coordinates.
        """
        return self.R.T

    def get_corners(self) -> np.ndarray:
        """
        Plain: All 2^D corners of the hyper-rectangle, rotated by R and translated by pos.
        Scientific: V_world = R V_local + t with V_local ∈ {±s_1/2}×…×{±s_D/2}.
        """
        he = 0.5 * self.size
        corners = []
        for i in range(1 << self.D):
            s = np.array([(1 if (i >> k) & 1 else -1) for k in range(self.D)], dtype=float)
            corners.append((self.R @ (he * s)) + self.position[:self.D])
        return np.stack(corners, axis=0)

    def get_aabb(self) -> AABB_ND:
        """
        Plain: ND axis-aligned bounding box around the corners.
        Scientific: Component-wise extrema over V_world yield Π_i [min, max].
        """
        c = self.get_corners()
        return AABB_ND(np.min(c, axis=0), np.max(c, axis=0))