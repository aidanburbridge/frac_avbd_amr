# BODIES
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class AABB:
    """
    The Axis_Aligned Bounding Box data structure for a Body.
    """
    min_x: float
    max_x: float
    min_y: float
    max_y: float

# START 2D squares

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
        self.mass = np.inf
        self.inv_mass = 0
        self.inertia = np.inf
        self.inv_inertia = 0

        # Solver related 
        self.initial_pos = self.position.copy()
        self.prev_vel = self.velocity.copy()

        # Inertial or "y" in paper
        self.inertial_pos = np.zeros_like(self.position)


class CollidableShape(Body, ABC):
    """
    Absract base class for any Body that has collisions.
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
    def get_aabb(self) -> AABB:
        """ Must return a list of world-space Axis-Aligned Bounding Box. """
        pass
    
    @abstractmethod
    def get_dim(self) -> int:
        pass 

class rect_2D(CollidableShape):
    def __init__(self, position, velocity, density:float, stiffness:float, size, static=False):
        super().__init__(position, velocity, density, stiffness, static)

        self.size = np.array(size, dtype=float)
        width, height = self.size
        area = width * height

        if not self.static:
            self.mass = area * self.density
            self.inv_mass = 1.0 / self.mass if self.mass >  0 else 0
            self.inertia = self.mass * (width**2 + height**2) / 12.0
            self.inv_inertia = 1.0 / self.inertia if self.inertia > 0 else 0
    
    def get_corners(self) -> np.array:
        """
        Returns the world coordinates of the rectangle.
        """

        width, height = self.size
        _, _, theta = self.position

        c, s = np.cos(theta), np.sin(theta)

        # Form rotation matrix for 2D world
        rot_matrix = np.array([[c, -s], [s, c]])

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
        return world_corners
    
    def get_aabb(self) -> AABB:
        """
        Computes and returns the Axis-Aligned Bounding Box for the body.
        """

        # Get the 2D rectangle corners
        corners = self.get_corners()

        # Form the AABB
        min_x = np.min(corners[:, 0])
        max_x = np.max(corners[:, 0])
        min_y = np.min(corners[:, 1])
        max_y = np.max(corners[:, 1])

        return AABB(min_x, max_x, min_y, max_y)
    
    def get_axes(self) -> np.array:
        """
        Returns unit vectors of body rotation, relative to world space.
        Needed for the Separating Axis Test (SAT).
        """

        x, y, theta = self.position
        c, s = np.cos(theta), np.sin(theta)

        x_axis = np.array([c, s])
        y_axis = np.array([-s, c])
        return np.array([x_axis, y_axis])
    
    def get_dim(self) -> int:
        return len(self.size) 