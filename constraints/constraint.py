# CONSTRAINT
import numpy as np
from bodies import Body
from abc import ABC, abstractmethod


# HARD CODED FOR 2D

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

class Constraint(ABC):
    def __init__(self, A: Body, B: Body, rows: int):                # Where A and B are bodies that have some sort of contraint between the two
        self.A = A
        self.B = B
        self._rows = int(rows)
        """ The number of DOFs that this contraint affects. 
            Ex: A ContactContstraint will have 2 rows - one frictional, one non-penetration
            A DistanceConstraint will have 1 row - rest distance to enforce"""

        # Per-row state
        self.JA = np.zeros((rows, 3))                               # Jacobian rows for A
        self.JB = np.zeros((rows, 3))                               # Jacobian rows for B
        self.HA = np.zeros((rows, 3, 3))                            # Optional geometric stiffness (can be zeros)
        self.HB = np.zeros((rows, 3, 3))
        self.C  = np.zeros(rows)                                    # Constraint function value

        self.lambda_ = np.zeros(rows)                               # Lambda
        self.penalty_k = np.ones(rows) * 1.0                       # k_solver
        self.k_max = np.ones(rows) * 1e6                            # Set cap to avoid crazy ratio
        self.stiffness = np.ones(rows) * np.inf
        self.fracture = np.ones(rows) * np.inf

        # Max values for row 
        self.fmin = -np.ones(rows) * np.inf
        self.fmax = np.ones(rows) * np.inf

        self._frame_initialized = False

    def rows(self) -> int:
        return self._rows

    @abstractmethod
    def compute_constraint(self, alpha:float) -> np.ndarray:
        """
        Computes and returns the constraint violation C(x) as a vector.
        C(x): shape (rows,)
        """
        pass

    @abstractmethod
    def compute_derivatives(self, body: Body) -> np.ndarray:
        """
        Computes and returns the Jacobian matrix J for the given body.
        J(body): shape (rows, 3)
        """
        pass
    
    def update_bounds(self):
        """
        Optional per-iteration bounds update for constraints with friction (or similar constraints).
        """
        pass

    def initialize(self) -> bool:
        """
        Called ONCE per time step. Cache anything constant over the frame:
        - For contacts: build manifold (clipping), compute basis, J rows, and C0.
        - For springs/joints: store C0 if needed.
        Return False to have the solver drop this force for this frame.
        """
        return True



# class ContactConstraint(Constraint):
#     """
#     Constraint with two rows:
#         0 : normal (non-penetration)    -> lambda_n  >= 0
#         1 : friction (tangent)          -> |lambda_t| <= mu * lambda_n 
#     """
#     def __init__(self, contact_info: Contact, friction: float):
#         super().__init__(contact_info.bodyA, contact_info.bodyB, rows=2)

#         self.contact_info = contact_info
#         self.friction = friction
#         self.fmin[0] = 0.0                                  # Normal row lower bound, enforce normal force can only push (no negative aka pulling)
    
#     def get_rows(self) -> int:
#         return 2                                            # 2 rows, one for friction, one for non-penetration
    
#     def violation(self) -> np.ndarray:

#         # Normal row violation
#         C_normal = self.contact_info.depth                  # Penetration depth informs normal contraint

#         # Tangent violation
#         normal_vector = self.contact_info.normal
#         tangent_vector = np.array([-normal_vector[1], normal_vector[0]]) # Perpendicular to normal vector
#         relative = (self.A.position[:2] - self.B.position[:2])
#         C_tangent = np.dot(relative, tangent_vector)

#         return np.array([C_normal, C_tangent], dtype=float)
    
#     def jacobian(self, body) -> np.ndarray:

#         normal = self.contact_info.normal
#         tangent = np.array([-normal[-1], normal[0]])        # Perpendicular to the normal

#         # Lever arm from body center of mass (COM) to contact point

#         rA = self.contact_info.point - self.A.position[:2]  # Body A COM to contact
#         rB = self.contact_info.point - self.B.position[:2]  # Body B COM to contact

#         # Moments produced about COMs
#         zA_norm =   rA[0] *  normal[1] - rA[1] *  normal[0]
#         zA_tan =    rA[0] * tangent[1] - rA[1] * tangent[0]
#         zB_norm =   rB[0] *  normal[1] - rB[1] *  normal[0]
#         zB_tan =    rB[0] * tangent[1] - rB[1] * tangent[0]
        
#         if body is self.A:
#             J_norm = np.array([ -normal[0], -normal[1], -zA_norm])
#             J_tan = np.array([ -tangent[0], -tangent[1], -zA_tan])
#         else:
#             J_norm = np.array([normal[0], normal[1], zB_norm])
#             J_tan = np.array([ tangent[0], tangent[1], zB_tan])           

#         return np.vstack([J_norm, J_tan])
    
#     def update_bounds(self):

#         lambda_n = max(self.lambda_[0], 0.0)
#         mu = self.friction
#         # Make the boudnary cone for friction, update based on current normal force
#         self.fmin[1] = -mu * lambda_n
#         self.fmax[1] = mu * lambda_n

# class DistanceConstraint(Constraint):
#     """
#     Constraint with one row:
#         0 : keep fixed distanced between COMs -> will adapt later to be like Young's Modulus
#     """
#     def __init__(self, A: Body, B: Body, length: float, stiffness: float):
#         super().__init__(A, B, rows=1)

#         self.length = length
#         self.pentalty_k[:] = stiffness
#         self.k_max[:] = stiffness
    
#     def violation(self) -> np.ndarray:
#         dist = self.B.position[:2] - self.A.position[:2]
#         current_length = np.linalg.norm(dist) + 1e-12
#         return np.array([ current_length - self.length])
    
#     def jacobian(self, body: Body) -> np.ndarray:

#         dist = self.B.position[:2] - self.A.position[:2]
#         current_length = np.linalg.norm(dist) + 1e-12
#         u = dist / current_length

#         if body is self.A:
#             J = np.array([-u[0], -u[1], 0.0])
#         else:
#             J = np.array([ u[0], u[1], 0.0])

#         return J.reshape(1,3)

#     def get_row_force(self):
#         # For a 1-row distance constraint:
#         # scalar generalized "force" on the row = λ + k*C
#         return (self.lambda_[0] + self.pentalty_k[0] * self.violation()[0])

# # could add class that adds an applied spring force
# # the spring force will be relative from where you clicked to how far you dragged the mouse
# # it would be a spring f=-kx and x is the length of the drag, set k_spring with argument
# # if I wanted to do a mouse interation it would just be a massless body B where A was the body clicked on