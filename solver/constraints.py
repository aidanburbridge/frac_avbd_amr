# CONSTRAINTS
import numpy as np
from dataclasses import dataclass
from geometry.primitives import Body
from abc import ABC, abstractmethod


class Constraint(ABC):
    """ 
    Abstract parent class of a constraint.
    The number of DOFs that this contraint affects is controlled by body velocity vector entries (3 for 2D and 6 for 3D).
    The number of rows a contstraint has is dependent on the type of constraint.
        Ex: A ContactConstraint has 2 - one frictional, one non-penetration.
            A DistanceConstraint will have 1 row - enforces rest distance.
    """
    def __init__(self, A: Body, B: Body, rows: int):            # Where A and B are bodies that have some sort of contraint between the two
        self.A = A
        self.B = B
        self._rows = int(rows)
        
        # Infer per-body generalized-velocity DOFs from velocity vectors
        dofA = int(self.A.velocity.shape[0]) if self.A is not None else 0
        dofB = int(self.B.velocity.shape[0]) if self.B is not None else 0

        # Jacobians for each body
        self.JA = np.zeros((rows, dofA))                        # Jacobian rows for A
        self.JB = np.zeros((rows, dofB))                        # Jacobian rows for B

        self.HA = np.zeros((rows, dofA, dofA))
        self.HB = np.zeros((rows, dofB, dofB))
        self.C  = np.zeros(rows)                                # Constraint function value

        self.lambda_ = np.zeros(rows)                           # Lambda
        self.penalty_k = np.ones(rows) * 1.0                    # k_solver
        self.k_max = np.ones(rows) * 1e6                        # Set cap to avoid crazy ratio
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
    
    @property
    def is_hard(self) -> np.ndarray:
        """ Returns the hard constraint rows of the constraint. """
        return np.isinf(self.stiffness)
    
    def make_hard(self, rows: np.ndarray | list | None = None):
        """ Set given rows as hard, if none is given, make all rows hard. """
        if rows is None:
            self.stiffness[:] = np.inf
        else:
            self.stiffness[np.asarray(rows)] = np.inf

    def make_soft(self, k: float, rows: np.ndarray | list | None = None):
        """ Set given rows to a soft constraint with supplied stiffness (k) value. """
        if rows is None:
            self.stiffness[:] = float(k)
        else:
            self.stiffness[np.asarray(rows)] = float(k)

### -------------------- Contact Constraint -------------------- ###

@dataclass
class _ContactFrameCache:
    """ Caches sized by DOF and row count. """
    C0: np.ndarray          # (rows,)
    JA: np.ndarray          # (rows, dofA)
    JB: np.ndarray          # (rows, dofB)

class ContactConstraint(Constraint):
    COLLISION_MARGIN = 5e-4

    def __init__(self, contact, friction: float):
        self.contact = contact
        self.friction = float(friction)

        A = contact.bodyA
        B =  contact.bodyB

        dim = A.get_dim()
        self.dof = A.velocity.shape[0]
        assert dim in (2,3)

        rows = 1 + (dim - 1)                                    # One for normal (no penetration) & tangents for friction (2D: dim - 1 = 1; 3D: dim - 1 = 2)
        super().__init__(A, B, rows=rows)

        self.make_hard()                                        # Make this a hard constraint - contact & friction should be hard
        self.fmin[0] = 0.0
        self.n = contact.normal.astype(float)
        self.tangets = _orthonormal_tangent_basis(self.n)       # Get tangents from normal

        self._cache: _ContactFrameCache | None = None

    def initialize(self) -> bool:                               # Use frame-start positions (q0) to build lever arms and C0.

        A = self.A
        B = self.B

        dim = A.get_dim()
        rows = self.rows()

        pA = self.contact.point.astype(float)
        depth = float(self.contact.depth)
        pB = pA - self.n * depth

        rA0 = pA - A.initial_pos[:dim]
        rB0 = pB - B.initial_pos[:dim]

        # Allocate caches with dim
        dof_A = A.velocity.shape[0]
        dof_B = B.velocity.shape[0]
        C0 = np.zeros(rows, dtype=float)
        JA = np.zeros((rows, dof_A), dtype=float)
        JB = np.zeros((rows, dof_B), dtype=float)

        # Row 0 of constraint: normal
        if dim == 2:                                            # 2D rotational
            ang_A = rA0[0]*self.n[1] - rA0[1]*self.n[0]
            ang_B = rB0[0]*self.n[1] - rB0[1]*self.n[0]
            JA[0, :dof_A] = [ self.n[0],  self.n[1],  ang_A]
            JB[0, :dof_B] = [-self.n[0], -self.n[1], -ang_B]
        else: # dim == 3                                        # 3D rotational
            ang_A = np.cross(rA0, self.n)
            ang_B = np.cross(rB0, self.n)
            JA[0, :dof_A] = np.hstack([ self.n,  ang_A])
            JB[0, :dof_B] = np.hstack([-self.n, -ang_B])

        # Rows 1-2 of constraint: tangential (friction)

        for k, t in enumerate(self.tangets, start=1):           # Start at 1 because row 0 is filled
            if dim == 2:
                ang_A = rA0[0] * t[1] - rA0[1] * t[0]
                ang_B = rB0[0] * t[1] - rB0[1] * t[0]
                JA[k, :dof_A] = [ t[0],  t[1],  ang_A]
                JB[k, :dof_B] = [-t[0], -t[1], -ang_B]
            else: # dim == 3
                ang_A = np.cross(rA0, t)
                ang_B = np.cross(rB0, t)
                JA[k, :dof_A] = np.hstack([ t,  ang_A])
                JB[k, :dof_B] = np.hstack([-t, -ang_B])
        
        delta0 = (A.position[:dim] + rA0) + (B.position[:dim] + rB0)
        C0[0] = float(np.dot(self.n, delta0)) + self.COLLISION_MARGIN
        
        for k, t in enumerate(self.tangets, start=1):
            C0[k] = float(np.dot(t, delta0))

        self._cache = _ContactFrameCache(C0=C0, JA=JA, JB=JB)
        return True

    def compute_constraint(self, alpha: float) -> None:
        assert self._cache is not None
        A = self.A
        B = self.B
        
        if self.dof == 3:
            dA = A.position - A.initial_pos
            dB = B.position - B.initial_pos
            self.C[:] = (1.0 - alpha) * self._cache.C0 + (self._cache.JA @ dA + self._cache.JB @ dB)

        
        elif self.dof == 6:
            dA = A.delta_twist_from(A.initial_pos) if A is not None else 0.0
            dB = B.delta_twist_from(B.initial_pos) if B is not None else 0.0
            self.C[:] = (1.0 - alpha) * self._cache.C0 + (self._cache.JA @ dA if np.ndim(dA) else 0.0) + (self._cache.JB @ dB if np.ndim(dB) else 0.0)


        # friction cones
        lam_n = max(self.lambda_[0], 0.0)
        mu = self.friction

        for r in range(1, self.rows()):
            self.fmin[r] = -mu * lam_n
            self.fmax[r] =  mu * lam_n

    def compute_derivatives(self, body: Body) -> None:
        assert self._cache is not None
        if body is self.A:
            self.JA[:,:] = self._cache.JA
        else:
            self.JB[:,:] = self._cache.JB
    
    def update_bounds(self):
        # Update friction bounds based on the current normal force lambda
        lambda_n = max(self.lambda_[0], 0.0)
        for r in range(1, self.rows()):
            self.fmin[r] = -self.friction * lambda_n
            self.fmax[r] =  self.friction * lambda_n


### -------------------- Utils & Helpers -------------------- ###
def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def _orthonormal_tangent_basis(n: np.ndarray) -> list[np.ndarray]:
    """ Return 1 tangent in 2D, 2 tangents in 3D. """
    n = np.asarray(n, float)
    D = n.shape[0]
    if D == 2: # 2D
        t = np.array([-n[1], n[0]], float)
        norm_t = np.linalg.norm(t)
        if norm_t == 0:
            return [t]
        t /= norm_t
        return [t]
    else: # 3D
        # Gram-Schmidt to guarantee 2 orthonormals exist for 3D
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = a - n * np.dot(n, a)
        t1 /= (np.linalg.norm(t1) + 1e-12)
        t2 = np.cross(n, t1)
        t2 /= (np.linalg.norm(t2) + 1e-12)
        return [t1, t2]
    
