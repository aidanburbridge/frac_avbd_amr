# CONSTRAINTS
import numpy as np
from dataclasses import dataclass
from geometry.primitives import Body, box_face_vectors
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
        self.bodyA = A
        self.bodyB = B
        self._rows = int(rows)
        
        # Infer per-body generalized-velocity DOFs from velocity vectors
        self.dof = int(self.bodyA.velocity.shape[0]) if self.bodyA is not None else 0

        # Jacobians for each body
        self.JA = np.zeros((rows, self.dof))                        # Jacobian rows for A
        self.JB = np.zeros((rows, self.dof))                        # Jacobian rows for B

        self.HA = np.zeros((rows, self.dof, self.dof))
        self.HB = np.zeros((rows, self.dof, self.dof))
        self.C  = np.zeros(rows)                                # Constraint function value

        self.lambda_ = np.zeros(rows)                           # Lambda
        self.penalty_k = np.zeros(rows)                         # k_solver
        self.k_min = np.ones(rows) * 1.0                        # small positive floor
        self.k_max = np.ones(rows) * 1e9                        # Set cap to avoid crazy ratio
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
        B = contact.bodyB

        dim = A.get_dim()
        self.dof = A.velocity.shape[0]
        assert dim in (2,3)

        rows = 1 + (dim - 1)                                    # One for normal (no penetration) & tangents for friction (2D: dim - 1 = 1; 3D: dim - 1 = 2)
        super().__init__(A, B, rows=rows)

        self.make_hard()                                        # Make this a hard constraint - contact & friction should be hard
        #self.fmax[0] = 0.0
        self.fmin[0] = 0.0
        self.n = contact.normal.astype(float)
        self.tangents = _orthonormal_tangent_basis(self.n)       # Get tangents from normal

        self._cache: _ContactFrameCache | None = None

        # for debugging
        self.point_list = []
        self.depth = None

    def initialize(self) -> bool:                               # Use frame-start positions (q0) to build lever arms and C0.

        A = self.bodyA
        B = self.bodyB

        dim = A.get_dim()
        rows = self.rows()

        pA = self.contact.point.astype(float)                   # Contact point 
        depth = float(self.contact.depth)                       # Distance between contact points
        self.depth = depth
        pB = pA - self.n * depth                                # Contact point on B derived from collision normal and depth
        
        rA0 = pA - A.initial_pos[:dim]
        rB0 = pB - B.initial_pos[:dim]

        # Allocate caches with dim
        C0 = np.zeros(rows, dtype=float)
        JA = np.zeros((rows, self.dof), dtype=float)
        JB = np.zeros((rows, self.dof), dtype=float)

        # Row 0 of constraint: normal
        ang_A = np.cross(rA0, self.n)
        ang_B = np.cross(rB0, self.n)
        JA[0, :self.dof] = np.hstack([ self.n,  ang_A])
        JB[0, :self.dof] = np.hstack([-self.n, -ang_B])

        # Rows 1-2 of constraint: tangential (friction)
        for k, t in enumerate(self.tangents, start=1):           # Start at 1 because row 0 is filled
            ang_A = np.cross(rA0, t)
            ang_B = np.cross(rB0, t)
            JA[k, :self.dof] = np.hstack([ t,  ang_A])
            JB[k, :self.dof] = np.hstack([-t, -ang_B])
    
        pA0 = A.position[:dim] + rA0
        pB0 = B.position[:dim] + rB0

        # For debugging
        self.point_list.extend([pA0, pB0])

        # Define normal constraint so C > 0 when penetration exceeds margin
        # dot(n, pA0 - pB0) = -depth, so use depth - margin
        #C0[0] = float(-np.dot(self.n, (pA0 - pB0)) - self.COLLISION_MARGIN)
        C0[0] = float(np.dot(self.n, (pA0 - pB0)) - self.COLLISION_MARGIN)

        #print(f"Pa - Pb: {pA0-pB0}, dot prod {np.dot(self.n, (pA0 - pB0))}, C0 {C0[0]}")

        for k, t in enumerate(self.tangents, start=1):
            C0[k] = float(np.dot(t, (pA0 - pB0)))

        self._cache = _ContactFrameCache(C0=C0, JA=JA, JB=JB)
        return True

    def compute_constraint(self, alpha: float) -> None:
        assert self._cache is not None
        A = self.bodyA
        B = self.bodyB

        dA = A.delta_twist_from(A.initial_pos) if A is not None else 0.0
        dB = B.delta_twist_from(B.initial_pos) if B is not None else 0.0
        
        self.C[:] = (1.0 - alpha) * self._cache.C0 + (self._cache.JA @ dA if np.ndim(dA) else 0.0) + (self._cache.JB @ dB if np.ndim(dB) else 0.0)

        # friction cones
        lam_n_mag = max(self.lambda_[0], 0.0)
        #lam_n_mag = abs(self.lambda_[0])
        mu = self.friction

        for r in range(1, self.rows()):
            self.fmax[r] =  mu * lam_n_mag
            self.fmin[r] = -mu * lam_n_mag

    def compute_derivatives(self, body: Body) -> None:
        assert self._cache is not None
        if body is self.bodyA:
            self.JA[:, :] = self._cache.JA
        elif body is self.bodyB:
            self.JB[:, :] = self._cache.JB
        else:
            # Unrelated body: do not modify Jacobians
            return
    
    def update_bounds(self):
        # Update friction bounds based on the current normal force lambda
        #lambda_mag = abs(self.lambda_[0])
        lambda_mag = max(self.lambda_[0], 0.0)

        for r in range(1, self.rows()):
            self.fmin[r] = -self.friction * lambda_mag
            self.fmax[r] =  self.friction * lambda_mag
    
    def warmstart_from(self, other):
        m = min(self.rows(), other.rows())
        self.lambda_[:m]   = other.lambda_[:m]
        self.penalty_k[:m] = other.penalty_k[:m]

class FaceBondPoint(Constraint):
    def __init__(self, A: Body, B: Body,
                 pA_local: np.ndarray, pB_local: np.ndarray,
                 normal: np.ndarray, k_n: float, k_t: float,
                 area: float,
                 tensile_strength: float,
                 fracture_energy: float | None = None,
                 fracture_toughness: float | None = None,
                 material_E: float | None = None,
                 material_nu: float | None = None):
        super().__init__(A, B, rows=3)

        # bodies
        self.bodyA = A
        self.bodyB = B

        # Local-space anchors
        self.pA_local = np.asarray(pA_local, dtype=float).reshape(3)
        self.pB_local = np.asarray(pB_local, dtype=float).reshape(3)

        # 1. Establish the basis vectors in world space initially
        n_world = _normalize(normal)
        t1_world, t2_world = _orthonormal_tangent_basis(n_world)

        # 2. Convert them to Body A's local space
        # This attaches the "bond direction" to Body A, so it rotates with the voxel
        # Solves co-rotation problem
        Ra_init = A.rotmat()
        self.n_local  = Ra_init.T @ n_world
        self.t1_local = Ra_init.T @ t1_world
        self.t2_local = Ra_init.T @ t2_world

        # Placeholders for the current frame's world-space vectors
        self.n_current = np.zeros(3)
        self.t1_current = np.zeros(3)
        self.t2_current = np.zeros(3)

        # Material parameters (bookkeeping)
        self.material_E = None if material_E is None else float(material_E)
        self.material_nu = None if material_nu is None else float(material_nu)

        # Base Elastic Stiffness
        # Rows: [0]-Normal / [1]-Tangent1 / [2]-Tangent2
        self.stiffness[:] = np.array([float(k_n), float(k_t), float(k_t)], dtype=float)

        self.rest = np.zeros(3, dtype=float)
        self._rest_initialized = False

        # Bounds preset
        self.fmax[:] = np.inf
        self.fmin[:] = -np.inf

        self._cache = None

        # Fracture Properties
        fracture_energy = _get_fracture_energy(material_E, material_nu, fracture_energy, fracture_toughness)

        # 1. Elastic limits
        self.delta_n0 = (tensile_strength * area) / float(k_n)
        self.delta_s0 = (tensile_strength * area) / float(k_t)
        # 2. Critical limits TODO check on the math here
        self.delta_nc = (2.0 * fracture_energy) / tensile_strength # normal
        self.delta_sc = (2.0 * fracture_energy) / tensile_strength # shear

        # Fracture State
        self.lam_current = 0.0
        self.is_broken = False
        self.is_cohesive = False

        # Fracture history
        self.lam_max_committed = 0.0

        # Damage to scale stiffness
        self.damage = 0.0

    def initialize(self):
        A, B = self.bodyA, self.bodyB

        # 1. Get current Rotations
        RA = A.rotmat()
        RB = B.rotmat()

        # 2. Update Anchors (World Space)
        rA = RA @ self.pA_local
        rB = RB @ self.pB_local
        pA = A.position[:3] + rA
        pB = B.position[:3] + rB
        dp = pA - pB

        # 3. Rotate the LOCAL basis vectors into CURRENT WORLD vectors
        # If the assembly rotates, RA changes, so n_current changes direction in world space,
        # but stays aligned relative to Body A.
        self.n_current  = RA @ self.n_local
        self.t1_current = RA @ self.t1_local
        self.t2_current = RA @ self.t2_local

        # 4. Initialize Rest Length (Scalar distances don't care about rotation)
        if not self._rest_initialized:
            self.rest[0] = float(np.dot(self.n_current,  dp))
            self.rest[1] = float(np.dot(self.t1_current, dp))
            self.rest[2] = float(np.dot(self.t2_current, dp))
            self._rest_initialized = True

        # 5. Build Jacobians using CURRENT directions
        directions = (self.n_current, self.t1_current, self.t2_current)
        for row, dir_vec in enumerate(directions):
            ang_A = np.cross(rA, dir_vec)
            ang_B = np.cross(rB, dir_vec)
            self.JA[row, :self.dof] = np.hstack([ dir_vec,  ang_A])
            self.JB[row, :self.dof] = np.hstack([-dir_vec, -ang_B])

        if self.is_broken:
            self.penalty_k[:] = 0.0
        else:
            self.penalty_k[:] = self.stiffness * (1.0 - self.damage)
        self._cache = None
        return True

    def compute_constraint(self, alpha: float):
        """ Project current separation onto CURRENT basis vectors. """

        # If broken, no constraint to enforce. 
        if self.is_broken:
            self.C[:] = 0.0
            self.penalty_k[:] = 0.0
            return
        
        A, B = self.bodyA, self.bodyB
        RA = A.rotmat()
        RB = B.rotmat()
        rA = RA @ self.pA_local
        rB = RB @ self.pB_local
        pA = A.position[:3] + rA
        pB = B.position[:3] + rB
        dp = pA - pB
        
        self.C[0] = float(np.dot(self.n_current,  dp) - self.rest[0])
        self.C[1] = float(np.dot(self.t1_current, dp) - self.rest[1])
        self.C[2] = float(np.dot(self.t2_current, dp) - self.rest[2])

        # Separation delta
        d_n = max(self.C[0], 0)                     # normal 
        d_s = np.sqrt(self.C[1]**2 + self.C[2]**2)  # shear magnitude

        # Crack initiation check 
        if not self.is_cohesive and not self.is_broken:
            Psi = (d_n / self.delta_n0)**2 + (d_s / self.delta_s0)**2
            if Psi >= 1.0: # Go to cohesive region
                self.is_cohesive = True
                self.lam_max_committed = self.lam_calc(d_n, d_s)

        # Cohesive softening
        if self.is_cohesive:
            lam = self.lam_calc(d_n, d_s)

            # Update history max lambda (irreverisbility)
            self.lam_current = max(self.lam_max_committed, lam)

            if self.lam_current >= 1.0:
                self.damage = 1.0
                self.penalty_k[:] = 0.0
            else:
                # Calculate damage based on current state
                curr_lam_cr = np.sqrt((self.delta_n0/self.delta_nc)**2) # Pure Mode I TODO maybe update for more complex situations
                if d_s > 1e-9: # Non-zero I guess (?) TODO
                    curr_lam_cr = np.sqrt( ((d_n/self.delta_nc)**2 + (d_s/self.delta_sc)**2)/
                                           ((d_n/self.delta_n0)**2 + (d_s/self.delta_s0)**2))
            
                if self.lam_max_committed > curr_lam_cr:
                    self.damage = (self.lam_max_committed - curr_lam_cr) / (1.0 - curr_lam_cr)
                    self.damage = clamp(self.damage, 0.0, 1.0)
                else:
                    self.damage = 0.0
                
                self.penalty_k[:] = self.stiffness * (1.0 - self.damage)
    
    def commit_damage(self):
        """Finalize and commit the damage at the end of the time step."""
        if self.is_cohesive:
            self.lam_max_committed = max(self.lam_max_committed, self.lam_current)

            if self.lam_max_committed >= 1.0:
                self.is_broken = True
    

    def compute_derivatives(self, body: Body):
        """Recompute J rows using CURRENT basis vectors."""
        A, B = self.bodyA, self.bodyB
        RA = A.rotmat()
        RB = B.rotmat()
        rA = RA @ self.pA_local
        rB = RB @ self.pB_local
        
        directions = (self.n_current, self.t1_current, self.t2_current)
        
        if body is self.bodyA:
            for row, dir_vec in enumerate(directions):
                ang_A = np.cross(rA, dir_vec)
                self.JA[row, :self.dof] = np.hstack([dir_vec, ang_A])
        elif body is self.bodyB:
            for row, dir_vec in enumerate(directions):
                ang_B = np.cross(rB, dir_vec)
                self.JB[row, :self.dof] = np.hstack([-dir_vec, -ang_B])

    def lam_calc(self, d_n, d_s):
        return np.sqrt((d_n/self.delta_nc)**2 + (d_s/self.delta_sc)**2)
                

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
        #t1 = a - n * np.dot(n, a)
        t1 = np.cross(n, a)
        t1 = _normalize(t1)
        t2 = np.cross(n, t1)
        t2 = _normalize(t2)
        return [t1, t2]
    
def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def build_face_bonds(A: Body, B: Body, E: float, nu: float, tensile_strength: float, fracture_toughness: float) -> list[FaceBondPoint]:
    """
    Create face bond constraints between two bodies for number of Guass points - (2x2 Guass points).
    Works in local coordinates.
    Returns a list of FaceBondPoint.
    """

    # Guass legendre polynomial points * half extents 
    # could do this:
    # 1) get A -> B vector with vec = B.center() - A.center()
    # 2) find face on A between A and B with max(dot_prod(vec, box_face_centers(A)))
    # 3) this will give me face center then add (half extents * xi) to center to get bond points

    # Shear modulus
    G = E / (2*(1+nu))

    # hard code 2 guass points per direction
    xi = 1.0 / np.sqrt(3)
    w_share = 0.25
    
    vec = B.get_center() - A.get_center()
    #print("Vec: ", vec)
    ni = int(np.argmax(np.abs(vec)))            # axis of separation
    #print("ax: ", ni)
    sgn = 1.0 if vec[ni] >= 0.0 else -1.0
    #print("sign: ", sgn)

    # local half extents
    hA = 0.5 * np.asarray(A.size, dtype=float)
    hB = 0.5 * np.asarray(B.size, dtype=float)

    # face normals in world aligned (initially); n points from A to B
    n_world = np.zeros(3)
    n_world[ni] = sgn

    # choose tangential axes (two remaining indices)
    tangential_axes = [i for i in range(3) if i != ni]
    t1i, t2i = tangential_axes
    #print("tangent axes: ", tangential_axes)

    # effective area and thickness (for stiffness scaling)
    area = (2*hA[t1i]) * (2*hA[t2i])

    # thickness along normal direction
    h_norm = (hA[ni] + hB[ni])  # for same sized voxels this is the same as the voxel width
    k_n = E * area / max(h_norm, 1e-12)
    k_t = G * area / max(h_norm, 1e-12)

    bonds: list[FaceBondPoint] = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            pA_local = np.zeros(3)
            pB_local = np.zeros(3)
            # normals
            pA_local[ni]    =  sgn * hA[ni]
            pB_local[ni]    = -sgn * hB[ni]
            # tangents
            pA_local[t1i]   =   s1 * xi * hA[t1i]
            pA_local[t2i]   =   s2 * xi * hA[t2i]
            pB_local[t1i]   =   s1 * xi * hB[t1i]
            pB_local[t2i]   =   s2 * xi * hB[t2i]

            bonds.append(
                FaceBondPoint(
                    A,
                    B,
                    pA_local,
                    pB_local,
                    n_world,
                    k_n*w_share,
                    k_t*w_share,
                    area*w_share,
                    tensile_strength=tensile_strength,
                    fracture_toughness=fracture_toughness,
                    material_E=E,
                    material_nu=nu,
                )
            )

    return bonds

def _get_fracture_energy( E: float, nu: float, G_c: float | None = None, K_Ic: float | None = None) -> float:
    """
    Get Critical Energy release rate (G_c) in J/m^2 from Fracture Toughness K_Ic.

    Also just convient here for error handling outside of constraint.
    """
    if G_c is not None:
        return float(G_c)
    
    if K_Ic is not None:
        
        calc_Gc = (K_Ic**2 * (1.0 - nu**2)) / E
        return calc_Gc
    
    raise ValueError("You must provide either Fracture Energy (G_c) OR Fracture Toughness (K_Ic).")    
