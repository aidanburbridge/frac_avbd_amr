from constraints.constraint import Constraint
from dataclasses import dataclass
import numpy as np
from bodies import Body
from collisions import Contact


@dataclass
class ContactPoint:
    # Minimal per-contact cache for the frame
    normal: np.ndarray        # (2,) unit vector from A→B
    tangent: np.ndarray       # (2,) perpendicular to normal (right-hand)
    rA: np.ndarray            # (2,) local-to-world offset of contact on A (world frame)
    rB: np.ndarray            # (2,) local-to-world offset of contact on B (world frame)
    C0:  np.ndarray           # (2,) [normal, tangent] at frame start (with margin on normal)
    # Precomputed Jacobian rows for speed (normal then tangent)
    JAn: np.ndarray           # (3,)
    JBn: np.ndarray           # (3,)
    JAt: np.ndarray           # (3,)
    JBt: np.ndarray           # (3,)
    feature_id: int           # persistent feature id for warm-merging
    stick: bool = False       # static friction flag (optional)


class ContactConstraint(Constraint):
    COLLISION_MARGIN = 5e-4

    def __init__(self, contact:Contact, friction):
        super().__init__(contact.bodyA, contact.bodyB, rows=2)
        self.contact = contact
        self.friction = float(friction)
        self.fmin[0] = 0.0    # λ_n ≥ 0

        n = contact.normal.astype(float)
        t = np.array([-n[1], n[0]], float)
        self.n, self.t = n, t

        # Frame-start caches
        self.C0 = np.zeros(2)
        self.JAn = np.zeros(3)
        self.JAt = np.zeros(3)
        self.JBn = np.zeros(3)
        self.JBt = np.zeros(3)

    def initialize(self) -> bool:
        # Use frame-start positions (q0) to build lever arms and C0.
        A, B = self.A, self.B
        n, t = self.n, self.t

        pA = self.contact.point.astype(float)
        depth = float(self.contact.depth)
        pB = pA - n * depth

        # self.n[:] = self.contact.normal
        # self.t[:] = [-self.n[1], self.n[0]]

        rA0 = pA - A.initial_pos[:2]
        rB0 = pB - B.initial_pos[:2]

        zA_n = rA0[0]*self.n[1] - rA0[1]*self.n[0]
        zB_n = rB0[0]*self.n[1] - rB0[1]*self.n[0]
        zA_t = rA0[0]*self.t[1] - rA0[1]*self.t[0]
        zB_t = rB0[0]*self.t[1] - rB0[1]*self.t[0]
        # zA_n = -np.cross(rA0, self.n)
        # zB_n =  np.cross(rB0, self.n)
        # zA_t = -np.cross(rA0, self.t)
        # zB_t =  np.cross(rB0, self.t)

        # Per-body rows cached
        self.JAn[:] = [self.n[0], self.n[1], zA_n]
        self.JAt[:] = [self.t[0], self.t[1], zA_t]
        self.JBn[:] = [-self.n[0], -self.n[1], -zB_n]
        self.JBt[:] = [-self.t[0], -self.t[1], -zB_t]

        # C0 at frame start, with small positive bias along normal
        # n·(pB - pA) = -depth for penetration
        # self.C0[0] = -depth + self.COLLISION_MARGIN
        # self.C0[1] = np.dot(t, (pB - pA))  # 0 for corner-depth model
        self.C0[0] = np.dot(n, (A.initial_pos[:2] + rA0) - (B.initial_pos[:2] + rB0)) + self.COLLISION_MARGIN
        self.C0[1] = np.dot(t, (A.initial_pos[:2] + rA0) - (B.initial_pos[:2] + rB0))

        # self.C0[0] = np.dot(self.n, (B.initial_pos[:2]+rB0) - (A.initial_pos[:2]+rA0)) + self.COLLISION_MARGIN
        # self.C0[1] = np.dot(self.t, (B.initial_pos[:2]+rB0) - (A.initial_pos[:2]+rA0))
        return True

    def compute_constraint(self, alpha: float) -> None:
        # Δp from frame start → current iterate
        dA = self.A.position - self.A.initial_pos
        dB = self.B.position - self.B.initial_pos
        # Taylor update (normal, then tangent)
        self.C[0] = (1.0 - alpha) * self.C0[0] +( np.dot(self.JAn, dA) + np.dot(self.JBn, dB))
        self.C[1] = (1.0 - alpha) * self.C0[1] +( np.dot(self.JAt, dA) + np.dot(self.JBt, dB))

        # friction cone from current λ_n
        lam_n = max(self.lambda_[0], 0.0)
        self.fmin[1] = -self.friction * lam_n
        self.fmax[1] =  self.friction * lam_n

    def compute_derivatives(self, body: Body) -> None:
        # Copy cached rows into JA/JB for the requested body
        if body is self.A:
            self.JA[0,:] = self.JAn
            self.JA[1,:] = self.JAt
        else:
            self.JB[0,:] = self.JBn
            self.JB[1,:] = self.JBt

    
    def update_bounds(self):
        # Update friction bounds based on the current normal force lambda
        lambda_n = max(self.lambda_[0], 0.0)
        self.fmin[1] = -self.friction * lambda_n
        self.fmax[1] =  self.friction * lambda_n