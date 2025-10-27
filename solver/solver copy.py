# SOLVER
import numpy as np
from geometry.primitives import Body, quat_from_rotvec, quat_log, quat_mult, quat_conjugate
from .constraints import Constraint, ContactConstraint, clamp
from . import collisions
from .manifold import Manifold
from collections import defaultdict


class Solver:
    def __init__(self, dt, num_iterations, gravity=-9.81):

        self.dt = float(dt)
        self.iterations = int(num_iterations)
        self.gravity = float(gravity)
        self.dof = None

        self.bodies : list[Body] = []
        self.persistent_constraints: list[Constraint] = []
        self.contact_constraints: list[ContactConstraint] = []

        self.beta  = 10      # penalty ramp factor
        self.gamma = 0.99     # warm-start decay
        self.alpha = 0.95     # stabilization for hard rows
        self.post_stabilize = True
        self.mu = 0.6         # default friction

        # # Private state for the solver step
        # self._all_constraints: list[Constraint] = []
        # self._contact_cache: dict = {}
        # self._incidence_map: dict = {}

        self.num_bodies:int = 0
        self.manifolds: dict[tuple[int,int], Manifold] = {}
        self.separation_frames = 1
        self._frame_id = 0
        

    def add_body(self, body:Body):
        
        if self.dof == None:
            self.dof = body.dof

        if self.dof == body.dof:
            self.num_bodies += 1

            body.body_id = self.num_bodies
            print(f"Body id: {body.body_id}")
            self.bodies.append(body)

        else:
            raise ValueError(f"Cannot add a {int((body.dof / 3) + 1)}D body to a {int((self.dof/3)+1)}D environment.")


    def _build_contact_constraints(self):

        raw = collisions.get_collisions(self.bodies, ignore_ids=None)
        
        # Group by canonical pair (ida < idb) and canonicalize each contact orientation
        pairmap = defaultdict(list)
        id2body = {int(b.body_id): b for b in self.bodies}

        for c in raw:
            ida = int(c.bodyA.body_id)
            idb = int(c.bodyB.body_id)
            a, b = (ida, idb) if ida < idb else (idb, ida)
            if c.bodyA.body_id != a:
                # Swap to keep normal pointing A->B consistently
                c.bodyA, c.bodyB = c.bodyB, c.bodyA
                c.normal = -c.normal
            pairmap[(a, b)].append(c)

        # Update manifolds per pair (rebuild fresh; warm-start only λ,k)
        for (a, b), contacts in pairmap.items():
            mf = self.manifolds.get((a, b))
            if mf is None:
                mf = Manifold(id2body[a], id2body[b])
                self.manifolds[(a, b)] = mf

            # OPTIONAL: cheap dedup per feature in the same frame
            uniq = {}
            for c in contacts:
                fid = getattr(c, "feature_id", None)
                uniq[fid if fid is not None else id(c)] = c

            mf.update_from_contacts(
                contacts=list(uniq.values()),
                friction=self.mu,
                dist_eps=0.1,
                cos_eps=0.95
            )

        # Age manifolds not seen this frame (no contacts): they’ll clear themselves
        for key in list(self.manifolds):
            if key not in pairmap and self.manifolds[key].is_empty():
                del self.manifolds[key]

        # The only constraints we solve are the current manifold constraints
        self.contact_constraints = [con
                                    for mf in self.manifolds.values()
                                    for con in mf.constraints]

    def step(self):

        dt = self.dt
        dof = self.dof

        # Broad and narrow phase collision detection + warm starting
        self._build_contact_constraints()

        #print(f"Manifold: {self.manifolds}")
        self._all_constraints = self.persistent_constraints + self.contact_constraints

        # Warm start
        for con in self._all_constraints:
            con.initialize()
            print(f"\nConstraints for body {con.A.body_id} and body {con.B.body_id}:")
            for r in range(con.rows()):
                if self.post_stabilize:
                    con.penalty_k[r] = np.clip(self.gamma * con.penalty_k[r], 1.0, con.k_max[r])
                else:
                    if con.is_hard[r]:
                        con.lambda_[r]   = con.lambda_[r] * self.alpha * self.gamma
                    con.penalty_k[r] = np.clip(self.gamma * con.penalty_k[r], 1.0, con.k_max[r])

                # Cap penalty by material stiffness for *non-hard* rows
                if not con.is_hard[r]:
                    con.penalty_k[r] = min(con.penalty_k[r], con.stiffness[r])
            print(f"\t Penalty value: {con.penalty_k[0]}\t\t Lambda value: {con.lambda_[0]}")

        
        # Set intertial state and guess position for faster solve
        for b in self.bodies:
            if b.static:                                            # Don't move static bodies 
                continue
            b.initial_pos = b.position.copy()
            y = b.position.copy()
            y[:3] += b.velocity[:3] * dt        # Update translation position
            y[3:] = b.integrate_rotation(dt)    # Update rotational position
            b.position = y.copy()               # TODO copy y to keep simple for now

            y[1] += self.gravity * dt *dt       # Update due to gravity position
            b.inertial_pos = y.copy()
            # b.position = b.position + b.velocity * dt # + warm start acceleration stuff
            
            # Add in acceleration warm start stuff
            # accel = (b.velocity - b.prev_vel) /dt
            # accelExt = accel[1]

        # Main solver loop - Newton Method iterations
        """
        q is position
        q_i+1 = q_i + f'(q)/f"(q)
        f' = ∇Π(q_i) = M/Δt²(y - q_i)
        f" = ∇²Π(q_i) = M/Δt²

                    LHS                               RHS 
        (M/Δt² + constraint stiffnesses)(Δq) = - (M/Δt² (y - q_i) + constraint forces)
        
        """
        # lhs = np.ndarray()
        # rhs = np.ndarray()
        # Gdiag = np.ndarray()
      # --- Main iterations in twist space (6D for 3D, 3D for 2D) ---
        total_iters = self.iterations + (1 if self.post_stabilize else 0)
        for num_it in range(total_iters):

            current_alpha = self.alpha
            if self.post_stabilize:
                current_alpha = 1.0 if num_it < self.iterations else 0.0

            for b in self.bodies:
                if b.static:
                    continue
                # Build inertial Hessian & residual in the correct dimensionality
                mI3 = b.mass * np.eye(3)
                Iw  = b.I_world()        # world-frame inertia
                M   = np.block([[mI3, np.zeros((3,3))],
                                [np.zeros((3,3)), Iw]])  # (6,6)
                e6 = b.delta_twist_from(b.inertial_pos)

                force = -(M / (dt*dt)) @ e6
                #print(f"Force: {force}")
                hessian = M / (dt*dt)

                for con in self._all_constraints:
                    #print("Curent alpha: ", current_alpha)
                    con.compute_constraint(current_alpha)
                    con.compute_derivatives(b)
                    J = con.JA if con.A is b else con.JB
                    H = con.HA if con.A is b else con.HB

                    for r in range(con.rows()):
                        lam = con.lambda_[r] if con.is_hard[r] else 0.0
                        k = con.penalty_k[r]
                        C = con.C[r]
                        Jr = J[r]
                        
                        if con.is_hard[r]:
                            f_add = clamp(k*C + lam, con.fmin[r], con.fmax[r])
                            #print(f"Additional force to apply: {f_add}")
                        else:
                            f_add = k*C

                        if H is not None and H.shape == (con.rows(), dof, dof) and np.any(H[r]):
                            col_norms = np.linalg.norm(H[r], axis=0)
                            Gdiag = np.diag(col_norms) * abs(f_add)

                        else:
                            Gdiag = np.diag(np.abs(Jr) * abs(f_add))

                        force -= Jr * f_add
                        hessian += np.outer(Jr, Jr * k) + Gdiag
                        #print("J[r]: ", Jr, "Force clipped: ", fmag, "Force: " ,(k*C +lam))
                        #print("RHS: ", rhs, "\t LHS: ", lhs) 

                try:
                    delta = np.linalg.solve(hessian, force)
                   # print(f"Positional delta: {delta[:3]}\tCurrent position: {b.position[:3]}")
                    dx = delta[:3]
                    dth = delta[3:]
                    b.position[:3] += dx
                    dq = quat_from_rotvec(dth)
                    b.position[3:] = quat_mult(dq, b.position[3:])

                    # print(f"[Iteration: {num_it}]\tbody {id(b)%1000}\t||rhs||: {np.linalg.norm(rhs):.3e} "
                    #       f"\tcond(lhs): {np.linalg.cond(lhs):.2e}\t||delta_x||: {np.linalg.norm(dx):.3e}"
                    #       f"\t||delta_rot||: {np.linalg.norm(dth)}\tbeta: {self.beta}")
                    
                except np.linalg.LinAlgError:
                    eps = 1e-9
                    hessian = hessian + eps*np.eye(hessian.shape[0])
                    delta = np.linalg.solve(hessian, force)
                    # print("Warning: Linear solve failed. LHS may be singular or ill-conditioned.")


            # Dual update (skip on the extra post-stabilization pass)
            if num_it < self.iterations:
                for con in self._all_constraints:
                    con.compute_constraint(current_alpha)           # refresh C at new positions - TODO the constraint is NOT recomputed, it says the same thing as before...
                    for r in range(con.rows()):                     # Hard constraints update λ via clamped force; soft rows keep λ=0

                        # 1) Update lambda for hard rows
                        if con.is_hard[r]:
                            con.lambda_[r] = clamp(con.penalty_k[r] * con.C[r] + con.lambda_[r], con.fmin[r], con.fmax[r])

                            if con.fmin[r] < con.lambda_[r] and con.lambda_[r] < con.fmax[r]:
                                con.penalty_k[r] = con.penalty_k[r] + self.beta * abs(con.C[r])

                        else:
                            con.penalty_k[r] = min(con.stiffness[r], con.penalty_k[r] + self.beta * abs(con.C[r]))


            if num_it == self.iterations - 1:
                inv_dt = 1.0 / dt                                   # Multiplication is faster
                for b in self.bodies:
                    if b.static:
                        continue
                    else:
                        q0 = b.initial_pos[3:]
                        q1 = b.position[3:]
                        q_rel = quat_mult(q1, quat_conjugate(q0))
                        d_theta = quat_log(q_rel)
                        omega = d_theta * inv_dt
                        b.velocity[:3] = (b.position[:3] - b.initial_pos[:3]) * inv_dt
                        b.velocity[3:] = omega

        #self._write_contact_cache_post_solve()