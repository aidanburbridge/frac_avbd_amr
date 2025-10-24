# SOLVER
import numpy as np
from geometry.primitives import Body, quat_from_rotvec, quat_log, quat_mult, quat_conjugate
from .constraints import Constraint, ContactConstraint, clamp
from . import collisions

class Solver:
    def __init__(self, dt, num_iterations, gravity=-9.81):

        self.dt = float(dt)
        self.iterations = int(num_iterations)
        self.gravity = float(gravity)
        self.dof = None

        self.bodies : list[Body] = []
        self.persistent_constraints: list[Constraint] = []
        self.contact_constraints: list[ContactConstraint] = []

        self.beta  = 100      # penalty ramp factor
        self.gamma = 0.99     # warm-start decay
        self.alpha = 0.99     # stabilization for hard rows
        self.post_stabilize = True
        self.mu = 0.6         # default friction

        # Private state for the solver step
        self._all_constraints: list[Constraint] = []
        self._contact_cache: dict = {}
        self._incidence_map: dict = {}
        

    def add_body(self, body:Body):
        
        if self.dof:
            if self.dof == body.dof:
                self.bodies.append(body)
            else:
                raise ValueError(f"Cannot add a {int((body.dof / 3) + 1)}D body to a {int((self.dof/3)+1)}D environment.")
        else:
            self.bodies.append(body)
            self.dof = body.dof


    def add_constraint(self, con: Constraint):
        self.persistent_constraints.append(con)

    def _build_contact_constraints(self):
        # Build the set of pairs to ignore from persistent constraints

        ignore_ids = set()
        for p_con in self.persistent_constraints:
            pair = tuple(sorted((id(p_con.A), id(p_con.B))))
            ignore_ids.add(pair)
            
        # Find contacts with broad phase + narrow phase
        raw_contacts = collisions.get_collisions(self.bodies, ignore_ids)
        
        # Create contact constraints and warm-start (read only from cache)
        new_contact_cons = []
        for c in raw_contacts:
            cons = ContactConstraint(contact=c, friction=self.mu)
            
            # Warm-start from the previous frame's cache
            ida, idb = id(c.bodyA), id(c.bodyB)
            cache = self._contact_cache

            feature_id = getattr(c, "feature_id", None)

            if feature_id is not None:
                key = (min(ida, idb), max(ida, idb), int(c.feature_id))     # min/max makes this order-independent

                if key in cache:
                    lam, kval = cache[key]
                    cons.lambda_[:] = lam
                    cons.penalty_k[:] = kval
            else:
                # optional fallback if feature_id is missing/unstable
                legacy_key = (min(ida, idb), max(ida, idb),
                              tuple(np.round(c.point, 3)),
                              tuple(np.round(c.normal, 3)))
                
                if legacy_key in cache:
                    lam, kval = cache[legacy_key]
                    cons.lambda_[:] = lam
                    cons.penalty_k[:] = kval

            new_contact_cons.append(cons)

        self.contact_constraints = new_contact_cons
    
    # Warm start the lambda and penalty (k) values for contacts - closer to correct answer
    def _write_contact_cache_post_solve(self):
        new_cache = {}
        for cons in self.contact_constraints:                       # Get contact patch - bodies involved - contact points (1-4 per contact) - normals (how many?)
            c = cons.contact
            feature_id = getattr(c, "feature_id", None)
            if feature_id is not None:
                key = (
                    min(id(c.bodyA), id(c.bodyB)),
                    max(id(c.bodyA), id(c.bodyB)),
                    int(c.feature_id)
                    )
            else:
                # fallback key if needed
                key = (min(id(c.bodyA), id(c.bodyB)),
                       max(id(c.bodyA), id(c.bodyB)),
                       tuple(np.round(c.point, 3)),
                       tuple(np.round(c.normal, 3)))
            new_cache[key] = (cons.lambda_.copy(), cons.penalty_k.copy())
        self._contact_cache = new_cache

    def step(self):
        dof = self.dof
        dt = self.dt
        for b in self.bodies:
            b.initial_pos = b.position.copy()

        # Broad and narrow phase collision detection + warm starting
        self._build_contact_constraints()
        self._all_constraints = self.persistent_constraints + self.contact_constraints
        print(f"Contacts: {sum(c.rows() for c in self.contact_constraints)}\n Pairs: {len(self.contact_constraints)}")
        for con in self.contact_constraints:
            c = con.contact
            print(f"\t Pair ({id(c.bodyA)%1000}, {id(c.bodyB)%1000}) normal: {c.normal.round(3)}")
        # TODO redundant loops!!

        # Initialize once per frame
        for con in self._all_constraints:
            con.initialize()

        # Build incidence map for efficient lookup when looping through constraints later
        self._incidence_map = {id(b): [] for b in self.bodies}
        for con in self._all_constraints:
            if con.A:                
                self._incidence_map[id(con.A)].append(con)
            if con.B:
                self._incidence_map[id(con.B)].append(con)

        # Warm start
        for con in self._all_constraints:
            for r in range(con.rows()):
                if self.post_stabilize:
                    con.penalty_k[r] = np.clip(self.gamma * con.penalty_k[r], 1.0, con.k_max[r])
                else:
                    if con.is_hard[r]:
                        con.lambda_[r]   = con.lambda_[r] * self.alpha * self.gamma
                    con.penalty_k[r] = np.clip(self.gamma * con.penalty_k[r], 1.0, con.k_max[r])

                # Cap penalty by material stiffness for *non-hard* rows
                if not np.isinf(con.stiffness[r]):
                    con.penalty_k[r] = min(con.penalty_k[r], con.stiffness[r])

        # Set intertial state and guess position for faster solve
        for b in self.bodies:
            #b.initial_pos = b.position.copy()
            if b.static:                                            # Don't move static bodies 
                continue

            if dof == 3:    # body is rect_2D
                y = b.position.copy() + b.velocity * dt
                y[1] += self.gravity * dt * dt                      # Update inertial position to include gravity's effect
                b.position = b.position + b.velocity * dt

            elif dof == 6:  # body is box_3D
                y = b.position.copy()
                y[:3] += b.velocity[:3] * dt         # Update translation position
                y[3:] = b.integrate_rotation(dt)
                b.position = y.copy()
                y[1] += self.gravity * dt * dt

            b.inertial_pos = y                                      # Copy inertial position y into internal variable intertial_pos

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
                if dof == 3: # --- 2D body: 3D solve ---
                    M = np.diag([b.mass, b.mass, b.inertia if np.isscalar(b.inertia) else float(b.inertia)])
                    pos_diff = (b.position - b.inertial_pos)
                    lhs = M / (dt*dt)
                    rhs = -(M / (dt*dt)) @ pos_diff

                elif dof == 6: # --- 3D rigid body: 6D solve ---
                    mI3 = b.mass * np.eye(3)
                    Iw  = b.I_world()        # world-frame inertia
                    M   = np.block([[mI3, np.zeros((3,3))],
                                    [np.zeros((3,3)), Iw]])  # (6,6)
                    e6 = b.delta_twist_from(b.inertial_pos)

                    lhs = M / (dt*dt)
                    rhs = (M / (dt*dt)) @ e6

                for con in self._incidence_map[id(b)]:
                    con.compute_constraint(current_alpha)
                    con.compute_derivatives(b)
                    J = con.JA if con.A is b else con.JB
                    H = con.HA if con.A is b else con.HB

                    for r in range(con.rows()):
                        hard = con.is_hard[r]
                        lam = con.lambda_[r] if hard else 0.0
                        k = con.penalty_k[r]
                        C = con.C[r]
                        Jr = J[r]
                        
                        if hard:
                            fmag = clamp(k*C + lam, con.fmin[r], con.fmax[r])
                        else:
                            fmag = k*C

                        absf = abs(fmag)

                        if H is not None and H.shape == (con.rows(), dof, dof) and np.any(H[r]):
                            col_norms = np.linalg.norm(H[r], axis=0)
                            Gdiag = np.diag(col_norms) * absf

                        else:
                            Gdiag = np.diag(np.abs(Jr) * absf)

                        rhs += Jr * fmag
                        lhs += k * np.outer(Jr, Jr) #+ Gdiag
                        #print("J[r]: ", Jr, "Force clipped: ", fmag, "Force: " ,(k*C +lam))
                        #print("RHS: ", rhs, "\t LHS: ", lhs) 

                try:
                    delta = np.linalg.solve(lhs, rhs)
                    if dof == 3:
                        b.position -= delta
                    elif dof == 6:
                        dx = delta[:3]
                        dth = delta[3:]
                        b.position[:3] -= dx
                        dq = quat_from_rotvec(dth)
                        b.position[3:] = quat_mult(dq, b.position[3:])
                    print(f"[Iteration: {num_it}]\tbody {id(b)%1000}\t||rhs||: {np.linalg.norm(rhs):.3e} "
                          f"\tcond(lhs): {np.linalg.cond(lhs):.2e}\t||delta_x||: {np.linalg.norm(dx):.3e}"
                          f"\t||delta_rot||: {np.linalg.norm(dth)}\tbeta: {self.beta}")
                except np.linalg.LinAlgError:
                    eps = 1e-9
                    lhs = lhs + eps*np.eye(lhs.shape[0])
                    delta = np.linalg.solve(lhs, rhs)
                    # print("Warning: Linear solve failed. LHS may be singular or ill-conditioned.")
                    # pass


            # Dual update (skip on the extra post-stabilization pass)
            if num_it < self.iterations:
                for con in self._all_constraints:
                    con.compute_constraint(current_alpha)           # refresh C at new positions
                    for r in range(con.rows()):                     # Hard constraints update λ via clamped force; soft rows keep λ=0
                        hard = con.is_hard[r]
                        kmax = con.k_max[r]
                        kmat = con.stiffness[r]
                        # 1) Update lambda for hard rows
                        if hard:
                            con.lambda_[r] = clamp(con.penalty_k[r] * con.C[r] + con.lambda_[r], con.fmin[r], con.fmax[r])
                            #print(f"Lambda: {con.lambda_[r]} Penalty K: {con.penalty_k[r]} C: {con.C[r]} f_min: {con.fmin[r]} f_max: {con.fmax[r]}")
                            #con.lambda_[r] = clamp(con.lambda_[r] - con.penalty_k[r]*con.C[r], con.fmin[r], con.fmax[r])

                        # 2) Ramp k
                        k_ramp = con.penalty_k[r] + self.beta * abs(con.C[r])
                        #print(f"K Ramp: ", k_ramp)

                        if hard:
                            if con.fmin[r] < con.lambda_[r] < con.fmax[r]:
                                con.penalty_k[r] = min(k_ramp, kmax)
                            else:
                                con.penalty_k[r] = min(k_ramp, min(kmax, kmat))
                        else:
                            con.penalty_k[r] = min(k_ramp, min(kmax, kmat))



            if num_it == self.iterations - 1:
                inv_dt = 1.0 / dt                                   # Multiplication is faster
                for b in self.bodies:
                    if b.static:
                        continue
                    else:
                        if dof == 3:
                            b.prev_vel = b.velocity.copy()
                            b.velocity = (b.position - b.initial_pos) * inv_dt
                        elif b.dof == 6:
                            q0 = b.initial_pos[3:]
                            q1 = b.position[3:]
                            q_rel = quat_mult(q1, quat_conjugate(q0))
                            d_theta = quat_log(q_rel)
                            omega = d_theta * inv_dt
                            b.velocity[:3] = (b.position[:3] - b.initial_pos[:3]) * inv_dt
                            b.velocity[3:] = omega

        self._write_contact_cache_post_solve()