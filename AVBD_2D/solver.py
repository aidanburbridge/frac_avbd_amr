# SOLVER
import numpy as np
from bodies import Body
from constraints.constraint import Constraint, clamp
from constraints.contact import ContactConstraint
import collisions_adv as collisions

class Solver:
    def __init__(self, dt, num_iterations, gravity=-9.81):

        self.dt = float(dt)
        self.iterations = int(num_iterations)
        self.gravity = float(gravity)

        self.bodies : list[Body] = []
        self.persistent_constraints: list[Constraint] = []
        self.contact_constraints: list[ContactConstraint] = []

        self.beta  = 1e6      # penalty ramp factor
        self.gamma = 0.98     # warm-start decay
        self.alpha = 0.99     # stabilization for hard rows
        self.post_stabilize = False
        self.mu = 0.6         # default friction

        # Private state for the solver step
        self._all_constraints: list[Constraint] = []
        self._contact_cache: dict = {}
        self._incidence_map: dict = {}
        

    def add_body(self, body:Body):
        self.bodies.append(body)

    def add_constraint(self, con: Constraint):
        self.persistent_constraints.append(con)

    def _build_contact_constraints(self):
        # 1. Build the set of pairs to ignore from persistent constraints
        ignore_ids = set()
        for p_con in self.persistent_constraints:
            pair = tuple(sorted((id(p_con.A), id(p_con.B))))
            ignore_ids.add(pair)
            
        # 2. Get raw contact data from the collision system
        raw_contacts = collisions.get_collisions(self.bodies, ignore_ids)
        
        new_contact_cons = []
        new_cache = {}
        for c in raw_contacts:
            cons = ContactConstraint(contact=c, friction=self.mu)
            
            # 3. Warm-start from the previous frame's cache
            ida, idb = id(c.bodyA), id(c.bodyB)
            # A simple cache key. The C++ version is more robust using feature IDs.
            # key = tuple(sorted((ida, idb))) 
            key = (min(ida, idb), max(ida, idb), tuple(np.round(c.point, 3)), tuple(np.round(c.normal, 3)))
            if key in self._contact_cache:
                lam, kval = self._contact_cache[key]
                cons.lambda_[:] = lam
                cons.penalty_k[:] = kval
            
            new_contact_cons.append(cons)
            
            # 4. Build the cache for the *next* frame
            new_cache[key] = (cons.lambda_.copy(), cons.penalty_k.copy())

            
        self._contact_cache = new_cache
        self.contact_constraints = new_contact_cons

   
    def step(self):

        dt = self.dt        # Will this be changed as we go?
        for b in self.bodies:
            b.initial_pos = b.position.copy()

        # Broad and narrow phase collision detection + warm starting
        self._build_contact_constraints()
        self._all_constraints = self.persistent_constraints + self.contact_constraints

        ##print("Num constraints: ", len(self._all_constraints))

        # initialize once per frame (could be useful?)
        for con in self._all_constraints:
            con.initialize()

        # Build incidence map for efficient lookup when looping through constraints later
        self._incidence_map = {id(b): [] for b in self.bodies}
        for con in self._all_constraints:
            if con.A:                
                self._incidence_map[id(con.A)].append(con)
            if con.B:
                self._incidence_map[id(con.B)].append(con)

        #print("Incidence map:")
        #for body_id, constraints in self._incidence_map.items():
            #print(f"Body id {body_id}: {[type(con).__name__ for con in constraints]}")

        # warm start
        # solver.py – inside the dual update pass
        for con in self._all_constraints:
            for r in range(con.rows()):
                if self.post_stabilize:
                    con.penalty_k[r] = np.clip(self.gamma * con.penalty_k[r], 1.0, con.k_max[r])
                else:
                    con.lambda_[r]   = con.lambda_[r] * self.alpha * self.gamma
                    con.penalty_k[r] = np.clip(self.gamma * con.penalty_k[r], 1.0, con.k_max[r])

                # Cap penalty by material stiffness for *non-hard* rows (matches C++)
                if not np.isinf(con.stiffness[r]):
                    con.penalty_k[r] = min(con.penalty_k[r], con.stiffness[r])

        # Get intertial state (y)
        for b in self.bodies:
            if not b.static:                                    # Don't move static bodies                                           # Inertial velocity updates 
                y = b.position + b.velocity*dt
                y[1] += self.gravity*dt*dt   # Gravity update for vertical pos

                b.inertial_pos = y                     # Copy inertial position into velocity as starting point
                b.position = b.position + b.velocity * dt

        # Main solver loop - Newton Method iterations
        """
        q is position
        q_i+1 = q_i + f'(q)/f"(q)
        f' = ∇Π(q_i) = M/Δt²(y - q_i)
        f" = ∇²Π(q_i) = M/Δt²

                    LHS                               RHS 
        (M/Δt² + constraint stiffnesses)(Δq) = - (M/Δt² (y - q_i) + constraint forces)
        
        """
        total_iters = self.iterations + (1 if self.post_stabilize else 0)
        for num_it in range(total_iters):

            current_alpha = self.alpha

            if self.post_stabilize:
                current_alpha = 1.0 if num_it < self.iterations else 0.0

            # # Update friction from normals
            # for cons in self._all_constraints:
            #     cons.update_bounds()

            for b in self.bodies:

                # Skip static bodies
                if b.static:
                    continue

                M = np.diag([b.mass, b.mass, b.inertia])

                lhs = M / (dt*dt)                                           # M/Δt² - inertial Hessian - (matrix)
                rhs = M / (dt*dt) @ (b.position - b.inertial_pos)           # (M/Δt²)(y - q) - inertial residual force - (vector)

                for con in self._incidence_map[id(b)]:
                    #if isinstance(con, ContactConstraint):
                        #print("Contact depth:", con.contact.depth, "Constraint: ", con.C[0])
                    con.compute_constraint(current_alpha) # Question: What is alpha??
                    con.compute_derivatives(b)

                    #print("Current constraint C[0]: ", con.C[0],"\t Current derivative JA of constraint: ", con.JA)
                    
                    J = con.JA if con.A is b else con.JB                              # Get corresponding Jacobian to body
                    H = con.HA if con.A is b else con.HB

                    for r in range(con.rows()):
                        lam = con.lambda_[r] if np.isinf(con.stiffness[r]) else 0.0
                        k = con.penalty_k[r]
                        C = con.C[r]
                        Jr = J[r]
                        
                        force_magnitude = clamp(k * C + lam, con.fmin[r], con.fmax[r])
                        ##print("Force magnitude: ", force_magnitude)
                        # Diagonally lumped geometric stiffness G' (Sec. 3.5)
                        # If the constraint provided true second derivatives H, use their column norms.
                        # Otherwise, fall back to a simple quasi-Newton diagonal from |J| like before.
                        absf = abs(force_magnitude)
                        #print("Force: ", absf)
                        if H is not None and H.shape == (con.rows(), 3, 3) and np.any(H[r]):
                            col_norms = np.linalg.norm(H[r], axis=0)  # (3,)
                            Gdiag = np.diag(col_norms) * absf
                        else:
                            Gdiag = np.diag(np.abs(Jr)) * absf
                        
                        rhs += Jr * force_magnitude
                        lhs += k * np.outer(Jr, Jr) + Gdiag

                        #print("RHS: ", rhs)
                        #print("LHS: ", lhs)


                # Solve for change of position
                try:
                    delta_p = np.linalg.solve(lhs, rhs)
                    #print("Body position: ",b.position,"\t Delta_p", delta_p)

                    b.position -= delta_p
                except np.linalg.LinAlgError:
                    pass

            # Dual update (skip on the extra post-stab pass)
            if num_it < self.iterations:
                for con in self._all_constraints:

                    con.compute_constraint(current_alpha)  # refresh C at new positions

                    for r in range(con.rows()):
                        # Hard constraints update λ via clamped force; soft rows keep λ=0
                        if np.isinf(con.stiffness[r]): # Only hard constraints have lambda updated this way

                            new_lambda = clamp(con.penalty_k[r] * con.C[r] + con.lambda_[r], con.fmin[r], con.fmax[r])
                        
                            con.lambda_[r] = new_lambda

                        # Ramp up penalty if force is not clamped
                        if con.fmin[r] < con.lambda_[r] < con.fmax[r]:
                            # Cap growth by both a global max and the material stiffness
                            max_cap = min(con.k_max[r], con.stiffness[r] if not np.isinf(con.stiffness[r]) else con.k_max[r])
                            con.penalty_k[r] = min(con.penalty_k[r] + self.beta * abs(con.C[r]), max_cap)

            if num_it == self.iterations -1:
                #print("We made it here. ")
                inv_dt = 1.0 / dt # Multiplication is faster
                for b in self.bodies:
                    if b.static:
                        continue
                    else:
                        b.prev_vel = b.velocity.copy()
                        b.velocity = (b.position - b.initial_pos) * inv_dt