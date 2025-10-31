# SOLVER
import numpy as np
from geometry.primitives import Body, quat_from_rotvec, quat_log, quat_mult, quat_conjugate
from .constraints import Constraint, ContactConstraint, clamp
from . import collisions
from .manifold import Manifold
from collections import defaultdict
from util.time_profiler import PhaseProfiler


class Solver:
    def __init__(self, dt, num_iterations, gravity=-9.81):

        self.dt = float(dt)
        self.iterations = int(num_iterations)
        self.gravity = float(gravity)
        self.dof = None

        self.bodies : list[Body] = []
        self.persistent_constraints: list[Constraint] = []
        self.contact_constraints: list[ContactConstraint] = []
        self.incidence_map: dict[int, list[ContactConstraint]] = {}        # dict[body_id, list[ContactConstraint]]

        self.beta  = 10      # penalty ramp factor
        self.gamma = 0.99     # warm-start decay
        self.alpha = 0.95     # stabilization for hard rows
        self.post_stabilize = True
        self.mu = 0.6         # default friction

        self.num_bodies:int = 0
        self.manifolds: dict[tuple[int,int], Manifold] = {}
        self.separation_frames = 3
        self._frame_id = 0

        # Debug/diagnostics
        self.debug_contacts: bool = False
        self.prof = PhaseProfiler()
        

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

        #print("New pairmap:")
        # Update manifolds per pair (rebuild fresh; warm-start only λ,k)
        for (a, b), contacts in pairmap.items():
            # print(f"\tBody A: {a} \tBody B: {b}\n\t\tContacts:")
            # for c in contacts:
            #     print("\t\t",c)
            mf = self.manifolds.get((a, b))
            if mf is None:
                mf = Manifold(id2body[a], id2body[b])
                self.manifolds[(a, b)] = mf
            # propagate solver threshold into manifold
            mf.max_sep_frames = int(self.separation_frames)

            # OPTIONAL: cheap dedup per feature in the same frame
            uniq = {}
            for c in contacts:
                fid = getattr(c, "feature_id", None)
                uniq[fid if fid is not None else id(c)] = c

            mf.update_from_contacts(
                contacts=list(uniq.values()),
                friction=self.mu,
                dist_eps=0.05,
                cos_eps=0.9
            )

        # Clear manifolds not seen this frame (no contacts this frame)
        for key, mf in list(self.manifolds.items()):
            if key not in pairmap:
                # Force-clear constraints to avoid stale contacts persisting
                mf.update_from_contacts(contacts=[], friction=self.mu, dist_eps=0.05, cos_eps=0.995)

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

        # For multiplication and not division
        inv_dt = 1.0 / dt
        inv_dt2 = 1.0 / (dt * dt)

        loop_amt = 0
        
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
            
            # TODO Add in acceleration warm start stuff
            # accel = (b.velocity - b.prev_vel) /dt
            # accelExt = accel[1]

        # Broad and narrow phase collision detection + warm starting
        with self.prof.phase("build cont. const."):
            self._build_contact_constraints()

        # Clear incidence map
        self.incidence_map = {}

        if self.debug_contacts:
            self.print_contacts_summary()

        self._all_constraints = self.persistent_constraints + self.contact_constraints

        # Warm start
        with self.prof.phase("warm start"):
            for con in self._all_constraints:
                con.initialize()
                #print(f"\nConstraints for body {con.A.body_id} and body {con.B.body_id}:")
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

                # Build incidence map #

                if int(con.bodyA.body_id) not in self.incidence_map:
                    self.incidence_map[int(con.bodyA.body_id)] = []
                self.incidence_map[int(con.bodyA.body_id)].append(con)

                if int(con.bodyB.body_id) not in self.incidence_map:
                    self.incidence_map[int(con.bodyB.body_id)] = []
                self.incidence_map[int(con.bodyB.body_id)].append(con)
        
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

            with self.prof.phase("compute constraints"):
                # Phase-local body deltas (positions are from inertial guess at this point)
                body_deltas = {}
                for b in self.bodies:
                    try:
                        body_deltas[int(b.body_id)] = b.delta_twist_from(b.initial_pos)
                    except Exception:
                        body_deltas[int(b.body_id)] = 0.0
                # Compute C using cached JA/JB when available and update friction bounds
                for c in self._all_constraints:
                    if hasattr(c, "_cache") and c._cache is not None:
                        da = body_deltas.get(int(c.bodyA.body_id), 0.0) if c.bodyA is not None else 0.0
                        db = body_deltas.get(int(c.bodyB.body_id), 0.0) if c.bodyB is not None else 0.0
                        C = (1.0 - current_alpha) * c._cache.C0
                        if np.ndim(da):
                            C = C + (c._cache.JA @ da)
                        if np.ndim(db):
                            C = C + (c._cache.JB @ db)
                        c.C[:] = C
                        # friction cones based on current normal lambda (match ContactConstraint semantics)
                        if c.rows() > 1:
                            lam_n_mag = max(c.lambda_[0], 0.0)
                            mu = float(getattr(c, "friction", 0.0))
                            c.fmax[1:] =  mu * lam_n_mag
                            c.fmin[1:] = -mu * lam_n_mag
                    else:
                        # Fallback for non-contact constraints
                        c.compute_constraint(current_alpha)

            with self.prof.phase("main solve"):
                for b in self.bodies:
                    if b.static:
                        continue

                    M = b.mass_mat
                    with self.prof.phase("delta twist"):
                        e6 = b.delta_twist_from(b.inertial_pos)
                        
                    with self.prof.phase("assemble solve"):
                        force = -(M * inv_dt2) @ e6
                        hessian = M * inv_dt2
                        cons_for_body = self.incidence_map.get(int(b.body_id), [])
                        if not cons_for_body:
                            pass
                        else:
                            J_rows = []
                            k_list = []
                            fadd_list = []
                            for con in cons_for_body:
                                # Select the Jacobian for this body without copying
                                if hasattr(con, "_cache") and con._cache is not None:
                                    if con.bodyA is b or int(con.bodyA.body_id) == int(b.body_id):
                                        J = con._cache.JA
                                    elif con.bodyB is b or int(con.bodyB.body_id) == int(b.body_id):
                                        J = con._cache.JB
                                    else:
                                        continue
                                else:
                                    # Fallback: compute derivatives (may copy)
                                    con.compute_derivatives(b)
                                    if con.bodyA is b:
                                        J = con.JA
                                    elif con.bodyB is b:
                                        J = con.JB
                                    else:
                                        continue
                                rows = con.rows()
                                if rows <= 0:
                                    continue
                                # Build f_add for all rows at once
                                k = con.penalty_k.copy()
                                C = con.C.copy()
                                lam = con.lambda_.copy()
                                fmin = con.fmin.copy()
                                fmax = con.fmax.copy()
                                hard = con.is_hard
                                f_add = np.where(hard, clamp(k * C + lam, fmin, fmax), k * C)
                                # Accumulate
                                J_rows.append(J)
                                k_list.append(k)
                                fadd_list.append(f_add)
                            if J_rows:
                                J_stack = np.vstack(J_rows)
                                k_stack = np.concatenate(k_list)
                                f_stack = np.concatenate(fadd_list)
                                # RHS: force -= J^T f
                                force -= J_stack.T @ f_stack
                                # LHS: hessian += J^T diag(k) J  == J^T (k[:,None]*J)
                                hessian += J_stack.T @ (k_stack[:, None] * J_stack)
                                # Diagonal stabilization term (avoid per-row np.diag allocations)
                                g_diag = (np.abs(J_stack).T @ np.abs(f_stack))
                                idx = np.arange(hessian.shape[0])
                                hessian[idx, idx] += g_diag

                    with self.prof.phase("lin alg solve"):
                        try:
                            delta = np.linalg.solve(hessian, force)
                        # print(f"Positional delta: {delta[:3]}\tCurrent position: {b.position[:3]}")
                            dx = delta[:3]
                            dth = delta[3:]
                            b.position[:3] += dx
                            dq = quat_from_rotvec(dth)
                            b.position[3:] = quat_mult(dq, b.position[3:])
                            
                        except np.linalg.LinAlgError:
                            eps = 1e-9
                            hessian = hessian + eps*np.eye(hessian.shape[0])
                            delta = np.linalg.solve(hessian, force)
                            # print("Warning: Linear solve failed. LHS may be singular or ill-conditioned.")

            with self.prof.phase("dual update"):
                # Dual update (skip on the extra post-stabilization pass)
                if num_it < self.iterations:
                    # Recompute body deltas with updated positions
                    body_deltas = {}
                    for b in self.bodies:
                        try:
                            body_deltas[int(b.body_id)] = b.delta_twist_from(b.initial_pos)
                        except Exception:
                            body_deltas[int(b.body_id)] = 0.0
                    for con in self._all_constraints:
                        # Refresh C quickly using cached J
                        if hasattr(con, "_cache") and con._cache is not None:
                            da = body_deltas.get(int(con.bodyA.body_id), 0.0) if con.bodyA is not None else 0.0
                            db = body_deltas.get(int(con.bodyB.body_id), 0.0) if con.bodyB is not None else 0.0
                            C = (1.0 - current_alpha) * con._cache.C0
                            if np.ndim(da):
                                C = C + (con._cache.JA @ da)
                            if np.ndim(db):
                                C = C + (con._cache.JB @ db)
                            con.C[:] = C
                        else:
                            con.compute_constraint(current_alpha)
                        # Vectorized lambda/penalty updates
                        hard = con.is_hard
                        if np.any(hard):
                            lam_new = clamp(con.penalty_k * con.C + con.lambda_, con.fmin, con.fmax)
                            # assign only hard rows
                            con.lambda_[hard] = lam_new[hard]
                            # interior rows (strictly within bounds)
                            interior = (con.lambda_ > con.fmin + 1e-12) & (con.lambda_ < con.fmax - 1e-12) & hard
                            if np.any(interior):
                                con.penalty_k[interior] = con.penalty_k[interior] + self.beta * np.abs(con.C[interior])
                        soft = ~hard
                        if np.any(soft):
                            con.penalty_k[soft] = np.minimum(con.stiffness[soft], con.penalty_k[soft] + self.beta * np.abs(con.C[soft]))
                        # Update friction bounds once from updated normal lambda
                        if con.rows() > 1:
                            lam_n_mag = max(con.lambda_[0], 0.0)
                            mu = float(getattr(con, "friction", 0.0))
                            con.fmax[1:] =  mu * lam_n_mag
                            con.fmin[1:] = -mu * lam_n_mag

            if num_it == self.iterations - 1:
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
        
        if (self._frame_id % 30) == 0:
            print(self.prof.report())
            self.prof.reset()
        self._frame_id += 1

        #print("Num loops: ", loop_amt) # 960 -> 192
        #self._write_contact_cache_post_solve()



### HELPERS ###
    def print_contacts_summary(self):
        """
        Print per-body contact summary for the current frame.
        Shows number of contact constraints and their points/normals/depths.
        """
        by_body = defaultdict(list)
        for con in self.contact_constraints:
            c = getattr(con, "contact", None)
            if c is None:
                continue
            # Entry from A's perspective
            by_body[int(con.bodyA.body_id)].append({
                "pair": (int(con.bodyA.body_id), int(con.bodyB.body_id)),
                "point": np.asarray(c.point, float),
                "normal": np.asarray(c.normal, float),
                "depth": float(c.depth),
                "rows": int(con.rows()),
                "lambda": con.lambda_,
            })
            # Entry from B's perspective (flip normal for clarity)
            by_body[int(con.bodyB.body_id)].append({
                "pair": (int(con.bodyA.body_id), int(con.bodyB.body_id)),
                "point": np.asarray(c.point, float),
                "normal": -np.asarray(c.normal, float),
                "depth": float(c.depth),
                "rows": int(con.rows()),
                "lambda": con.lambda_,
            })

        if not by_body:
            print("[contacts] No contact constraints this frame.")
            return

        print("[contacts] Per-body contact summary:")
        for bid in sorted(by_body.keys()):
            entries = by_body[bid]
            n_contacts = len(entries)
            n_rows = sum(e["rows"] for e in entries)
            print(f"  Body {bid}: {n_contacts} contacts, {n_rows} rows")
            for e in entries:
                p = e["point"]; n = e["normal"]; d = e["depth"]; pair = e["pair"]; l = e["lambda"]
                print(f"    pair={pair} p={p.round(3)} n={n.round(3)} d={d:.4f} lam={l}")