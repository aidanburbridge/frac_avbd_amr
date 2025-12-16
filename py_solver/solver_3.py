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
        self.next_assembly_id: int = 0
        self.manifolds: dict[tuple[int,int], Manifold] = {}
        self.separation_frames = 3
        self._frame_id = 0
        self.max_contacts_per_assembly_pair = 4

        # Debug/diagnostics
        self.debug_contacts: bool = False
        self.prof = PhaseProfiler()
        

    def add_body(self, body:Body):
        
        if self.dof == None:
            self.dof = body.dof

        if self.dof == body.dof:
            self.num_bodies += 1
            body.body_id = self.num_bodies

            if body.assembly_id is None:
                body.assembly_id = self.next_assembly_id
                self.next_assembly_id += 1
            else:
                self.next_assembly_id = max(self.next_assembly_id, int(body.assembly_id) + 1)

            print(f"Body id: {body.body_id} & assembly id: {body.assembly_id}")
            self.bodies.append(body)

        else:
            raise ValueError(f"Cannot add a {int((body.dof / 3) + 1)}D body to a {int((self.dof/3)+1)}D environment.")


    def _build_contact_constraints(self):

        raw = collisions.get_collisions(self.bodies, ignore_ids=None)
        #raw = self._limit_contacts_by_assembly(raw)
        
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

    def _limit_contacts_by_assembly(self, contacts: list) -> list:
        """
        Enforce a 4-point limit per assembly pair so voxel assemblies behave like single bodies.
        """
        max_per_pair = getattr(self, "max_contacts_per_assembly_pair", None)
        if not contacts or not max_per_pair or max_per_pair <= 0:
            return contacts

        buckets: dict[tuple[int, int], list] = defaultdict(list)
        passthrough = []

        def _assembly_id(body: Body) -> int:
            aid = getattr(body, "assembly_id", None)
            if aid is None:
                base = getattr(body, "body_id", None)
                return int(base if base is not None else id(body))
            return int(aid)

        for con in contacts:
            aid_a = _assembly_id(con.bodyA)
            aid_b = _assembly_id(con.bodyB)

            if aid_a == aid_b:
                passthrough.append(con)
                continue

            pair = (aid_a, aid_b) if aid_a < aid_b else (aid_b, aid_a)
            buckets[pair].append(con)

        if not buckets:
            return contacts

        allowed_ids = set()
        for pair_cons in buckets.values():
            pair_cons.sort(key=lambda c: float(getattr(c, "depth", 0.0)), reverse=True)
            for con in pair_cons[:max_per_pair]:
                allowed_ids.add(id(con))

        filtered = passthrough + [con for con in contacts if id(con) in allowed_ids]
        return filtered
    
    def add_persistent_constraints(self, constraints: list[Constraint]):
        self.persistent_constraints.extend(constraints)
                
    def step(self):

        if self.bodies is None:
            raise RuntimeError("No bodies have been added to the simulation!")
            

        dt = self.dt
        dof = self.dof

        # For multiplication and not division
        inv_dt = 1.0 / dt
        inv_dt2 = 1.0 / (dt * dt)
        
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
                        con.penalty_k[r] = np.clip(self.gamma * con.penalty_k[r], con.k_min[r], con.k_max[r])
                    else:
                        #if con.is_hard[r]:
                        con.lambda_[r]   = con.lambda_[r] * self.alpha * self.gamma
                        con.penalty_k[r] = np.clip(self.gamma * con.penalty_k[r], con.k_min[r], con.k_max[r])

                    # Cap penalty by material stiffness for *non-hard* rows
                    if not con.is_hard[r]:
                        con.penalty_k[r] = min(con.penalty_k[r], con.stiffness[r])
                       # print("Penalty value: ", con.penalty_k[0], "Stiffness: ", con.stiffness[0])

                # Build incidence map #

                if int(con.bodyA.body_id) not in self.incidence_map:
                    self.incidence_map[int(con.bodyA.body_id)] = []
                self.incidence_map[int(con.bodyA.body_id)].append(con)

                if int(con.bodyB.body_id) not in self.incidence_map:
                    self.incidence_map[int(con.bodyB.body_id)] = []
                self.incidence_map[int(con.bodyB.body_id)].append(con)


        # assemblies = {
        #     (b.assembly_id if b.assembly_id is not None else b.body_id)
        #     for b in self.bodies
        # }
        # print(f"{len(assemblies)} assemblies in the solver")

        
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

            with self.prof.phase("main solve"):
                for b in self.bodies:
                    if b.static:
                        continue

                    M = b.mass_mat
                    with self.prof.phase("delta twist"):
                        e6 = b.delta_twist_from(b.inertial_pos)
                        
                    with self.prof.phase("assemble solve"):
                        force = b.force_ws
                        force[:] = M @ e6
                        force *= -inv_dt2

                        hessian = b.hessian_ws
                        np.copyto(hessian, M)
                        hessian *= inv_dt2

                        cons_for_body = self.incidence_map.get(int(b.body_id), [])
                        if cons_for_body:
                            needed_rows = 0
                            prepared: list[tuple[Constraint, int]] = []
                            for con in cons_for_body:
                                con.compute_constraint(current_alpha)
                                con.compute_derivatives(b)
                                if hasattr(con, "update_bounds"):
                                    con.update_bounds()
                                m = con.rows()
                                if m <= 0:
                                    continue
                                prepared.append((con, m))
                                needed_rows += m

                            if needed_rows > 0:
                                ws = b.get_ws(needed_rows)
                                Jbuf, kbuf, fbuf = ws["J"], ws["k"], ws["f"]
                                rows = 0
                                for con, m in prepared:
                                    if con.bodyA is b:
                                        J = con.JA
                                    elif con.bodyB is b:
                                        J = con.JB
                                    else:
                                        continue

                                    k = con.penalty_k
                                    C = con.C
                                    lam = con.lambda_
                                    fmin = con.fmin
                                    fmax = con.fmax
                                    hard = con.is_hard
                                    f_add = np.where(hard, clamp(k * C + lam, fmin, fmax), k * C)

                                    Jbuf[rows:rows+m, :] = J
                                    kbuf[rows:rows+m] = k
                                    fbuf[rows:rows+m] = f_add
                                    rows += m

                                if rows:
                                    Jv = Jbuf[:rows, :]
                                    kv = kbuf[:rows]
                                    fv = fbuf[:rows]

                                    force -= Jv.T @ fv
                                    hessian += Jv.T @ (kv[:, None]*Jv)

                                    g_diag = (np.abs(Jv).T @ np.abs(fv))
                                    hessian[np.diag_indices_from(hessian)] += g_diag

                    with self.prof.phase("lin alg solve"):
                        try:
                            L = np.linalg.cholesky(hessian)
                            z = np.linalg.solve(L, force)
                            delta = np.linalg.solve(L.T, z)

                            # delta = np.linalg.solve(hessian, force)
                            # dx = delta[:3]
                            # dth = delta[3:]
                            # b.position[:3] += dx
                            # dq = quat_from_rotvec(dth)
                            # b.position[3:] = quat_mult(dq, b.position[3:])
                            
                        except np.linalg.LinAlgError:
                            eps = 1e-8
                            n = hessian.shape[0]
                            L = np.linalg.cholesky(hessian + eps*np.eye(n))
                            z = np.linalg.solve(L, force)
                            delta = np.linalg.solve(L.T, z)
                            # hessian = hessian + eps*np.eye(hessian.shape[0])
                            # delta = np.linalg.solve(hessian, force)

                        dx = delta[:3]
                        dth = delta[3:]
                        b.position[:3] += dx
                        dq = quat_from_rotvec(dth)
                        b.position[3:] = quat_mult(dq, b.position[3:])

            with self.prof.phase("dual update"):
                # Dual update (skip on the extra post-stabilization pass)
                if num_it < self.iterations:

                    for con in self._all_constraints:
                        con.compute_constraint(current_alpha)
                        hard = con.is_hard
                        if np.any(hard):
                            lam_new = clamp(con.penalty_k * con.C + con.lambda_, con.fmin, con.fmax)
                            con.lambda_[hard] = lam_new[hard]

                            inside_bounds = hard & (con.lambda_ > con.fmin) & (con.lambda_ < con.fmax)
                            if np.any(inside_bounds):
                                con.penalty_k[inside_bounds] = np.minimum(con.k_max[inside_bounds], con.penalty_k[inside_bounds] + self.beta * np.abs(con.C[inside_bounds]))
                        
                        soft = ~hard
                        if np.any(soft):
                            con.penalty_k[soft] = np.minimum(
                                np.minimum(con.k_max[soft], con.stiffness[soft]),
                                con.penalty_k[soft] + self.beta * np.abs(con.C[soft])
                                )
                            #print("penalty: ",con.penalty_k[soft],"  const: ", con.C[soft])
                        if hasattr(con, "update_bounds"):
                            con.update_bounds()

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
