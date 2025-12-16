
from geometry.primitives import Body
from .constraints import Constraint, ContactConstraint
import numpy as np

def _contact_key(point: np.ndarray, normal: np.ndarray, decimals: int = 1) -> tuple[tuple, tuple]:
    
    p_key = tuple(np.round(point.astype(float), decimals))
    n_key = tuple(np.round(normal.astype(float), decimals))
    return p_key, n_key


class Manifold():
    """
    Persistent container for constraints and contacts between bodies (a pair of bodies).
    """

    def __init__(self, A: Body, B: Body):
        self.bodyA = A
        self.bodyB = B

        self.constraint_dict = {}    # dictionary of { (p_key, n_key), ContactConstraint }
        self.separation_frames = {}                  # dictionary of { (p_key, n_key), int }
        self.constraint_list = []
        self.max_sep_frames = 3                      # frames to keep stale contacts

    @staticmethod
    def _proximity_test(c, old_con, dist_eps = 2e-3, cos_eps=0.985):
        """
        Tests if old contact point and normal are close to new ones.
        """
        dp = float(np.linalg.norm(c.point - old_con.contact.point))
        ndot = float(np.dot(c.normal, old_con.contact.normal))
        #print(f"point new: {c.point}, point old: {old_con.contact.point}, dp: {dp} ... ndot: {ndot}")
        return (dp <= dist_eps) and (ndot >= cos_eps)
    
    @staticmethod
    def _key_from_contact(con, p_dec=3, n_dec=2):
        """Deterministic key without a feature id: quantized A-local point & normal."""

        p_key = tuple(np.round(con.point.astype(float), p_dec))
        n_key = tuple(np.round(con.normal.astype(float), n_dec))
        return ("local", p_key, n_key)
    
    def update_from_contacts(self, contacts: list, friction: float, dist_eps = 0.1, cos_eps=0.95) -> None:
        
        old = self.constraint_dict

        new_dict = {}
        used_old = set()
        
        for c in contacts:
            con = ContactConstraint(c, friction)

            matched_key = None
            for k_old, con_old in old.items():
                if con_old in used_old:
                    continue
                if self._proximity_test(c, con_old, dist_eps, cos_eps):
                    matched_key = k_old               # reuse key to keep continuity
                    con.warmstart_from(con_old)       # copy ONLY lambda_, penalty_k
                    used_old.add(con_old)
                    # Reset separation counter for this key
                    self.separation_frames[k_old] = 0
                    break

            key = matched_key if matched_key is not None else self._key_from_contact(c)
            new_dict[key] = con
            # Initialize sep counter for new keys
            if key not in self.separation_frames:
                self.separation_frames[key] = 0

        # Carry over unmatched old constraints for a limited number of frames
        for k_old, con_old in old.items():
            if con_old in used_old:
                continue
            sep = self.separation_frames.get(k_old, 0) + 1
            # If clearly separated (AABB disjoint), drop immediately
            if not self._aabb_overlap():
                sep = self.max_sep_frames + 1
            if sep <= self.max_sep_frames:
                new_dict[k_old] = con_old
                self.separation_frames[k_old] = sep
            else:
                self.separation_frames.pop(k_old, None)

        self.constraint_dict = new_dict
        self.constraints = list(new_dict.values())


    def is_empty(self) -> bool:
        return len(self.constraint_dict) == 0

    def _aabb_overlap(self) -> bool:
        """Conservative test using AABBs to quickly detect clear separation."""
        try:
            a = self.bodyA.get_aabb()
            b = self.bodyB.get_aabb()
        except Exception:
            return True

        # ND AABB case
        if hasattr(a, 'mins') and hasattr(a, 'maxs') and hasattr(b, 'mins') and hasattr(b, 'maxs'):
            minsA = np.asarray(a.mins, float)
            maxsA = np.asarray(a.maxs, float)
            minsB = np.asarray(b.mins, float)
            maxsB = np.asarray(b.maxs, float)
            return np.all(maxsA >= minsB) and np.all(maxsB >= minsA)

        # Named fields (2D/3D)
        def bounds(obj):
            xr = (getattr(obj, 'min_x', -np.inf), getattr(obj, 'max_x', np.inf))
            yr = (getattr(obj, 'min_y', -np.inf), getattr(obj, 'max_y', np.inf))
            zr = None
            if hasattr(obj, 'min_z') and hasattr(obj, 'max_z'):
                zr = (getattr(obj, 'min_z'), getattr(obj, 'max_z'))
            return xr, yr, zr

        ax, ay, az = bounds(a)
        bx, by, bz = bounds(b)
        def ov1(r1, r2):
            return not (r1[1] < r2[0] or r2[1] < r1[0])
        if not ov1(ax, bx) or not ov1(ay, by):
            return False
        if az is not None and bz is not None and not ov1(az, bz):
            return False
        return True
