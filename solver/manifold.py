
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
        self.A = A
        self.B = B

        self.constraint_dict : dict[int, ContactConstraint] = {}    # dictionary of { (p_key, n_key), ContactConstraint }
        self.separation_frames: dict[int, int]= {}                  # dictionary of { (p_key, n_key), int }

    @property
    def constraints(self) -> list[Constraint]:
        return list(self.constraint_dict.values())
    
    def update_from_contacts(self, contacts: list, friction: float, max_sep_frames: int) -> None:
        
        seen_keys = set()

        for c in contacts:
            print("Contact feature id: ", c.feature_id)
            key = c.feature_id

            if key in self.constraint_dict:
                print(f"\t\t!!Old constraint used for {c.bodyA.body_id} and {c.bodyB.body_id}.!!\n")
                cons = self.constraint_dict[key]
                cons.contact = c
                self.separation_frames[key] = 0
            else:
                print(f"\t\tNew constraint made for {c.bodyA.body_id} and {c.bodyB.body_id}.")
                cons = ContactConstraint(contact=c, friction=friction)
                self.constraint_dict[key] = cons
                self.separation_frames[key] = 0
            seen_keys.add(key)

        print("Seen keys: ", seen_keys)
        print("Current keys: ", self.constraint_dict.keys())

        to_delete = []

        for key in self.constraint_dict.keys():
            if key not in seen_keys:
            #     self.separation_frames[key] += 1
            #     if self.separation_frames[key] > max_sep_frames:
                to_delete.append(key)
        
        for key in to_delete:
            self.constraint_dict.pop(key, None)
            self.separation_frames.pop(key, None)

    def is_empty(self) -> bool:
        return len(self.constraint_dict) == 0