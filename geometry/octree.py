# OCTREE
# Python standard libraries
from dataclasses import dataclass
import numpy as np
from typing import Optional
from tqdm import trange

# Project specific
from geometry.primitives import box_3D, box_face_vectors

### -------------------- Data structures -------------------- ###
@dataclass
class Bond:
    """ 
    Connection between two voxel bodies.
    Currently done across a shared face (6 connections per voxel).
    """
    bodyA_id : int
    faceA: int
    bodyB_id: int
    area: float
    length: float
    kn: float
    kt: float

@dataclass(frozen=True)                                         # Freeze class so values can't change -> must make new leaves instead
class Leaf:
    level: int
    i: int
    j: int
    k: int

    def size(self, h_base: float) -> float:
        return h_base / ( 2 ** self.level)
    
    def center(self, origin: np.ndarray, h_base: float) -> np.ndarray:
        sub_division = 2 ** self.level
        h = h_base / sub_division
        return origin + np.array([(self.i + 0.5) * h, (self.j + 0.5) * h, (self.k + 0.5) * h], dtype=float)

    def key(self) -> tuple[int, int, int]:
        return (self.level, self.i, self.j, self.k)
    
### --------------------  Constants -------------------- ###

# Offsets to find 6 face connected neighbors in the 3D grid
_NEIGHBOR_OFFSETS = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

_FACE_MAP = {
    (1,0,0): (0, 1),    # +X: A face +X = 0, B face -X = 1
    (0,1,0): (2, 3),    # +Y: A face +Y = 2, B face -Y = 3
    (0,0,1): (4, 5),    # +Z: A face +Z = 4, B face -Z = 5
    }

_DEF_NORMALS = np.array([
    [ 1, 0, 0],
    [-1, 0, 0],
    [ 0, 1, 0],
    [ 0,-1, 0],
    [ 0, 0, 1],
    [ 0, 0,-1],
    ], dtype=float)

### -------------------- Octree Functions -------------------- ###

def octree_from_occ(occ: np.ndarray, h_base: float = 1.0) -> tuple[list[Leaf], float]:
    """
    Create level 0 leaves from a uniform occupancy grid. An unrefined octree.
    Returns (leaves, h_base)
        - leaves: list of all the leaves in the tree at level 0
        - h_base: the dimension of the voxels at base level
    """
    nx, ny, nz = occ.shape

    leaves: list[Leaf]= []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if bool(occ[i, j, k]):
                    leaves.append(Leaf(0, i, j, k))

    return leaves, float(h_base)

def _leaf_neighbors6(leaf: Leaf) -> list[tuple[int, int, int]]:
    """ Return the 6 face neighbors' coordinates of a given leaf. """

    L, i, j, k = leaf.level, leaf.i, leaf.j, leaf.k
    return [(L, i+di, j+dj, k+dk) for (di,dj,dk) in _NEIGHBOR_OFFSETS]

def _index_by_level(leaves: list[Leaf]) -> dict[tuple[int, int, int], Leaf]:
    """ Create an index of leaves by their (level, i, j, k) coordinates. """
    levels: dict[int, dict[tuple[int, int, int], Leaf]] = {}
    for leaf in leaves:
        levels.setdefault(leaf.level, {})[(leaf.i, leaf.j, leaf.k)] = leaf
        # setdefault creates a new dictionary for a level if it doesn't exist yet
        # if it does exist, it will use the existing dictionary
    return levels

def _find_leaf_neighbor(level_maps, L, i, j, k) -> Optional[Leaf]:
    """ 
    Find a leaf neighbor at level L with face centered at (i, j, k).
    Tries the same level first, then coarser (L-1), then finer (L+1).
    """
    # Try same level
    leaf = level_maps.get(L, {}).get((i, j, k))
    if leaf is not None:
        return leaf

    # Try coarser level (parent cell)
    if L > 0:
        parent_idx = (i // 2, j // 2, k // 2)                   # Integer division to find parent
        leaf = level_maps.get(L - 1, {}).get(parent_idx)
        if leaf is not None:
            return leaf

    # Try finer level (children cells)
    child_indices = [
        (i * 2 + di, j * 2 + dj, k * 2 + dk)                    # Create list of child indices
        for di in (0, 1)
        for dj in (0, 1)
        for dk in (0, 1)
    ]
    for child_idx in child_indices:
        leaf = level_maps.get(L + 1, {}).get(child_idx)
        if leaf is not None:
            return leaf

    return None                                                 # No neighbor found

def _split_leaf(newset: dict[tuple[int, int, int], Leaf],
                leaf: Leaf,
                max_level: int):
    """
    Replace `leaf` in `newset` with its 8 children (if level < max_level).
    Returns True if a split occurred, False otherwise.
    """
    if leaf.level >= max_level:
        return False
    L = leaf.level + 1
    i0, j0, k0 = leaf.i*2, leaf.j*2, leaf.k*2

    for di in (0, 1):                                           # Insert child leaves
        for dj in (0, 1):
            for dk in (0, 1):
                child = Leaf(L, i0+di, j0+dj, k0+dk)
                newset[child.key()] = child

    newset.pop(leaf.key(), None)                                # Remove the original parent leaf
    return True

def octree_refine(
        leaves: list[Leaf],
        max_level: int = 3,
        metric: Optional[dict[Leaf, float]] = None,
        threshold: Optional[float] = None,
        enforce_balancing: bool = True,
        ) -> list[Leaf]:
    """
    Split leaves who's metrics exceed the threshold.
    Enforce the 2:1 balancing so neighbor leaf levels differ by at most 1 level.
    Return new leaf list
    """

    new_leaf_set: dict[tuple[int, int, int], Leaf] = {leaf.key(): leaf for leaf in leaves}            # Use a dict to avoid duplicates
    to_split: list[Leaf] = []

    # Stage 1: create split list
    if metric is not None and threshold is not None:            # metric.get(leaf, 0.0) is dict and gets 0.0 as fall-back TODO change to relastic val
        to_split = [leaf for leaf in leaves if leaf.level < max_level and metric.get(leaf, 0.0) >= threshold]

    # Stage 2: split leaves
    for leaf in to_split:
        current = new_leaf_set.get(leaf.key())
        if current is not None:                                 # Leaf might have been split already
            _split_leaf(new_leaf_set, current, max_level)
    
    # Stage 3: check neighbor leaves and ensure 2:1 ratio (4 faces to 1 face max)
    # Loops through leaves indefinitely until each leaf has a 2:1 ratio with neighboring leaves
    if enforce_balancing:       
        changed = True
        while changed:
            changed = False
            leaf_maps = _index_by_level(list(new_leaf_set.values()))
            for key in list(new_leaf_set.keys()):
                lf = new_leaf_set[key]
                L = lf.level
                for (nL, ni, nj, nk) in _leaf_neighbors6(lf):
                    nb = _find_leaf_neighbor(leaf_maps, (nL, ni, nj, nk))
                    if nb is None:
                        continue
                if nb.level + 1 < L:                            # If neighbor is more than 1 level lower, split it
                    if _split_leaf(new_leaf_set, nb, max_level):
                        changed = True
    
    return list(new_leaf_set.values())


### -------------------- DSU -------------------- ###

class DSU:
    """
    Disjoint-set / union-find
    Tracks which voxel belonds to which structure/component.
    Useful for collision between structures, can disable collision within a structure.
    Union sets of voxels if they belong to the same structure.
    """
    def __init__(self, n:int):                                  # n - total number of voxels 
        self.parent = np.arange(n, dtype=np.int64)              # Each node starts as own parent
        self.rank = np.zeros(n, dtype=np.int8)                  # Rank aka height/score used to keep trees shallow when we union sets

    def find_root(self, x:int) -> int:                          # Loops through voxles until it finds it's parent
        p = self.parent         
        while p[x] != x:                                        # While not itself, if it is parent will return itself
            p[x] = p[p[x]]                                      # Point to "grandparent"
            x = p[x]                                            # Move x up a level
        return x                                                # Return highest level parent (captain/root)
    
    def union(self, a: int, b: int) -> None:                    # "bond" the voxels a & b together if they belong to same body
        root_a, root_b = self.find_root(a), self.find_root(b)   # Find the root of each given voxel
        if root_a == root_b:                                    # The rank here is the same, so it's in the same set/body, nothing to do
            return
        if self.rank[root_a] < self.rank[root_b]:               # Compare ranks of voxels
            self.parent[root_a] = root_b                        # Attach the smaller trees to 
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:                                                   # Same rank, pick one (root_a) as the new root and increase its rank by 1
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

### -------------------- Primitive building functions -------------------- ###

# def instantiate_boxes_from_tree(
#         leaves: list[Leaf],
#         origin: np.ndarray,
#         h_base: float,
#         density: float = 1.0,
#         penalty_gain: float = 1e5,
#         static: bool = False):
#     """ Create box_3D for each leaf in the octree. """

#     bodies: list[box_3D] = []
#     #leaf_id_to_body: dict[int, int] = {}
#     leaf_key_to_body_index: dict[tuple[int,int,int,int], int] = {}


#     zeroes = (0.0, 0.0, 0.0)                                    # for zero velocity initializaiton
#     q_id = (1.0, 0.0, 0.0, 0.0)                                 # for quaternion initialization

#     # Loop through grid size, add primitive where occ dictates
#     for leaf_id, lf in enumerate(leaves):
        
#         h = lf.size(h_base)                                     # get box size from leaf level
#         pos = tuple(lf.center(origin, h_base))                  # get primitive positions from centers grid
        
#         b = box_3D(
#             pos=pos,
#             quat=q_id,
#             linear_vel=zeroes,
#             ang_vel=zeroes,
#             density=float(density),
#             penalty_gain=float(penalty_gain),
#             size=(float(h), float(h), float(h)),                # voxelization dictates cube
#             static=bool(static), 
#         )
#         bodies.append(b)
#         #leaf_id_to_body[leaf_id] = lf
#         leaf_key_to_body_index[lf.key()] = len(bodies) - 1

    
#     return bodies, leaf_key_to_body_index


def instantiate_boxes_from_tree(
        leaves: list[Leaf],
        origin: np.ndarray,
        h_base: float,
        density: float = 1.0,
        penalty_gain: float = 1e5,
        static: bool = False,
        show_progress: bool = True):
    """
    Instantiate a box_3D primitive for each octree leaf.
    
    Args:
        leaves: list of Leaf objects
        origin: world-space origin of grid
        h_base: base voxel edge length
        density: material density for each box
        penalty_gain: penalty coefficient
        static: mark boxes as static (infinite mass/inertia)
        show_progress: if True, display tqdm progress bar

    Returns:
        bodies: list[box_3D]
        leaf_key_to_body_index: mapping {leaf.key(): index in bodies}
    """

    bodies: list[box_3D] = []
    leaf_key_to_body_index: dict[tuple[int,int,int,int], int] = {}

    zero_vel = (0.0, 0.0, 0.0)            # zero linear & angular velocity
    quat_id = (1.0, 0.0, 0.0, 0.0)        # identity quaternion

    iterator = trange(len(leaves), desc="Building box_3D primitives") if show_progress else range(len(leaves))
    for leaf_id in iterator:
        lf = leaves[leaf_id]

        h = lf.size(h_base)                       # cube edge length
        pos = tuple(lf.center(origin, h_base))    # cube center

        b = box_3D(
            trans_pos=pos,
            quat_pos=quat_id,
            linear_vel=zero_vel,
            ang_vel=zero_vel,
            density=float(density),
            penalty_gain=float(penalty_gain),
            size=(h, h, h),
            static=bool(static),
        )

        bodies.append(b)
        leaf_key_to_body_index[lf.key()] = len(bodies) - 1

    return bodies, leaf_key_to_body_index
