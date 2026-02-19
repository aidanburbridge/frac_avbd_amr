# OCTREE
# Python standard libraries
from dataclasses import dataclass
import numpy as np
from typing import Optional
from tqdm import trange

# Project specific
from geometry.primitives import box_3D
from geometry.bond_data import BondData

### -------------------- Data structures -------------------- ###
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

    def key(self) -> tuple[int, int, int, int]:
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

_FACE_DIRS = [
    (1, 0, 0),   # +X
    (-1, 0, 0),  # -X
    (0, 1, 0),   # +Y
    (0, -1, 0),  # -Y
    (0, 0, 1),   # +Z
    (0, 0, -1),  # -Z
]

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
                    nb = _find_leaf_neighbor(leaf_maps, nL, ni, nj, nk)
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
        self.parent = np.arange(n, dtype=np.int64)              # Each node starts as own parent - int value used as the assembly ID
        self.rank = np.zeros(n, dtype=np.int8)                  # Rank aka height/score used to keep trees shallow when we union sets

    def find_root(self, x:int) -> int:                          # Loops through voxles until it finds it's parent
        p = self.parent         
        while p[x] != x:                                        # While not itself, if it is parent will return itself
            p[x] = p[p[x]]                                      # Point to "grandparent"
            x = p[x]                                            # Move x up a level
        return x                                                # Return highest level parent (captain/root) - return the assembly ID
    
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
def instantiate_boxes_from_tree(
        leaves: list[Leaf],
        origin: np.ndarray,
        h_base: float,
        density: float = 1.0,
        penalty_gain: float = 1e5,
        static: bool = False,
        show_progress: bool = True,
        valid_mask: Optional[list[bool]] = None):
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

    active_mask: list[bool] = []

    iterator = trange(len(leaves), desc="Building box_3D primitives") if show_progress else range(len(leaves))
    for leaf_id in iterator:
        lf = leaves[leaf_id]

        h = lf.size(h_base)                       # cube edge length
        pos = tuple(lf.center(origin, h_base))    # cube center

        # If highest level parent, set active in mask list, else don't
        is_valid = True if valid_mask is None else bool(valid_mask[leaf_id])
        if lf.level == 0 and is_valid:
            active_mask.append(True)
        else:
            active_mask.append(False)

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

    return bodies, leaf_key_to_body_index, active_mask

### ---------------------- Full Octree w/ Refinement ---------------------- ###

def build_full_hierarchy(
        coarse_occ: np.ndarray,
        max_level: int,
        origin: Optional[np.ndarray] = None,
        h_base: Optional[float] = None,
        contains_fn=None):

    # Instantiate empty return
    nodes: list[Leaf] = []      # list of leaves
    key_to_id: dict[tuple[int, int, int, int], int] = {} # (level, i, j, k) -> node_id
    parent: list[int] = []      # node id to parent id; -1 if coarsest level (root)
    child_count: list[int]= []  # node id to number of children, 0 or 8
    child_start: list[int] = [] # node id to child start in node list, -1 if none
    valid_mask: list[bool] = [] # True if cell center is inside STL (or no contains_fn provided)

    nx, ny, nz = coarse_occ.shape

    def _key(L: int, i: int, j: int, k: int) -> tuple[int, int, int, int]:
        return (L, i, j, k)
    
    def add_node(leaf: Leaf, parent_id: int, is_valid: bool = True) -> int:
        node_id = len(nodes)
        nodes.append(leaf)
        key_to_id[_key(leaf.level, leaf.i, leaf.j, leaf.k)] = node_id
        parent.append(parent_id)
        child_start.append(-1) # default -1, aka no children
        child_count.append(0) # default 0 children
        valid_mask.append(bool(is_valid))
        return node_id
    
    def add_children(parent_id: int):
        """Recursive function to add children"""
        p = nodes[parent_id]
        if p.level >= max_level:
            # Jump out @ max level, no more refinement
            return
        
        Lc = p.level +1
        i0, j0, k0 = 2*p.i, 2*p.j, 2*p.k

        start = len(nodes) # store contiguosly - parent, children, grandchildren etc.
        child_leaves: list[Leaf] = []
        for di in (0,1):
            for dj in (0,1):
                for dk in (0,1):
                    child_leaves.append(Leaf(level=Lc, i=i0+di, j=j0+dj, k=k0+dk))

        if contains_fn is not None:
            if origin is None or h_base is None:
                raise ValueError("origin and h_base must be provided when contains_fn is used.")
            centers = np.array([lf.center(origin, h_base) for lf in child_leaves], dtype=float)
            inside = np.asarray(contains_fn(centers), dtype=bool).reshape(-1)
        else:
            inside = np.ones((len(child_leaves),), dtype=bool)

        for lf, ok in zip(child_leaves, inside):
            add_node(lf, parent_id, is_valid=bool(ok))

        child_start[parent_id] = start
        child_count[parent_id] = 8 # keep ordering/size consistent even if some are invalid

        for offset, ok in enumerate(inside):
            if ok:
                add_children(start + offset)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if bool(coarse_occ[i,j,k]):
                    root = Leaf(level=0,i=i,j=j,k=k)
                    root_id = add_node(root, parent_id=-1, is_valid=True)
                    add_children(root_id)

    # Build neighbor map (same-level, 6-face adjacency)
    level_maps = _index_by_level(nodes)
    neighbor_map = np.full((len(nodes), 6), -1, dtype=int)
    for node_id, lf in enumerate(nodes):
        L, i, j, k = lf.level, lf.i, lf.j, lf.k
        for dir_idx, (di, dj, dk) in enumerate(_FACE_DIRS):
            nb_leaf = level_maps.get(L, {}).get((i + di, j + dj, k + dk))
            if nb_leaf is None:
                continue
            nb_id = key_to_id.get(nb_leaf.key(), -1)
            neighbor_map[node_id, dir_idx] = nb_id

    # Precompute can_refine: True if node has at least one valid child
    can_refine = [False] * len(nodes)
    for node_id in range(len(nodes)):
        start = child_start[node_id]
        count = child_count[node_id]
        if start < 0 or count <= 0:
            continue
        for child_id in range(start, start + count):
            if valid_mask[child_id]:
                can_refine[node_id] = True
                break

    return nodes, key_to_id, parent, child_start, child_count, valid_mask, neighbor_map, can_refine




### -------------------- Constraint building functions -------------------- ###

def build_constraints_from_tree(
        leaves: list[Leaf],
        bodies: list[box_3D],
        leaf_key_to_body_index: dict[tuple[int,int,int,int], int],
        E: float,
        nu: float,
        tensile_strength: float,
        fracture_toughness: float,
        damping_val: float = 0.0,
        damping: float | None = None,
        valid_mask: Optional[list[bool]] = None) -> list[BondData]:
    """
    Creates a list of bonds between adjacent octree voxels using D3Q26 connectivity.
    Axial neighbors get 4 Gauss-point bonds; edge/corner neighbors get single
    truss-style bonds scaled for isotropy.
    """

    # TODO mix quadrature face points and the edges and corners
    if damping is not None:
        damping_val = damping
    
    # Surface check directions (Face neighbors only)
    axial_dirs = [(1,0,0),(0,1,0),(0,0,1),(-1,0,0),(0,-1,0),(0,0,-1)]
    
    bonds: list[BondData] = []
    seen_pairs = set()  # track (min_id, max_id) to prevent duplicates
    
    if valid_mask is None:
        valid_keys = set(lf.key() for lf in leaves)
    else:
        valid_keys = set(lf.key() for lf, ok in zip(leaves, valid_mask) if ok)
    occ = valid_keys
    dsu = DSU(len(bodies))
    level_maps = _index_by_level(leaves)

    shear_ratio = 1.0 / (2.0 * (1.0 + nu))

    # Map squared distance (int) to stiffness scale factor for diagonals
    # 2 (Face diag) -> 0.5
    # 3 (Body diag) -> 0.25
    # Axial (1) handled separately with quadrature
    scale_map = {2: 0.5, 3: 0.25}

    # Convert fracture toughness (K_Ic) to fracture energy Gc
    fracture_energy = (fracture_toughness ** 2 * (1.0 - nu ** 2)) / E

    for lf_idx, lf in enumerate(leaves):

        # Check first if valid leaf
        if valid_mask is not None and not valid_mask[lf_idx]:
            continue
        a_idx = leaf_key_to_body_index.get(lf.key())
        if a_idx is None: 
            continue
            
        L, i, j, k = lf.level, lf.i, lf.j, lf.k
        body_A = bodies[a_idx]
        h = body_A.size[0]

        # 1) Surface check (for collision flags)
        is_surface = any((L, i+di, j+dj, k+dk) not in occ for (di,dj,dk) in axial_dirs)
        body_A.collidable = is_surface

        # 2) Full 26-Neighbor Loop
        # We iterate -1, 0, 1 for all axes to catch every possible neighbor
        for ni in (-1, 0, 1):
            for nj in (-1, 0, 1):
                for nk in (-1, 0, 1):
                    # Skip self
                    if ni == 0 and nj == 0 and nk == 0:
                        continue
                    
                    # Calculate squared distance in grid units (1, 2, or 3)
                    dist_sq_int = ni*ni + nj*nj + nk*nk
                    
                    # If dist_sq_int > 3, it's not a valid 26-neighbor (shouldn't happen with -1..1 loop)
                    if dist_sq_int not in (1, 2, 3):
                        continue

                    nb_leaf = level_maps.get(L, {}).get((i+ni, j+nj, k+nk))


                    if nb_leaf is None:
                        continue
                    if nb_leaf.key() not in valid_keys:
                        continue

                    b_idx = leaf_key_to_body_index.get(nb_leaf.key())
                    if b_idx is None:
                        continue

                    # --- Duplicate Check ---
                    pair_id = (a_idx, b_idx) if a_idx < b_idx else (b_idx, a_idx)
                    if pair_id in seen_pairs:
                        continue

                    seen_pairs.add(pair_id)

                    body_B = bodies[b_idx]
                    dsu.union(a_idx, b_idx)

                    if dist_sq_int == 1:
                        # === AXIAL BOND (Face-to-Face) - STANDARD SINGLE BOND ===
                        
                        # Calculate geometric properties
                        vec_sep = body_B.get_center() - body_A.get_center()
                        axis_idx = int(np.argmax(np.abs(vec_sep)))
                        sgn = 1.0 if vec_sep[axis_idx] >= 0.0 else -1.0

                        hA = 0.5 * np.asarray(body_A.size, dtype=float)
                        hB = 0.5 * np.asarray(body_B.size, dtype=float)

                        # Normal pointing from A to B
                        n_world = np.zeros(3)
                        n_world[axis_idx] = sgn

                        # Face area for stiffness scaling
                        t1i, t2i = [d for d in range(3) if d != axis_idx]
                        face_area = (2.0 * hA[t1i]) * (2.0 * hA[t2i])
                        h_norm = hA[axis_idx] + hB[axis_idx]

                        # Standard Stiffness (E * A / L)
                        G = E / (2.0 * (1.0 + nu))
                        k_n = E * face_area / max(h_norm, 1e-12)
                        k_t = G * face_area / max(h_norm, 1e-12)

                        # Anchor points (Center of the face)
                        pA_local = np.zeros(3)
                        pB_local = np.zeros(3)
                        
                        # Set the component along the axis to the face surface
                        pA_local[axis_idx] = sgn * hA[axis_idx]
                        pB_local[axis_idx] = -sgn * hB[axis_idx]
                        # Tangent components remain 0.0 (Center of face)

                        bonds.append(
                            BondData(
                                idxA=a_idx,
                                idxB=b_idx,
                                pA_local=pA_local,
                                pB_local=pB_local,
                                normal=n_world,
                                k_n=k_n,
                                k_t=k_t,
                                area=face_area,
                                tensile_strength=tensile_strength,
                                fracture_energy=fracture_energy,
                                damp_val=damping_val,
                            )
                        )
                    else:
                        # === DIAGONAL BOND (Edge/Corner) ===
                        scale_factor = scale_map[dist_sq_int]

                        dir_vec = np.array([ni, nj, nk], dtype=float)
                        dist = np.sqrt(float(dist_sq_int))
                        n_world = dir_vec / dist

                        pA_local = n_world * (dist * h * 0.5)
                        pB_local = -(n_world * (dist * h * 0.5))

                        area = (h ** 2) / dist
                        k_base = (E * area) / (dist * h)

                        kn = k_base * scale_factor
                        kt = kn * shear_ratio
                        tensile = tensile_strength * scale_factor

                        bonds.append(
                            BondData(
                                idxA=a_idx,
                                idxB=b_idx,
                                pA_local=pA_local,
                                pB_local=pB_local,
                                normal=n_world,
                                k_n=kn,
                                k_t=kt,
                                area=area,
                                tensile_strength=tensile,
                                fracture_energy=fracture_energy,
                                damp_val=damping_val,
                            )
                        )
                    dsu.union(a_idx, b_idx)
    
    for idx, b in enumerate(bodies):
        b.assembly_id = int(dsu.find_root(idx))

    return bonds


def build_contsraints_from_hierarchy(
        leaves: list[Leaf],
        bodies: list[box_3D],
        leaf_key_to_body_index: dict[tuple[int,int,int,int], int],
        E: float,
        nu: float,
        tensile_strength: float,
        fracture_toughness: float,
        damping_val: float = 0.0,
        valid_mask: Optional[list[bool]] = None,
        max_level: Optional[int] = None) -> list[BondData]:
    
    """
    for each leaf:
        if invalid leaf:
            continue
        
        # same level bonds
        for neighbor:
            build same level bond

        # intermediary bonds
        if level.level < max_level # this leaf should have intermediary bonds to finer neighbors
            get_fine_neighbors
            for f_lf in finer_neighbors 
                build bond (or build_intermediary bond)

        return bond_list
    """
    # Solve for material params TODO should I do this in like a solver setup script?
    fracture_energy = (fracture_toughness ** 2 * (1.0 - nu ** 2)) / E
    G = E / (2.0 * (1.0 + nu))

    # TODO what if max_level doesn't agree with actual max - I suppose need single source of truth passed
    if max_level is None:
        max_level = max(lf.level for lf in leaves) if leaves else 0

    level_maps = _index_by_level(leaves)
    if valid_mask is None:
        valid_mask = [True] * len(leaves)

    valid_keys = set(lf.key() for lf, ok in zip(leaves, valid_mask) if ok)

    bonds: list[BondData] = []
    seen_pairs = set()
    dsu = DSU(len(bodies))

    
    ## ---------- Helpers in Contraint Building Function ---------- ##

    # basic bond building function (used for same level & intermediary level bonds)
    def _build_bond(idxA, idxB, pA, pB, normal, area, len):
        """Build a single bond (1 axial/normal & 2 shear/tangent components)."""
        k_n = E * area / max(len, 1e-12)
        k_t = G * area / max(len, 1e-12)
        return BondData(
            idxA=idxA,
            idxB=idxB,
            pA_local=pA,
            pB_local=pB,
            normal=normal,
            k_n=k_n,
            k_t=k_t,
            area=area,
            tensile_strength=tensile_strength,
            fracture_energy=fracture_energy,
            damp_val=damping_val)
    
    def _build_intermediary_bonds(parent_leaf, parent_idx):
        """Build the L to L+1 bonds between this leaf and neighboring finer cells."""

        if parent_leaf.level >= max_level:
            # Can't have bonds with smaller voxels -- it's already at the finest level 
            return []
        
        body_A = bodies[parent_idx]
        hA = 0.5 * np.asarray(body_A.size, dtype=float)
        parent_center = body_A.get_center()

        new_bonds = []

        L, i, j, k = parent_leaf.level, parent_leaf.i, parent_leaf.j, parent_leaf.k

        for face_dir in _FACE_DIRS:
            # Neighbor leaf at same level
            nb_leaf = level_maps.get(L, {}).get((i + face_dir[0], j + face_dir[1], k + face_dir[2]))
            if nb_leaf is None:
                continue

            opp_face = (-face_dir[0], -face_dir[1], -face_dir[2])

            # Loop over the 4 children on neighbor's face touching this parent
            for di in (0,1):
                for dj in (0,1):
                    for dk in (0,1):

                        # Child on neighbor face (L+1)
                        fine_key = _child_at_face(nb_leaf, opp_face, di, dj, dk)
                        if fine_key is None:
                            continue
                        fine_leaf = level_maps.get(L + 1, {}).get((fine_key[1], fine_key[2], fine_key[3]))
                        if fine_leaf is None:
                            continue
                        if fine_leaf.key() not in valid_keys:
                            continue

                        child_idx = leaf_key_to_body_index.get(fine_leaf.key())
                        if child_idx is None:
                            continue

                        # Avoid duplicates 
                        pair_id = (parent_idx, child_idx) if parent_idx < child_idx else (child_idx, parent_idx)
                        if pair_id in seen_pairs:
                            continue
                        seen_pairs.add(pair_id)

                        body_B = bodies[child_idx]
                        dsu.union(parent_idx, child_idx) # TODO why add this to the DSU? I thought DSU was just for grouping voxels belonging to the same assembly? would this not already happen in the same-level boinding?

                        vec_sep = body_B.get_center() - parent_center
                        axis_idx = int(np.argmax(np.abs(vec_sep))) # longest axis
                        sgn = 1.0 if vec_sep[axis_idx] >= 0.0 else -1.0

                        # Face area
                        hB = 0.5 * np.asarray(body_B.size, dtype=float)
                        # Get tangent axes
                        t1i, t2i = [d for d in range(3) if d != axis_idx]
                        area = (2.0 * hB[t1i]) * (2.0 * hB[t2i])
                        length = hA[axis_idx] + hB[axis_idx]

                        # Bond anchors
                        pA_local = np.zeros(3)
                        pB_local = np.zeros(3)
                        pA_local[axis_idx] = sgn * hA[axis_idx]
                        pB_local[axis_idx] = -sgn * hB[axis_idx]

                        # Tangent offsets (apply to both sides)
                        offset = body_B.get_center() - parent_center
                        pA_local[t1i] = offset[t1i]
                        pA_local[t2i] = offset[t2i]
                        pB_local[t1i] = offset[t1i]
                        pB_local[t2i] = offset[t2i]

                        # Normals
                        n_world = np.zeros(3)
                        n_world[axis_idx] = sgn

                        new_bonds.append(
                            _build_bond(parent_idx, child_idx, pA_local, pB_local, n_world, area, length)
                        )
        return new_bonds


    # TODO make this less UGLY if possible - gotta be a cleaner way to write this
    def _child_at_face(parent_leaf, face_dir, di, dj, dk):
        """Return the child adjacent to a given parent face & quadrant."""
        L, i, j, k = parent_leaf.level, parent_leaf.i, parent_leaf.j, parent_leaf.k
        Lf = L + 1

        if face_dir == (1, 0, 0):     # +X
            return (Lf, 2*i + 1, 2*j + dj, 2*k + dk)
        if face_dir == (-1, 0, 0):    # -X
            return (Lf, 2*i + 0, 2*j + dj, 2*k + dk)
        if face_dir == (0, 1, 0):     # +Y
            return (Lf, 2*i + di, 2*j + 1, 2*k + dk)
        if face_dir == (0, -1, 0):    # -Y
            return (Lf, 2*i + di, 2*j + 0, 2*k + dk)
        if face_dir == (0, 0, 1):     # +Z
            return (Lf, 2*i + di, 2*j + dj, 2*k + 1)
        if face_dir == (0, 0, -1):    # -Z
            return (Lf, 2*i + di, 2*j + dj, 2*k + 0)
        return None

    # ---------- MAIN LOOP ---------- #

    for lf_idx, lf in enumerate(leaves):
        if not valid_mask[lf_idx]:
            continue

        a_idx = leaf_key_to_body_index.get(lf.key())
        if a_idx is None:
            continue
            
        body_A = bodies[a_idx]

        # Make same level bonds
        L, i, j, k = lf.level, lf.i, lf.j, lf.k
        for (di, dj, dk) in _FACE_DIRS:
            nb_leaf = level_maps.get(L, {}).get((i + di, j + dj, k + dk))

            # Check if neighbor leaf exists
            if nb_leaf is None:
                continue
            if nb_leaf.key() not in valid_keys:
                continue

            b_idx = leaf_key_to_body_index.get(nb_leaf.key())
            if b_idx is None:
                continue

            pair_id = (a_idx, b_idx) if a_idx < b_idx else (b_idx, a_idx)
            if pair_id in seen_pairs:
                continue
            seen_pairs.add(pair_id)

            body_B = bodies[b_idx]
            dsu.union(a_idx, b_idx)

            vec_sep = body_B.get_center() - body_A.get_center()
            axis_idx = int(np.argmax(np.abs(vec_sep)))
            sgn = 1.0 if vec_sep[axis_idx] >= 0.0 else -1.0

            hA = 0.5 * np.asarray(body_A.size, dtype=float)
            hB = 0.5 * np.asarray(body_B.size, dtype=float)
            t1i, t2i = [d for d in range(3) if d != axis_idx]
            hx = min(hA[t1i], hB[t1i])
            hy = min(hA[t2i], hB[t2i])
            patch_area = (2.0 * hx) * (2.0 * hy) / 4.0
            length = hA[axis_idx] + hB[axis_idx]

            n_world = np.zeros(3)
            n_world[axis_idx] = sgn

            g = 1.0 / np.sqrt(3.0)
            for s1 in (-g, g):
                for s2 in (-g, g):
                    pA_local = np.zeros(3)
                    pB_local = np.zeros(3)
                    pA_local[axis_idx] = sgn * hA[axis_idx]
                    pB_local[axis_idx] = -sgn * hB[axis_idx]
                    pA_local[t1i] = s1 * hx
                    pA_local[t2i] = s2 * hy
                    pB_local[t1i] = s1 * hx
                    pB_local[t2i] = s2 * hy

                    bonds.append(_build_bond(a_idx, b_idx, pA_local, pB_local, n_world, patch_area, length))

        # intermediary bonds
        if lf.level < max_level:
            bonds.extend(_build_intermediary_bonds(lf, a_idx))

    # assign assembly ids
    for idx, b in enumerate(bodies):
        b.assembly_id = int(dsu.find_root(idx))

    return bonds


# Neighbor lookup for 2:1 refinement assurance
def build_neighbor_map(nodes: list[Leaf], key_to_id: dict[tuple[int, int, int, int], int]) -> np.ndarray:
    """
    Create a (N, 6) array where each row contains the node IDs for all 6 face-face neighbors of node i.
    For nodes of the SAME level.
    """
    # Initialize full neighbor lookup array.
    n_nodes = len(nodes)
    neighbors = np.full((n_nodes, 6), -1, dtype=np.int32)

    dirs = _FACE_DIRS

    for node_idx, leaf in enumerate(nodes):
        L, i, j, k = leaf.level, leaf.i, leaf.j, leaf.k

        for d_idx, (di, dj, dk) in enumerate(dirs):
            nk = (L, i + di, j + dj, k + dk)

            if nk in key_to_id:
                neighbors[node_idx, d_idx] = key_to_id[nk]
    
    return neighbors
