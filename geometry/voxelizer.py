# VOXELIZER
"""
Key Elements:
    - Voxelizer
    - Neighbor Graph Generator
    - Octree Refinement

Requirements:
    - box_3D primitives from primitives.py as "voxels"
    - box_face_vectors from primitives.py
    - numpy
    - trimesh

Voxelizer:
    - STL voxelization to a target number of occupied voxels

Neighbor Graph Generator:
    - Get 6-neighbor graph for adjacent voxels
    - Generates Bond objects with face tagging
    - Collision exclusion for bonded neighbors TODO incorporate this with constraints
    - TODO add edge and corner influence for isotropy

Octree Refinement:
    - Splits voxels at set areas
    - Applies 2:1 balancing with a default mazimum split of 3 levels

"""
# Python standard libraries
import numpy as np
import trimesh
from tqdm import trange


### -------------------- Geometry Voxelization -------------------- ###

class STLVoxelizer:
    """
    Encapsulates all functions involved in STL voxelization.
        - mesh
        - aabb
        - parameters/settings
    """
    def __init__(self,
                 stl_path: str,
                 pad_voxels: int = 1,
                 flood_fill: bool = True,
                 repair: bool = True,
                 phi_guess: float = 0.3):
        
        self.stl_path   = stl_path
        self.pad        = int(pad_voxels)
        self.flood_fill = bool(flood_fill)
        self.repair     = bool(repair)
        self.phi_guess  = float(phi_guess)

        self.mesh       = self._load_mesh(stl_path, repair)
        self.mins, self.maxs = self.mesh.bounds
        self.bbox_size  = self.maxs - self.mins                 # AABB size (dx,dy,dz)
        self.bbox_vol   = float(np.prod(self.bbox_size))        # AABB volume

        # Values from voxelization
        self.h      = None                                      # Current voxel size
        self.origin = None                                      # Origin in global coordinates
        self.occ    = None                                      # Occupancy grid (bool [Nx,Nx,Nz])
        self.n      = None                                      # grid shape (Nx,Ny,Nz)
        self.xs     = None                                      # x centers (Nx,)
        self.ys     = None                                      # y centers (Ny,)
        self.zs     = None                                      # z centers (Nz,)


    # -------------------- Public API -------------------- #

    # TODO maybe make it so I can either give a resolution or a set voxel size h
    def voxelize_to_resolution(
            self,
            target_resolution: int,
            iters: int = 6,
            tol: float = 0.10,
            max_backoff: int = 10):
        """ 
        Voxelize an STL toward a target resolution/number of voxels. 
        Guesses h (voxel size) from the bounding box volume and a resolution target.
        h is corrected over a few iterations to acheive desired resolution.
        """
        
        if self.bbox_vol <=0:
            raise ValueError("AABB not valid, check STL dimensions/units.")
        
        # Guess initial h value with fill-fraction (phi)
        phi = max(self.phi_guess, 1e-6)
        h = (self.bbox_vol / max(target_resolution / phi, 1.0)) ** (1.0 / 3.0)

        # First make sure num voxels isn't zero
        backoff = 0
        while True:
            occ, origin, (xs,ys,zs) = self._voxelize_with_h(h)
            num_occ = int(occ.sum())
            if num_occ > 0:
                break
            if backoff >= max_backoff:
                self._set_state(h, origin, occ, xs, ys, zs)     # Update state for debugging
                return self.occ, self.origin, self.h            
            h *= 0.5                                            # Reduce h size to try and fit inside mesh
            backoff += 1
        
        # Move h toward target resolution
        for _ in range(max(1, iters)):
            num_occ = int(occ.sum())
            if abs(num_occ - target_resolution) / max(target_resolution, 1.0) <= tol:
                break
            scale = (num_occ / float(target_resolution)) ** (1.0 / 3.0)
            scale = float(np.clip(scale, 0.5, 2.0))
            h *= scale
            occ, origin, (xs, ys, zs) = self._voxelize_with_h(h)
        
        self._set_state(h, origin, occ, xs, ys, zs)
        return self.occ, self.origin, self.h

    def voxelize_to_h(self, h: float):
        """
        Voxelize once to a fixed voxel size h.
        """

        occ, origin, (xs,ys,zs) = self._voxelize_with_h(float(h))
        self._set_state(float(h), origin, occ, xs, ys, zs)
        return self.occ, self.origin, self.h
    
    
    # -------------------- Core Helpers -------------------- #

    def _voxelize_with_h(self, hval: float):
        """ Build occ grid for a given voxel size h, sample center-in-solid. Flood fill optional. """
        n, origin, (xs, ys, zs) = self._grid_from_aabb(self.mins, self.maxs, hval, self.pad)
        occ = self._contains_centers(self.mesh, xs, ys, zs)

        if self.flood_fill:
            outside = self._flood_fill_outside(~occ)
            occ = ~outside
        return occ, origin, (xs,ys,zs)
    
    @staticmethod
    def _grid_from_aabb(aabb_mins: np.ndarray, aabb_maxs: np.ndarray, h: float, pad_voxels: int):
        """
        Compute grid shape n=(Nx,Ny,Nz), origin, and 1D center arrays for size h.
        Adds pad_voxels of empty space around the AABB for robust flood-fill.
        Grid is centered on the AABB midpoint to preserve geometric symmetry.

        Input:
            - aabb_mins: min values of mesh AABB
            - aabb_mins: max values of mesh AABB
            - h: size of each voxel
            - pad_voxel: amount of padding to add
        """
        size = np.asarray(aabb_maxs) - np.asarray(aabb_mins)        # size of the mesh's AABB (width, height, depth)

        # Number of cells along each axis
        n = np.ceil(size / h).astype(int)                           # Number of voxels that fit along each axis
        n = np.maximum(n, 1)                                        # Clamp to at least 1
        n = n + 2 * pad_voxels                                      # Add safety padding
        midpoint = 0.5 * (np.asarray(aabb_mins) + np.asarray(aabb_maxs))
        origin = midpoint - 0.5 * n.astype(float) * h               # world coordinate of the (0,0,0) voxel corner
        
        # Centers
        x_centers = origin[0] + (np.arange(n[0]) + 0.5) * h         # x-coordinates of voxel centers
        y_centers = origin[1] + (np.arange(n[1]) + 0.5) * h         # y-coordinates of voxel centers
        z_centers = origin[2] + (np.arange(n[2]) + 0.5) * h         # z-coordinates of voxel centers

        return n, origin, (x_centers, y_centers, z_centers)         # grid shape, world origin of grid, and 1D arrays of center positions along each axis

    @staticmethod
    def _contains_centers(mesh: trimesh.Trimesh, x_centers: np.ndarray, y_centers: np.ndarray, z_centers: np.ndarray) -> np.ndarray:
        """ 
        Evaluate if the voxel center is inside or outside the mesh using trimesh.
        Called center-in-solid test.
        """
        X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])    # Stack unraveled vectors as columns (N, 3)
        #inside = mesh.contains(pts)                                 # boolean (N,)
        inside = _contains_points_chunked(mesh, pts, chunk=200_000)  # with progress

        occ = inside.reshape(X.shape)                               # occ: occupancy grid - grid of booleans (center lies inside mesh) - all have same shape
        return occ
    
    @staticmethod
    def _flood_fill_outside(empty_mask: np.ndarray) -> np.ndarray:
        """
        Use to fill internal voids.
        Label all empty cells connected to the grid boundary as 'outside'.
        What remains unlabeled is trapped interior should be solid, so fill it and make it solid.
        Do not use for hollow shells or models with internal cavities.
        """
        from collections import deque                               # "fast queue" - like a fast list
        nx, ny, nz = empty_mask.shape                               # Num voxels on x,y,z
        outside = np.zeros_like(empty_mask, dtype=bool)             # empty_mask is a 3D boolean grid where True is an empty voxel
        q = deque()

        def push(i,j,k):                                            # add valid voxels to the queue for expansion
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz and empty_mask[i,j,k] and not outside[i,j,k]:
                outside[i,j,k] = True
                q.append((i,j,k))
        
        # "seed" boundary faces, 6 boundary faces of the grid 
        for j in range(ny):         # -X and +X faces
            for k in range(nz):
                push(0, j, k)
                push(nx-1, j, k)
        for i in range(nx):         # -Y and +Y faces
            for k in range(nz):
                push(i, 0, k)
                push(i, ny-1, k)                 
        for i in range(nx):         # -Z and +Z faces
            for j in range(ny):
                push(i, j, 0)
                push(i, j, nz-1)        

        # Now spread from boundary faces to neighboring cells, "pushing" eac voxel to build "outside" grid
        neighbors = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

        while q:
            i,j,k = q.popleft()
            for di,dj,dk in neighbors:
                push(i+di, j+dj, k+dk)

        return outside

    @staticmethod
    def _load_mesh(path: str, repair: bool) -> trimesh.Trimesh:
        """ Load STL and optionally repair it. """

        mesh = trimesh.load(path, force='mesh')

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("The provided file is not a valid STL file.")
        
        if repair:
            try:
                mesh.remove_duplicate_faces()
                mesh.remove_degenerate_faces()
                mesh.remove_unreferenced_vertices()
                mesh.process()
                trimesh.repair.fill_holes(mesh)
            except Exception:
                pass
        return mesh
    
    def _set_state(self, h: float, origin: np.ndarray, occ: np.ndarray,
                   xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
        
        self.h      = float(h)
        self.origin = origin
        self.occ    = occ.astype(bool, copy=True)
        self.n      = self.occ.shape
        self.xs     = xs
        self.ys     = ys
        self.zs     = zs
    
    # -------------------- Getters -------------------- #
    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Return arrays of voxel center coordinates. """
        if self.xs is None:
            raise RuntimeError("No voxels generated yet.")
        return self.xs, self.ys, self.zs
    
    def stats(self) -> dict:
        """ Stats dictionary: counts, grid shape, h, origin. """
        if self.occ is None:
            raise RuntimeError("No voxels generated yet.")
        return {
            "voxels": int(self.occ.sum()),
            "grid_shape": tuple(self.occ.shape),
            "h": float(self.h),
            "origin": np.asarray(self.origin, float).copy(),
        }


### -------------------- Extra Helpers -------------------- ###
def _contains_points_chunked(mesh, pts: np.ndarray, chunk: int = 200_000, show_progress: bool = True) -> np.ndarray:
    """
    Fast + memory-safe mesh.contains with optional progress.
    pts: (N,3) float array
    """
    N = len(pts)
    out = np.empty(N, dtype=bool)

    if show_progress:
        iterator = trange(0, N, chunk, desc="Ray tests (mesh.contains)")
    else:
        iterator = range(0, N, chunk)

    for s in iterator:
        e = min(s + chunk, N)
        out[s:e] = mesh.contains(pts[s:e])
    return out
