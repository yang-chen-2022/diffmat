
import numpy as np
from scipy.spatial import ConvexHull



# ============================================================
# utilities
# ============================================================

def normalize(v, eps=1e-12):
    norm = np.sqrt( np.sum(v * v, axis=-1, keepdims=True))
    norm = np.where(norm < eps, eps, norm)  # avoid division by zero
    return v / norm



def order_polygon_vertices(X, planarity_tol=1e-10):
    """
    Order the vertices of polygon by polar angle

    Parameters
    ----------
    X : (npts, 3) ndarray
        Polygon vertices in 3D (unordered).
    planarity_tol : float, optional
        Tolerance for planarity check.

    Returns
    -------
    X : float
        Ordered vertices
    """

    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X must have shape (npts, 3)")

    npts = X.shape[0]
    if npts < 3:
        return 0.0

    # ------------------------------------------------------------
    # 1. Center the points
    # ------------------------------------------------------------
    centroid = X.mean(axis=0)
    Xc = X - centroid

    # ------------------------------------------------------------
    # 2. Find best-fit plane using SVD
    # ------------------------------------------------------------
    _, S, Vt = np.linalg.svd(Xc)

    # Smallest singular value corresponds to plane normal
    normal = Vt[-1]

    # Planarity check (optional but recommended)
    distances = Xc @ normal
    if np.max(np.abs(distances)) > planarity_tol:
        raise ValueError("Points are not planar within tolerance")

    # ------------------------------------------------------------
    # 3. Build 2D coordinates in the plane
    # ------------------------------------------------------------
    e1 = Vt[0]  # first in-plane direction
    e2 = Vt[1]  # second in-plane direction

    x2d = Xc @ e1
    y2d = Xc @ e2

    # ------------------------------------------------------------
    # 4. Order vertices by polar angle
    # ------------------------------------------------------------
    angles = np.arctan2(y2d, x2d)
    order = np.argsort(angles)

    return X[order]


def polygon_area_3d(X, planarity_tol=1e-10):
    """
    Compute the area of a planar polygon in 3D with unordered vertices.

    Parameters
    ----------
    X : (npts, 3) ndarray
        Polygon vertices in 3D (unordered).
    planarity_tol : float, optional
        Tolerance for planarity check.

    Returns
    -------
    area : float
        Area of the polygon.
    """

    # order the vertices
    X = order_polygon_vertices(X, planarity_tol=1e-10)

    # compute area using 3D cross-product formula
    X_next = np.roll(X, -1, axis=0)
    cross_sum = np.sum(np.cross(X, X_next), axis=0)

    return 0.5 * np.linalg.norm(cross_sum)


####
####
####

class fibre:
    '''
    :arg x0: (K,3) fibre axis reference point
    :arg v:  (K,3) fibre direction
    :arg r:  (K,) fibre radius
    '''

    def __init__(self, x0, v, r):
        self.x0 = np.array(x0)
        self.r = np.array(r)
        v = np.array(v)
        self.v = v / np.sqrt( np.sum(v * v, axis=-1, keepdims=True) )

class grid_spec:
    def __init__(self, Lx, Ly, Lz, dx, dy, dz):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.nx = int(Lx/dx)
        self.ny = int(Ly/dy)
        self.nz = int(Lz/dz)

    def build_centers(self):
        x = np.linspace(self.dx/2, self.Lx - self.dx/2, self.nx)
        y = np.linspace(self.dy/2, self.Ly - self.dy/2, self.ny)
        z = np.linspace(self.dz/2, self.Lz - self.dz/2, self.nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        return np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
        

def fibre_association(centers, h_vec, x0, v, r):
    # --------------------------------------------------------
    # voxel vertices
    # --------------------------------------------------------
    VERT = np.array([
        [-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1],
        [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1],
    ]) * 0.5

    V = centers[:,None,:] + VERT[None,:,:] * h_vec

    # --------------------------------------------------------
    # fibre association
    # --------------------------------------------------------
    dx = centers[:,None,:] - x0[None,:,:]
    proj = np.sum(dx * v[None,:,:], axis=-1, keepdims=True) * v[None,:,:]
    dist = np.linalg.norm(dx - proj, axis=-1)

    idx = np.argmin(dist, axis=1)

    x0c = x0[idx]
    vc  = v[idx]
    rc  = r[idx]

    # --------------------------------------------------------
    # signed level-set at vertices based on distance to fibre center line
    # --------------------------------------------------------
    dxv = V - x0c[:,None,:]
    projv = np.sum(dxv * vc[:,None,:], axis=-1, keepdims=True) * vc[:,None,:]
    Dv = np.linalg.norm(dxv - projv, axis=-1)

    phi = Dv - rc[:,None]

    inside = phi <= 0

    all_in  = np.all(inside, axis=1)
    all_out = np.all(~inside, axis=1)
    composite = ~(all_in | all_out)

    return all_in, all_out, composite, idx

# ============================================================
# plane reconstruction from fibre geometry
# ============================================================

def reconstruct_plane(centers, x0, v, radius):
    """
    centers : (N,3) voxel centers
    x0      : (K,3) fibre axis reference point
    v       : (K,3) fibre direction
    radius  : (K,) fibre radius

    returns:
    x_int   : (N,3) interface point on plane 
    n       : (N,3) plane normal
    """
    
    v = normalize(v)

    # local normal to fibre axis (plane normal)
    dx = centers[:,None,:] - x0[None,:,:]
    proj = np.sum(dx * v[None,:,:], axis=-1, keepdims=True) * v[None,:,:]
    xp = x0[None,:,:] + proj
    dist = np.linalg.norm(dx - proj, axis=-1)

    idx = np.argmin(dist, axis=1)

    xp = np.take_along_axis(xp, idx[:,None,None], axis=1).squeeze(axis=1)  # (N,3)
    r = radius[idx]  # (N,)
    dist = np.take_along_axis(dist, idx[:,None], axis=1).squeeze(axis=1)  # (N,)

    n = normalize(centers - xp)  # (N,3)

    # interface point
    x_int = xp + r[:,None] * n

    return x_int, n



def level_set(x_int, n, x):
    '''
    Level set function for any point x in a voxel as phi = (x - x_int) . n 
    x_int   : (N,3) interface point on plane
    n       : (N,3) plane normal
    x       : (N,M,3) points at which to evaluate the level set
    '''
    
    return np.sum((x - x_int[:,None,:]) * n[:,None,:], axis=-1)


def get_level_set_at_points(centers, h_vec, x_int, n, VERT=None):
    if VERT is None:
        VERT = np.array([
            [-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1],
            [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1],
        ]) * 0.5
    else:
        VERT = np.array(VERT)

    # Voxel vertices
    V = centers[:,None,:] + VERT[None,:,:] * h_vec
    phi = level_set(x_int, n, V)

    return phi


# ============================================================
# polyhedron vertex (voxel cut be plane)
# ============================================================

def get_polyhedron_vertices(centerv, h_vec, phiv):
    # Voxel vertices
    Vv = np.array([
        [-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1],
        [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1],
    ]) * 0.5 * h_vec[None,:] + centerv[None,:]  # (8,3)

    EDGES = np.array([
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ])

    # intersection points on edges
    i0 = EDGES[:,0]
    i1 = EDGES[:,1]

    denom = phiv[i0] - phiv[i1]
    denom = np.where(np.abs(denom)<1e-12, 1e-12, denom)
    tv = phiv[i0] / denom
    
    idx = (tv > 0) & (tv < 1)

    X = Vv[i0][idx] + tv[idx,None]*(Vv[i1][idx] - Vv[i0][idx])

    #

    # vertex points inside fibre
    infibre = Vv[phiv <= 0]

    return X, infibre


# ============================================================
# clip voxel by plane and compute intersection area and volume
# ============================================================

def voxel_clipping(centers, h_vec, phi):

    def process_voxel(centerv, phiv):
        # intersection points and vertices inside fibre
        X, infibres = get_polyhedron_vertices(centerv, h_vec, phiv)

        # intersection areas (from cut edges only)
        area = polygon_area_3d(X)

        # build convex boundary point set
        pts = np.concatenate([X, infibres], axis=0)
        hull = ConvexHull(pts)
        volume = hull.volume

        return volume, area
        
    vols = []
    areas = []
    for centerv, phiv in zip(centers, phi):
        volume, area = process_voxel(centerv, phiv)
        vols.append(volume)
        areas.append(area)

    vols = np.array(vols)
    areas = np.array(areas)

    return vols, areas



