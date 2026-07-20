from jax import numpy as jnp
import numpy as np

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


def minimum_image(dr, box):
    """
    Apply minimum-image convention.

    Parameters
    ----------
    dr : (...,3) ndarray
        Separation vectors.
    box : array-like (3,)

    Returns
    -------
    (...,3) ndarray
    """
    box = np.asarray(box)
    return dr - box * np.round(dr / box)


def assign_voxels_to_particles(grid, particle_centers, particle_radii):
    """
    Parameters
    ----------
    grid : grid_spec
    particle_centers : (Np, 3) ndarray
    particle_radii : (Np,) ndarray

    Returns
    -------
    labels : (nx, ny, nz) ndarray
        0 = matrix
        k = particle index + 1
    """

    voxel_centers = grid.build_centers()
    n_voxels = voxel_centers.shape[0]

    labels = np.zeros(n_voxels, dtype=np.int32)

    for pid, (center, radius) in enumerate(
            zip(particle_centers, particle_radii), start=1):

        dist2 = np.sum((voxel_centers - center)**2, axis=1)

        inside = dist2 <= radius**2

        labels[inside] = pid

    return labels.reshape(grid.nx, grid.ny, grid.nz)



def voxelise_particles_periodic(grid, positions, radii):

    centres = grid.build_centers()

    box = np.array([grid.Lx, grid.Ly, grid.Lz])

    dr = centres[:, None, :] - positions[None, :, :]

    dr = minimum_image(dr, box)

    dist2 = np.sum(dr**2, axis=-1)

    occupied = np.any(
        dist2 <= radii[None, :]**2,
        axis=1
    )

    return occupied.reshape(
        grid.nx,
        grid.ny,
        grid.nz
    ).astype(np.uint8)



def initialise_materials_phasefield(
    grid, matID, lmbda, mu, gc, lc, dtype=np.float32
):
    """
    Initialize material property grids for phase-field fracture simulation.
    
    Args:
        grid: Computational grid specification
        matID: Material ID map for each voxel
        lmbda: Lame parameter (list, one per material)
        mu: Shear modulus (list, one per material)
        gc: Critical energy release rate (list, one per material)
        lc: Characteristic length scale (list, one per material)
        dtype: Data type for arrays
        
    Returns:
        Tuple of (lmbda_grid, mu_grid, gc_grid, lc_grid)
    """
    lmbda_grid = np.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
    mu_grid = np.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
    gc_grid = np.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
    lc_grid = np.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
    
    num_mats = len(lmbda)
    matids = np.unique(matID)
    
    if num_mats != len(matids):
        raise ValueError(
            "Number of material property entries must match the number "
            "of unique material IDs in the material map."
        )
    
    for i in range(num_mats):
        lmbda_grid[matID == matids[i]] = lmbda[i]
        mu_grid[matID == matids[i]] = mu[i]
        gc_grid[matID == matids[i]] = gc[i]
        lc_grid[matID == matids[i]] = lc[i]
    
    return jnp.array(lmbda_grid), jnp.array(mu_grid), gc_grid, lc_grid







def voigt_to_tensor(v):
    """Convert 6-component Voigt notation (6, Nx, Ny, Nz) to 3x3 symmetric tensor (Nx, Ny, Nz, 3, 3)."""
    v0, v1, v2, v3, v4, v5 = v[0], v[1], v[2], v[3], v[4], v[5]
    
    row0 = jnp.stack([v0, v5, v4], axis=-1) #slots into the end
    row1 = jnp.stack([v5, v1, v3], axis=-1) 
    row2 = jnp.stack([v4, v3, v2], axis=-1)
    
    return jnp.stack([row0, row1, row2], axis=-2) #


def tensor_to_voigt(t):
    """Convert 3x3 symmetric tensor (Nx, Ny, Nz, 3, 3) back to 6-component Voigt (6, Nx, Ny, Nz)."""
    v0 = t[..., 0, 0]
    v1 = t[..., 1, 1]
    v2 = t[..., 2, 2]
    v3 = t[..., 1, 2]
    v4 = t[..., 0, 2]
    v5 = t[..., 0, 1]
    
    return jnp.stack([v0, v1, v2, v3, v4, v5], axis=0)



