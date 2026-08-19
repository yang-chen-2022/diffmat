import numpy as np

def generate_random_particles(
        n_particles,
        box_size,
        radius_range,
        max_attempts=100000):
    """
    Generate non-overlapping spherical particles with random radii.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    box_size : tuple
        (Lx, Ly, Lz)
    radius_range : tuple
        (r_min, r_max)
    max_attempts : int

    Returns
    -------
    positions : ndarray (N, 3)
    radii : ndarray (N,)
    """

    Lx, Ly, Lz = box_size
    r_min, r_max = radius_range

    positions = []
    radii = []

    attempts = 0

    while len(positions) < n_particles:

        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                f"Could only place {len(positions)} particles."
            )

        # Random radius
        r = np.random.uniform(r_min, r_max)

        # Random center inside box
        candidate = np.array([
            np.random.uniform(r, Lx - r),
            np.random.uniform(r, Ly - r),
            np.random.uniform(r, Lz - r)
        ])

        if len(positions) == 0:
            positions.append(candidate)
            radii.append(r)
            continue

        pos = np.asarray(positions)
        rad = np.asarray(radii)

        # Distances to existing particles
        dist = np.linalg.norm(pos - candidate, axis=1)

        # No overlap condition
        if np.all(dist >= (rad + r)):
            positions.append(candidate)
            radii.append(r)

    return np.asarray(positions), np.asarray(radii)



def generate_particles_periodic(
        n_particles,
        box_size,
        radius_range,
        min_gap=0.0,
        max_attempts=100000):

    Lx, Ly, Lz = box_size
    box = np.array([Lx, Ly, Lz])

    positions = []
    radii = []

    attempts = 0

    while len(positions) < n_particles:

        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                f"Could only place {len(positions)} particles"
            )

        r = np.random.uniform(*radius_range)

        # periodic => centre can be anywhere
        candidate = np.array([
            np.random.uniform(0.0, Lx),
            np.random.uniform(0.0, Ly),
            np.random.uniform(0.0, Lz)
        ])

        if len(positions) == 0:
            positions.append(candidate)
            radii.append(r)
            continue

        pos = np.asarray(positions)
        rad = np.asarray(radii)

        dr = candidate - pos
        dr = minimum_image(dr, box)

        dist2 = np.sum(dr**2, axis=1)

        overlap = dist2 < (rad + r + min_gap)**2

        if not np.any(overlap):
            positions.append(candidate)
            radii.append(r)

    return np.asarray(positions), np.asarray(radii)


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

    voxel_centers = grid.voxel_centers
    n_voxels = voxel_centers.shape[0]

    labels = np.zeros(n_voxels, dtype=np.int32)

    for pid, (center, radius) in enumerate(
            zip(particle_centers, particle_radii), start=1):

        dist2 = np.sum((voxel_centers - center)**2, axis=1)

        inside = dist2 <= radius**2

        labels[inside] = pid

    return labels.reshape(grid.nx, grid.ny, grid.nz)



def voxelise_particles_periodic(grid, positions, radii):

    centres = grid.voxel_centers

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



# ============================================================================
# Material Properties for Phase-Field Fracture Model
# ============================================================================
import jax.numpy as jnp
def init_material(
        matID,
        lmbda_list, 
        mu_list, 
        gc_list, 
        lc_list,
        dtype=jnp.float64,
        ):

    nx, ny, nz = matID.shape

    lmbda_grid = jnp.zeros((nx, ny, nz), dtype=dtype)
    mu_grid = jnp.zeros((nx, ny, nz), dtype=dtype)
    gc_grid = jnp.zeros((nx, ny, nz), dtype=dtype)
    lc_grid = jnp.zeros((nx, ny, nz), dtype=dtype)
    
    num_mats = len(lmbda_list)
    matids = np.unique(matID)
    
    for i in range(num_mats):
        lmbda_grid = lmbda_grid.at[matID == matids[i]].set(lmbda_list[i])
        mu_grid = mu_grid.at[matID == matids[i]].set(mu_list[i])
        gc_grid = gc_grid.at[matID == matids[i]].set(gc_list[i])
        lc_grid = lc_grid.at[matID == matids[i]].set(lc_list[i])
    
    return lmbda_grid, mu_grid, gc_grid, lc_grid

