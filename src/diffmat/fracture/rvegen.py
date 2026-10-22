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


def generate_voids_periodic(grid, particle_mask, void_fraction, mean_diameter, std_diameter=0.0):
    """
    Add random periodic spherical voids in the matrix until the void volume
    fraction (of the whole RVE) reaches `void_fraction` (slightly overshoots).
    Void voxels inside particles are cut out, so particle shapes stay intact.
    Diameters ~ normal(mean_diameter, std_diameter), clipped to one voxel.

    Returns
    -------
    mask : (nx, ny, nz) bool ndarray
    """
    centres = grid.voxel_centers
    box = np.array([grid.Lx, grid.Ly, grid.Lz])
    min_diameter = min(grid.Lx / grid.nx, grid.Ly / grid.ny, grid.Lz / grid.nz)

    matrix = ~np.asarray(particle_mask, dtype=bool).ravel()
    if void_fraction > matrix.mean():
        raise ValueError("void_fraction exceeds the matrix volume fraction")

    mask = np.zeros(centres.shape[0], dtype=bool)
    n_voids = 0

    while mask.mean() < void_fraction:
        diameter = max(np.random.normal(mean_diameter, std_diameter), min_diameter)
        centre = np.random.uniform(0.0, 1.0, size=3) * box
        dr = minimum_image(centres - centre, box)
        mask |= (np.sum(dr**2, axis=1) <= (0.5 * diameter) ** 2) & matrix
        n_voids += 1
        if n_voids > 100000:
            raise RuntimeError("Could not reach the target void fraction")

    return mask.reshape(grid.nx, grid.ny, grid.nz)


def generate_brick_mortar_rve(
        brick_length,
        brick_width,
        mortar_thickness,
        n_rep=(1, 1),
        voxel_size=None,
        resolution=None):
    """
    2D brick-and-mortar RVE (running bond), voxelised on a regular grid.

    Unit cell (periodic), size Lx = L + t by Ly = 2 (W + t)
    (bricks = [ ], mortar = blank space, t = mortar thickness):
         ________    ________ 
        |________|  |________|
        |  ________________  |
        | |      brick     | |
        | |________________| |
        |________    ________|
        |________|  |________|

    Parameters
    ----------
    brick_length, brick_width : float
        L (x direction) and W (y direction) of a full brick.
    mortar_thickness : float
        t, mortar thickness between bricks.
    n_rep : int or (int, int)
        Number of unit-cell repetitions in x and y.
    voxel_size : float, optional
        Voxel edge length. Alternative to `resolution`.
    resolution : (int, int), optional
        Total number of voxels (nx, ny). Must be divisible by n_rep.
        Exactly one of voxel_size / resolution must be given.

    Returns
    -------
    matID : (nx, ny) uint8 ndarray
        0 = mortar, 1 = brick
    (Lx, Ly) : tuple of float
        Overall RVE size.
    """
    L, W, t = float(brick_length), float(brick_width), float(mortar_thickness)
    if L <= 0 or W <= 0 or t < 0:
        raise ValueError("brick_length and brick_width must be > 0, mortar_thickness >= 0")
    nrx, nry = (n_rep, n_rep) if np.isscalar(n_rep) else n_rep
    if nrx < 1 or nry < 1:
        raise ValueError("n_rep must be >= 1")
    if (voxel_size is None) == (resolution is None):
        raise ValueError("Specify exactly one of voxel_size or resolution")

    cx, cy = L + t, 2.0 * (W + t)
    Lx, Ly = nrx * cx, nry * cy
    if resolution is None:
        nx, ny = int(round(Lx / voxel_size)), int(round(Ly / voxel_size))
    else:
        nx, ny = resolution
    if nx % nrx or ny % nry:
        raise ValueError("Resolution must be divisible by n_rep (grid does not match the unit cell)")

    # Voxel centres folded into one unit cell
    x = (np.arange(nx) + 0.5) * Lx / nx % cx
    y = (np.arange(ny) + 0.5) * Ly / ny % cy
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Middle row: brick centred in x, spanning y in [cy/2 - W/2, cy/2 + W/2]
    mid = (np.abs(X - cx / 2) <= L / 2) & (np.abs(Y - cy / 2) <= W / 2)

    # Edge rows: brick centred at the cell corner (periodic), split into
    # two half bricks in x and two half-height strips in y.
    dx = np.minimum(X, cx - X)
    dy = np.minimum(Y, cy - Y)
    edge = (dx <= L / 2) & (dy <= W / 2)

    return (mid | edge).astype(np.uint8), (Lx, Ly)



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
    
    # List index i corresponds to matID value i (0 matrix, 1 particle, 2 void).
    for i in range(len(lmbda_list)):
        lmbda_grid = lmbda_grid.at[matID == i].set(lmbda_list[i])
        mu_grid = mu_grid.at[matID == i].set(mu_list[i])
        gc_grid = gc_grid.at[matID == i].set(gc_list[i])
        lc_grid = lc_grid.at[matID == i].set(lc_list[i])
    
    return lmbda_grid, mu_grid, gc_grid, lc_grid

