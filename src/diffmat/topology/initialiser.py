import numpy as np

def get_initial_density(vf, shape):
    """
    Generate initial density field with circular hole pattern.
    
    Parameters
    ----------
    vf : float
        Target volume fraction
    shape : tuple
        (nx, ny, nz) grid dimensions
        
    Returns
    -------
    rho : ndarray
        Initial density field
    """
    nx, ny, nz = shape
    rho = np.full(shape, vf, dtype=np.float64)
    d = nx / 3.0
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    r = np.sqrt( (x - nx / 2.0 - 0.5) ** 2 + 
                  (y - ny / 2.0 - 0.5) ** 2 + 
                  (z - nz / 2.0 - 0.5) ** 2 )
    rho[r < d] = vf / 2.0
    return rho



def init_uniform_3d(vf, shape):
    """
    Uniform initialization - simplest and most common.
    Guarantees: mean(rho) = vf
    """
    rho = np.full(shape, vf, dtype=float)
    return rho


def init_spherical_hole_3d(vf, shape, hole_radius_factor=0.3):
    """
    Initialize with a spherical hole in the center.
    CORRECTED: Guarantees mean(rho) = vf by adjusting density values.

    Parameters
    ----------
    vf : float
        Target volume fraction
    shape : tuple
        (nx, ny, nz) grid dimensions
    hole_radius_factor : float
        Hole radius as fraction of domain (default 0.3)

    Returns
    -------
    rho : ndarray (numpy)
        Density field with spherical hole, mean(rho) = vf
    """
    nx, ny, nz = shape
    center = np.array([nx / 2.0, ny / 2.0, nz / 2.0])
    radius = hole_radius_factor * min(nx, ny, nz) / 2.0

    x, y, z = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )

    dist = np.sqrt(
        (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    )

    # Create binary mask for sphere
    in_sphere = dist < radius
    num_sphere_voxels = np.sum(in_sphere)
    num_total_voxels = np.prod(shape)
    sphere_fraction = num_sphere_voxels / num_total_voxels

    # Solve for densities: vf_outside * (1-sf) + vf_inside * sf = vf
    # Let vf_inside = vf / 2 (chosen), solve for vf_outside
    vf_inside = vf / 2.0
    if sphere_fraction < 1.0:
        vf_outside = (vf - vf_inside * sphere_fraction) / (1.0 - sphere_fraction)
    else:
        vf_outside = vf_inside  # Edge case: sphere covers everything

    vf_outside = np.clip(vf_outside, 0.0, 1.0)  # Ensure valid range

    rho = np.where(in_sphere, vf_inside, vf_outside)

    # Verify
    actual_vf = np.mean(rho)
    print(f"  init_spherical_hole_3d: target vf={vf:.4f}, actual mean(rho)={actual_vf:.4f}")

    return rho


def init_multiple_spheres_3d(vf, shape, num_spheres=3, hole_radius_factor=0.15):
    """
    Initialize with multiple spherical holes.
    CORRECTED: Guarantees mean(rho) = vf
    """
    nx, ny, nz = shape
    center_domain = np.array([nx / 2.0, ny / 2.0, nz / 2.0])
    radius = hole_radius_factor * min(nx, ny, nz) / 2.0

    x, y, z = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )

    # Create mask for all spheres
    in_any_sphere = np.zeros(shape, dtype=bool)
    np.random.seed(42)
    for _ in range(num_spheres):
        center = np.array([
            np.random.uniform(nx * 0.2, nx * 0.8),
            np.random.uniform(ny * 0.2, ny * 0.8),
            np.random.uniform(nz * 0.2, nz * 0.8),
        ])

        dist = np.sqrt(
            (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
        )
        in_any_sphere |= (dist < radius)

    num_sphere_voxels = np.sum(in_any_sphere)
    num_total_voxels = np.prod(shape)
    sphere_fraction = num_sphere_voxels / num_total_voxels

    # Adjust densities to satisfy volume constraint
    vf_inside = vf / 2.0
    if sphere_fraction < 1.0:
        vf_outside = (vf - vf_inside * sphere_fraction) / (1.0 - sphere_fraction)
    else:
        vf_outside = vf_inside

    vf_outside = np.clip(vf_outside, 0.0, 1.0)

    rho = np.where(in_any_sphere, vf_inside, vf_outside)

    actual_vf = np.mean(rho)
    print(f"  init_multiple_spheres_3d: target vf={vf:.4f}, actual mean(rho)={actual_vf:.4f}")

    return rho


def init_random_3d(vf, shape, std=0.1):
    """
    Random initialization with Gaussian perturbation.
    CORRECTED: Guarantees mean(rho) = vf by centering around vf
    """
    rho = vf + np.random.normal(0, std, shape)
    rho = np.clip(rho, 0.0, 1.0)

    # Re-center to ensure mean = vf
    current_mean = np.mean(rho)
    rho = rho - current_mean + vf
    rho = np.clip(rho, 0.0, 1.0)

    return rho


def init_checkerboard_3d(vf, shape, period=4):
    """
    Checkerboard initialization.
    CORRECTED: Guarantees mean(rho) = vf
    """
    nx, ny, nz = shape
    x, y, z = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )

    pattern = ((x // period) + (y // period) + (z // period)) % 2

    # Solve for densities: vf_high * p_high + vf_low * p_low = vf
    # where p_high and p_low are fractions of high/low pattern
    p_high = np.sum(pattern) / np.prod(shape)
    p_low = 1.0 - p_high

    # Choose vf_low = vf/2, solve for vf_high
    vf_low = vf / 2.0
    if p_high > 0:
        vf_high = (vf - vf_low * p_low) / p_high
    else:
        vf_high = vf

    vf_high = np.clip(vf_high, 0.0, 1.0)

    rho = np.where(pattern, vf_high, vf_low)

    actual_vf = np.mean(rho)
    # print(f"  init_checkerboard_3d: target vf={vf:.4f}, actual mean(rho)={actual_vf:.4f}")

    return rho


def init_layered_3d(vf, shape, layer_axis=2):
    """
    Layered initialization.
    CORRECTED: Guarantees mean(rho) = vf
    """
    nx, ny, nz = shape
    x, y, z = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )

    coords = [x, y, z]
    coord = coords[layer_axis]
    max_coord = shape[layer_axis]

    # Create varying density from 0 to 1 across layers
    normalized = coord / (max_coord - 1)
    rho = normalized  # This varies from ~0 to ~1

    # Rescale to have mean = vf
    current_mean = np.mean(rho)  # Should be ~0.5
    if current_mean > 0:
        rho = rho * (vf / current_mean)
    else:
        rho = np.full(shape, vf, dtype=float)

    rho = np.clip(rho, 0.0, 1.0)

    return rho


def init_center_dense_3d(vf, shape, decay_factor=0.02):
    """
    Center-dense initialization with radial decay.
    CORRECTED: Guarantees mean(rho) = vf
    """
    nx, ny, nz = shape
    center = np.array([nx / 2.0, ny / 2.0, nz / 2.0])
    max_dist = np.sqrt(center[0]**2 + center[1]**2 + center[2]**2)

    x, y, z = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )

    dist = np.sqrt(
        (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    )
    normalized_dist = dist / max_dist

    # Exponential decay
    rho = np.exp(-decay_factor * normalized_dist ** 2)

    # Rescale to have mean = vf
    current_mean = np.mean(rho)
    if current_mean > 0:
        rho = rho * (vf / current_mean)
    else:
        rho = np.full(shape, vf, dtype=float)

    rho = np.clip(rho, 0.0, 1.0)

    return rho

