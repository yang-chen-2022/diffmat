import numpy as np
import numpy as np
from diffmat.utilities import minimum_image



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

        overlap = dist2 < (rad + r)**2

        if not np.any(overlap):
            positions.append(candidate)
            radii.append(r)

    return np.asarray(positions), np.asarray(radii)




# Example
if __name__ == "__main__":
    np.random.seed(42)

    positions, radii = generate_random_particles(
        n_particles=200,
        box_size=(20, 20, 20),
        radius_range=(0.2, 1.0)
    )

    print("Number of particles:", len(radii))
    print("First particle:")
    print("Position =", positions)
    print("Radius   =", radii)

