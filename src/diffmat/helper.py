import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def _infer_step(u):
    """Infer grid spacing from sorted unique coordinates."""
    u = np.asarray(u)
    if u.size < 2:
        return 1.0
    d = np.diff(np.sort(u))
    # robust spacing estimate
    return float(np.median(d[d > 0])) if np.any(d > 0) else 1.0

def _set_axes_equal(ax):
    """Make 3D axes have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_voxel_walls_3d(
    centers,
    voxel_size=None,                 # float or (dx,dy,dz); inferred if None
    face_color=(0.2, 0.6, 0.9),       # RGB tuple
    face_alpha=0.25,                 # transparency for walls
    edge_color=(0.05, 0.1, 0.2),
    edge_lw=0.3,
    dot_color="k",
    dot_size=8,
    elev=20,
    azim=35,
    ax=None,
    title=None
):
    """
    Plot semi-transparent outer faces ("walls") of axis-aligned voxel cubes and center dots.

    centers: (N,3) array of voxel centers on a rectilinear grid.
    voxel_size: scalar or (dx,dy,dz). If None, inferred from center coordinate spacing.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (N, 3)")

    xs, ys, zs = centers[:, 0], centers[:, 1], centers[:, 2]

    # infer voxel sizes from unique coordinates if not provided
    if voxel_size is None:
        dx = _infer_step(np.unique(xs))
        dy = _infer_step(np.unique(ys))
        dz = _infer_step(np.unique(zs))
    else:
        if np.isscalar(voxel_size):
            dx = dy = dz = float(voxel_size)
        else:
            dx, dy, dz = map(float, voxel_size)

    # convert centers to integer grid indices (so neighbors can be found)
    # we map unique coords -> index
    ux = np.sort(np.unique(xs))
    uy = np.sort(np.unique(ys))
    uz = np.sort(np.unique(zs))
    x_to_i = {v: i for i, v in enumerate(ux)}
    y_to_j = {v: j for j, v in enumerate(uy)}
    z_to_k = {v: k for k, v in enumerate(uz)}

    idx = np.column_stack([np.vectorize(x_to_i.get)(xs),
                           np.vectorize(y_to_j.get)(ys),
                           np.vectorize(z_to_k.get)(zs)]).astype(int)

    occ = {tuple(p) for p in idx}

    # half-sizes for cube corners
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0

    faces = []

    # For each voxel, add only boundary faces (no neighbor in that direction)
    # Face definitions in terms of corner offsets
    # Each face is a list of 4 vertices (x,y,z) in order.
    # Directions: +/-x, +/-y, +/-z
    neigh_dirs = {
        (+1, 0, 0): lambda x, y, z: [(x+hx, y-hy, z-hz), (x+hx, y+hy, z-hz), (x+hx, y+hy, z+hz), (x+hx, y-hy, z+hz)],
        (-1, 0, 0): lambda x, y, z: [(x-hx, y-hy, z-hz), (x-hx, y-hy, z+hz), (x-hx, y+hy, z+hz), (x-hx, y+hy, z-hz)],
        (0, +1, 0): lambda x, y, z: [(x-hx, y+hy, z-hz), (x-hx, y+hy, z+hz), (x+hx, y+hy, z+hz), (x+hx, y+hy, z-hz)],
        (0, -1, 0): lambda x, y, z: [(x-hx, y-hy, z-hz), (x+hx, y-hy, z-hz), (x+hx, y-hy, z+hz), (x-hx, y-hy, z+hz)],
        (0, 0, +1): lambda x, y, z: [(x-hx, y-hy, z+hz), (x+hx, y-hy, z+hz), (x+hx, y+hy, z+hz), (x-hx, y+hy, z+hz)],
        (0, 0, -1): lambda x, y, z: [(x-hx, y-hy, z-hz), (x-hx, y+hy, z-hz), (x+hx, y+hy, z-hz), (x+hx, y-hy, z-hz)],
    }

    # convert each index back to its center coordinate
    for (i, j, k), (x, y, z) in zip(idx, centers):
        for d, face_fn in neigh_dirs.items():
            ni, nj, nk = i + d[0], j + d[1], k + d[2]
            if (ni, nj, nk) not in occ:
                faces.append(face_fn(x, y, z))

    # Prepare axes
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    # Add walls (outer faces)
    poly = Poly3DCollection(
        faces,
        facecolors=(*face_color, face_alpha) if len(face_color) == 3 else face_color,
        edgecolors=edge_color,
        linewidths=edge_lw
    )
    ax.add_collection3d(poly)

    # Plot centers as dots
    ax.scatter(xs, ys, zs, c=dot_color, s=dot_size, depthshade=True)

    # Limits with a small padding
    pad_x, pad_y, pad_z = dx, dy, dz
    ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
    ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
    ax.set_zlim(zs.min() - pad_z, zs.max() + pad_z)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    _set_axes_equal(ax)

    if title:
        ax.set_title(title)

    return ax


def plot_plane_3d(
    plane_point,
    plane_normal,
    size=1.0,
    resolution=20,
    ax=None,
    plane_color="cyan",
    plane_alpha=0.5,
    point_color="k",
    vector_color="r",
    vector_length=None
):
    """
    Plot a 3D plane defined by a point and a normal vector,
    with customizable colors.

    Parameters
    ----------
    plane_point : array-like (3,)
        A point on the plane.
    plane_normal : array-like (3,)
        Normal vector of the plane.
    size : float
        Half-size of the plane patch.
    resolution : int
        Grid resolution of the plane.
    ax : matplotlib 3D axis or None
        If None, a new figure and axis are created.
    plane_color : color
        Color of the plane surface.
    plane_alpha : float
        Transparency of the plane surface.
    point_color : color
        Color of the reference point.
    vector_color : color
        Color of the normal vector.
    vector_length : float or None
        Length of the normal vector (default = size).
    """

    P0 = np.asarray(plane_point, dtype=float)
    n = np.asarray(plane_normal, dtype=float)

    if np.linalg.norm(n) == 0:
        raise ValueError("Plane normal vector must be non-zero.")

    if vector_length is None:
        vector_length = size

    # Create axis if necessary
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    # Grid in x–y around P0
    x = np.linspace(P0[0] - size, P0[0] + size, resolution)
    y = np.linspace(P0[1] - size, P0[1] + size, resolution)
    X, Y = np.meshgrid(x, y, indexing='ij')

    a, b, c = n

    # Compute Z robustly
    if abs(c) > 1e-8:
        print('case1: in xy-plane')
        Z = P0[2] - (a * (X - P0[0]) + b * (Y - P0[1])) / c
    elif abs(b) > 1e-8:
        print('case2: in xz-plane')
        Z = Y
        Y = P0[1] - (a * (X - P0[0]) + c * (Z - P0[2])) / b
    else:
        print('other cases')
        Z = Y
        X = P0[0] - (b * (Y - P0[1]) + c * (Z - P0[2])) / a

    # Plot plane surface
    ax.plot_surface(
        X.T, Y.T, Z.T,
        color=plane_color,
        alpha=plane_alpha,
        shade=True
    )

    # Plot reference point
    ax.scatter(
        P0[0], P0[1], P0[2],
        color=point_color,
        s=50
    )

    # Plot normal vector
    ax.quiver(
        P0[0], P0[1], P0[2],
        -n[0], -n[1], -n[2],
        color=vector_color,
        length=vector_length,
        normalize=True
    )

    return ax


def plot_fibres(fibres_x0, 
                fibres_v, 
                t_range=None, 
                ax=None, 
                color='red', 
                linewidth=5, 
                point_color='black', 
                point_size=50):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    if t_range is None:
        t_range = np.array((-0.5, 2))

    for ifib in range(len(fibres_x0)):
        point0 = fibres_x0[ifib]
        direction = fibres_v[ifib]
        points = point0[None,:] + direction[None,:] * t_range[:,None]

        ax.plot(points[:,0], points[:,1], points[:,2], linestyle='-', color=color, linewidth=linewidth)
        ax.scatter(point0[0], point0[1], point0[2], color=point_color, s=point_size)

    return ax


# save into vtk
import pyvista as pv
def save_arrays_to_vti(
    filename,
    arrays,
    names=None,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    stack_components=False,
):
    """
    Save multiple (N, nx, ny, nz) arrays to a VTK ImageData (.vti) file.

    Parameters
    ----------
    filename : str
        Output filename, e.g. "data.vti"
    arrays : list of np.ndarray
        List of arrays with shape (N, nx, ny, nz)
    names : list of str, optional
        Base names for each array; defaults to array0, array1, ...
    spacing : tuple of float
        Grid spacing (dx, dy, dz)
    origin : tuple of float
        Grid origin
    stack_components : bool
        If False (default), each component is saved as a separate scalar field.
        If True, components are saved as multi-component arrays.
    """

    if names is None:
        names = [f"array{i}" for i in range(len(arrays))]

    # Sanity check for grid size consistency
    nx, ny, nz = arrays[0].shape[1:]
    for a in arrays:
        if a.shape[1:] != (nx, ny, nz):
            raise ValueError("All arrays must share the same (nx, ny, nz).")

    # Create uniform grid
    grid = pv.ImageData(dimensions=(nx, ny, nz))
    grid.spacing = spacing
    grid.origin = origin

    # Add data
    for arr, name in zip(arrays, names):
        N = arr.shape[0]

        if stack_components:
            # shape -> (nx*ny*nz, N)
            data = arr.reshape(N, -1, order="F").T
            grid.point_data[name] = data
        else:
            for i in range(N):
                comp_name = f"{name}_{i}"
                grid.point_data[comp_name] = arr[i].ravel(order="F")

    # Save file
    grid.save(filename)


