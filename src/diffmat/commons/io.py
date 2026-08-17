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


