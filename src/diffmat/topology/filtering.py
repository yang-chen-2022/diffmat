import numpy as np
from scipy.signal import convolve

def periodic_convolve(x, kernel):
    """
    Apply periodic convolution (wrap boundary conditions).
    
    Parameters
    ----------
    x : ndarray
        Field to convolve
    kernel : ndarray
        Convolution kernel
        
    Returns
    -------
    result : ndarray
        Convolved field
    """
    pad_width = [(k // 2, k // 2) for k in kernel.shape]
    x_pad = np.pad(x, pad_width, mode="wrap")
    return convolve(x_pad, kernel, mode="valid")


def build_filter_kernel(radii):
    """
    Build a 3D filter kernel for sensitivity filtering.
    
    Parameters
    ----------
    radii : tuple
        (rx, ry, rz) - Filter radii in each direction
        
    Returns
    -------
    kernel : ndarray (2*rx+1, 2*ry+1, 2*rz+1)
        Normalized filter kernel
    """
    rx, ry, rz = radii
    x = np.arange(-rx, rx + 1)
    y = np.arange(-ry, ry + 1)
    z = np.arange(-rz, rz + 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    dist = np.sqrt(xx**2 + yy**2 + zz**2)
    maxr = float(max(radii))
    kernel = np.maximum(0.0, maxr - dist)
    kernel = kernel / np.sum(kernel)
    return kernel


def apply_sensitivity_filter(ft_type, x, dc, dv, kernel):
    """
    Apply sensitivity filter to objective and volume gradients.
    
    Parameters
    ----------
    ft_type : int
        Filter type: 1=sensitivity filter, 2=density-style averaging
    x : ndarray (nx, ny, nz)
        Density field
    dc : ndarray (nx, ny, nz)
        Objective sensitivity (gradient)
    dv : ndarray (nx, ny, nz)
        Volume sensitivity (typically all ones)
    kernel : ndarray
        Precomputed filter kernel
        
    Returns
    -------
    dc_filtered : ndarray
        Filtered objective sensitivity
    dv_filtered : ndarray (optional)
        Filtered volume sensitivity (only if ft_type==2)
    """
    Hs = kernel.sum()
    
    if ft_type == 1:
        # Sensitivity filter: weight by current density
        numerator = periodic_convolve(x * dc, kernel)
        dc_filtered = numerator / Hs / np.maximum(1e-3, x)
        return dc_filtered, dv
        
    elif ft_type == 2:
        # Density filter: standard convolution
        dc_filtered = periodic_convolve(dc, kernel) / Hs
        dv_filtered = periodic_convolve(dv, kernel) / Hs
        return dc_filtered, dv_filtered
    
    else:
        raise ValueError("ft_type must be 1 or 2")
