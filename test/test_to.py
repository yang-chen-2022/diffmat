## test case on topology optimisation
## inspired by 
##  Morhit Pundir & David S. Kammer, 2025, Computer Methods in Applied Mechanics and Enigneering, 435, 117572




import jax
import jax.numpy as jnp

import numpy as np
#from scipy.ndimage import convolve
from jax.scipy.signal import convolve

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from jaxmaterials.common import GridSpec
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time

from functools import partial




def compute_sigma_iso(epsilon, params):
    lmbda, mu = params
    tr_epsilon = epsilon[0, ...] + epsilon[1, ...] + epsilon[2, ...]
    sigma = jnp.zeros_like(epsilon)
    sigma = sigma.at[:3].set( (lmbda*tr_epsilon)[None,...] + 
                               2.*mu*epsilon[:3] )
    sigma = sigma.at[3:].set(2.*mu*epsilon[3:])
    return sigma



def print_array(arr, precision=3, col_width=10):
    """
    Prints a 2D array in a clean, aligned format.
    
    :param arr: list or numpy array
    :param precision: number of decimal places for floats
    :param col_width: width of each column for alignment
    """
    # Convert to NumPy array for consistent handling
    arr = np.array(arr)

    # Format string for each number
    fmt = f"{{:>{col_width}.{precision}f}}" if np.issubdtype(arr.dtype, np.floating) else f"{{:>{col_width}}}"

    for row in arr:
        print("[" + " ".join(fmt.format(x) for x in row) + "]")



# Filtering
def periodic_convolve(x, kernel):
    pad_width = [(k // 2, k // 2) for k in kernel.shape]
    x_pad = jnp.pad(x, pad_width, mode="wrap")
    return convolve(x_pad, kernel, mode="valid")


def build_filter_kernel(radii):
    rx, ry, rz = radii
    x = np.arange(-rx, rx + 1)
    y = np.arange(-ry, ry + 1)
    z = np.arange(-rz, rz + 1)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    
    dist = np.sqrt(xx**2 + yy**2 + zz**2)
    
    kernel = jnp.maximum(0.0, max(radii) - dist)
    
    return kernel / kernel.sum()

'''
def build_filter_kernel(rmin, spacing=(1., 1., 1.,)):
    
    dx, dy, dz = spacing
    
    rx = int(np.ceil(rmin[0] / dx))
    ry = int(np.ceil(rmin[1] / dy))
    rz = int(np.ceil(rmin[2] / dz))

    x = np.arange(-rx, rx + 1) * dx
    y = np.arange(-ry, ry + 1) * dy
    z = np.arange(-rz, rz + 1) * dz
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    
    dist = np.sqrt(xx**2 + yy**2 + zz**2)
    
    kernel = jnp.maximum(0.0, max(rmin) - dist)
    
    return kernel
'''

@jax.jit(static_argnames=["ft_type"])
def apply_sensitivity_filter(ft_type, x, dc, dv, kernel):
    """
    3D structured-grid sensitivity filter.

    Parameters
    ----------
    ft_type : int
        1 = sensitivity filter
        2 = density-style averaging filter
    x : ndarray (nx, ny, nz)
        density field
    dc : ndarray
        objective sensitivity
    dv : ndarray
        volume sensitivity
    kernel : ndarray
        precomputed filter kernel

    Returns
    -------
    dc_filtered, dv_filtered
    """

    ones = jnp.ones_like(x)
    Hs = kernel.sum()

    if ft_type == 1:
        #numerator = convolve(x * dc, kernel, mode="same")
        numerator = periodic_convolve(x * dc, kernel)
        dc_filtered = numerator / Hs / jnp.maximum(1e-3, x) #TODO: 1e-3 --> user defined?

    elif ft_type == 2:
        #dc_filtered = convolve(dc, kernel, mode="same") / Hs
        #dv_filtered = convolve(dv, kernel, mode="same") / Hs
        dc_filtered = periodic_convolve(dc, kernel) / Hs
        dv_filtered = periodic_convolve(dv, kernel) / Hs
        return dc_filtered, dv_filtered

    return dc_filtered, dv


# initial density field
def get_initial_density(vf, shape):
    nx, ny, nz = shape

    rho = np.full(shape, vf, dtype=float)

    d = nx / 3.0

    x, y, z = np.meshgrid(
        np.arange(nx),
        np.arange(ny),
        np.arange(nz),
        indexing="ij"
    )

    r = np.sqrt((x - nx / 2.0 - 0.5) ** 2 +
                (y - ny / 2.0 - 0.5) ** 2  
               )

    rho[r < d] = vf / 2.0

    return jnp.array(rho)


# optimality criteria update
def oc(rho, dc, dv, ft_type, vf, kernel=None, move=0.2, tol=1e-6):
    """
    Optimality Criteria (OC) update for structured-grid topology optimization.

    Parameters
    ----------
    rho : ndarray
        Current density field.
    dc : ndarray
        Objective sensitivity field.
    dv : ndarray
        Volume sensitivity field.
    ft_type : int
        Filter type:
            1 = sensitivity-filtered formulation
            2 = density-filtered formulation
    vf : float
        Target volume fraction.
    kernel : ndarray, optional
        Convolution kernel for density filtering (required if ft_type == 2).
    move : float, optional
        Maximum per-iteration density change.
    tol : float, optional
        Binary-search tolerance for the Lagrange multiplier.
        
    Returns
    -------
    rho_new : ndarray
        Updated density field.
    change : float
        Maximum absolute design-variable change.
        
    Notes
    -----
    The OC update rule is:
        rho_trial = rho * sqrt( -dc / (dv * lambda) )
    with move limits and box constraints:
        rho_new = clip(rho_trial, rho - move, rho + move)
        rho_new = clip(rho_new, 0, 1)
    If ft_type == 2, density filtering is applied as convolution:
        rho_filtered = (w * rho_new) / sum(w)
    where * denotes convolution.
    """

    x = rho.copy()
    l1, l2 = 0.0, 1e9

    if ft_type == 2:
        if kernel is None:
            raise ValueError("kernel must be provided when ft_type == 2")
        Hs = kernel.sum()

    while (l2 - l1) > tol:
        lmid = 0.5 * (l1 + l2)

        # Safe OC ratio
        dr = np.abs(-dc / (dv * lmid))

        # Trial update
        x_trial = x * np.sqrt(dr)

        # Apply move limits first
        lower = x - move
        upper = x + move
        xnew = np.clip(x_trial, lower, upper)

        # Apply physical bounds
        xnew = np.clip(xnew, 0.0, 1.0)

        # Filtering step
        if ft_type == 1:
            rho_candidate = xnew
        elif ft_type == 2:
            rho_candidate = periodic_convolve(xnew, kernel) / Hs
        else:
            raise ValueError("ft_type must be 1 or 2")

        # Volume constraint check
        if rho_candidate.mean() > vf:
            l1 = lmid
        else:
            l2 = lmid

    change = np.max(np.abs(xnew - x))

    return jnp.array(rho_candidate), change



# Domain
Lx = 0.5  #mm
Ly = 0.5
Lz = 0.5

nx = 99
ny = 99
nz = 1

grid_spec = GridSpec(nx, ny, nz, Lx, Ly, Lz)
dtype = np.float64
shape = (nx, ny, nz)

dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
    
# material parameters
E0 = 1e-1 #MPa, N/mm^2
E1 = 1 #MPa
nu = 0.3
kk = 0.  #1.e-2

def eng2lame(E, nu):
    lmbda = E*nu / (1.+nu) / (1. - 2.*nu)
    mu = E / 2 / (1. + nu)
    return lmbda, mu

#
def param(rho):
    return E0 + (E1 - E0) * (rho + kk) ** penalty


# optimisation setting
vf = 0.3
penalty = 5.

ft_type = 1
#kernel = build_filter_kernel(rmin=(dx*1.3, dy*1.3, 0), spacing=(dx, dy, dz))
kernel = build_filter_kernel((2,2,0))

# FFT solve
def fft_solve(lmbda, mu, epsilon_bar):
    lmbda0 = jax.lax.stop_gradient(0.5 * (jnp.max(lmbda) + jnp.min(lmbda)))
    mu0 = jax.lax.stop_gradient(0.5 * (jnp.max(mu) + jnp.min(mu)))
    epsilon, sigma = lippmann_schwinger(
        compute_sigma_iso, 
        (lmbda, mu),
        epsilon_bar,
        ref_params = {"lambda":lmbda0, "mu":mu0},
        grid_spec=grid_spec,
        verbose=0, 
        depth=4,
     )

    return jnp.mean(sigma, axis=[1,2,3])


##
#def compute_c(mu, lmbda):
#    epsilon_bar = jnp.array([0.,0.,0.,1.,0.,0.])
#    sigma_bar = fft_solve(mu, lmbda, epsilon_bar)
#    return -sigma_bar[3]  #shear modulus / compliance

def compute_c(lmbda, mu):
    epsilon_bars = jnp.eye(6)[:2]
    batched_fft_solve = jax.vmap(lambda eps: fft_solve(lmbda, mu, eps))
    Cmacro = batched_fft_solve(epsilon_bars)
    return -Cmacro[0,0]-Cmacro[1,1]-Cmacro[0,1]-Cmacro[1,0]  #bulk modulus / compliance

def compute_c_and_dc(rho):
    lmbda, mu = eng2lame(param(rho), nu)
    value_grad_fn = jax.value_and_grad(compute_c, argnums=1, has_aux=False)
    c, dc_dmu = value_grad_fn(lmbda, mu) 
    dc = dc_dmu * (E1-E0) / 2. / nu * penalty * (rho+kk)**(penalty-1) * (nx*ny*nz)
    return c, dc




'''
## use strain energy method for computing compliance and its sensitivity
mu0, lmbda0 = eng2lame(E0, nu)
mu1, lmbda1 = eng2lame(E1, nu)
lmbda10 = lmbda1 - lmbda0
mu10 = mu1 - mu0
@jax.jit
def compute_c_and_dc(rho):
    mu, lmbda = eng2lame(param(rho), nu)
    epsilon_bar = jnp.array([1., 1., 0., 0., 0., 0.])
    epsilon, sigma = lippmann_schwinger_isotropic(
            {"lambda":lmbda, "mu":mu},
            epsilon_bar, 
            grid_spec=grid_spec, 
            use_cuda=False, 
            verbose=0, 
            depth=4,
           )
    c = - (epsilon_bar*np.mean(sigma, axis=(1,2,3))).sum()

    sigma10 = jnp.zeros_like(sigma)
    sigma10 = sigma10.at[:3].set( lmbda10 * jnp.sum(epsilon[:3], axis=0)[None,...] + 
                                   2. * mu10 * epsilon[:3] )
    sigma10 = sigma10.at[3:].set(2. * mu10 * epsilon[3:])
    dc = -jnp.sum( epsilon*((rho[None,...]+kk)**(penalty-1))*sigma10, axis=0)  * penalty
    #dc = -np.sum(epsilon*sigma, axis=0) * penalty / (rho+kk) / (nx*ny*nz)
    return c, dc
'''

##
def optimize(maxIter=100):
    rho = jnp.array(get_initial_density(vf, shape))
    change, loop = 10.0, 0

    objective_values = []
    
    while change > 0.01 and loop < maxIter:
        loop += 1
        t0 = time.time()
        c, dc = compute_c_and_dc(rho)
        print(f'  ----> max(|dc|) = {jnp.max(jnp.abs(dc))}')
        c.block_until_ready()
        print(f'    compute_c_and_dc took: {time.time()-t0} s') 

        t0 = time.time()
        dv = jnp.ones(shape)
        dc, dv = apply_sensitivity_filter(ft_type, rho, dc, dv, kernel)
        dc.block_until_ready()
        print(f'    apply_sensitivity_filter took: {time.time()-t0} s') 

        t0 = time.time()
        rho, change = oc(rho, dc, dv, ft_type, vf, kernel=kernel, move=0.2, tol=1e-6)
        rho.block_until_ready()
        print(f'    oc_update took: {time.time()-t0} s') 
        
        status = "iter {:d} ;  obj {:.2F} ; vol {:.2F}".format(loop, c, jnp.mean(rho))
        objective_values.append(c)
        if loop % 20 == 0:
            plt.figure(figsize=(2, 2))
            plt.imshow(-np.array(rho[...,nz//2]), cmap="gray")
            plt.title(status)
            plt.show()

        print(status, "change {:.2F}".format(change))

    plt.figure(figsize=(2, 2))
    plt.imshow(-np.array(rho[...,nz//2]), cmap="gray")
    plt.title(status)
    plt.show()

    return rho, objective_values

##
rho, objective_values = optimize(30)

plt.figure()
plt.plot(-np.array(objective_values), '*-')
plt.show()


## plot stress/strain field
epsilon_bar = epsilon_bars = jnp.array([1.,0.,0.,0.,0.,0.])
lmbda, mu = eng2lame(param(rho), nu)
lmbda0 = 0.5 * (lmbda.max() + lmbda.min())
mu0 = 0.5 * (mu.max() + mu.min())
epsilon, sigma = lippmann_schwinger(
	compute_sigma_iso,
	(lmbda, mu),
	epsilon_bar, 
	ref_params = {"lambda":lmbda0, "mu":mu0},
	grid_spec=grid_spec, 
	verbose=0, 
	depth=4,
        )

fig, ax = plt.subplots(1,3)
ax[0].imshow(-np.array(rho[...,nz//2]), cmap="gray")
ax[1].imshow(epsilon[0][...,nz//2])
ax[2].imshow(sigma[0][...,nz//2])
plt.show()


