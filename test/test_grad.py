import numpy as np
import jax

from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxmaterials.common import GridSpec
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger

import time
from jaxmaterials.utilities import save_to_vtk

import numpy as np

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"



def compute_sigma_iso(epsilon, params):
    lmbda, mu = params
    tr_epsilon = epsilon[0, ...] + epsilon[1, ...] + epsilon[2, ...]
    sigma = jnp.zeros_like(epsilon)
    sigma = sigma.at[:3].set( (lmbda*tr_epsilon)[None,...] + 
                               2.*mu*epsilon[:3] )
    sigma = sigma.at[3:].set(2.*mu*epsilon[3:])
    return sigma


contrast = 1e2

# Domain
Lx = 0.5  #mm
Ly = 0.5
Lz = 0.5

nx = 10
ny = 10
nz = 10

grid_spec = GridSpec(nx, ny, nz, Lx, Ly, Lz)
dtype = np.float64

# mesh
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
x = np.linspace(dx/2, Lx - dx/2, nx)
y = np.linspace(dy/2, Ly - dy/2, ny)
z = np.linspace(dz/2, Lz - dz/2, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# inclusion / pore
r = 0.15 #mm
mask = (X-Lx/2)**2 + (Y-Ly/2)**2 + (Z-Lz/2)**2 <= r**2

# material parameters
E_matrix = 2.8e3 #MPa
nu_matrix = 0.4

def eng2lame(E, nu):
    lmbda = E*nu / (1.+nu) / (1. - 2.*nu)
    mu = E / 2 / (1. + nu)
    return lmbda, mu

lmbda_matrix, mu_matrix = eng2lame(E_matrix, nu_matrix)
lmbda_fibre = lmbda_matrix * contrast
mu_fibre = mu_matrix * contrast

lmbda = np.zeros(mask.shape) + lmbda_matrix
mu = np.zeros(mask.shape) + mu_matrix

lmbda[mask] = lmbda_fibre
mu[mask] = mu_fibre

lmbda = lmbda.astype(dtype)
mu = mu.astype(dtype)

lmbda0 = 0.5 * (lmbda.max() + lmbda.min())
mu0 = 0.5 * (mu.max() + mu.min())

# 



########################
## forward simulation ##
########################

#epsilon_bar = np.array([0.01, 0, 0, 0, 0, 0], dtype=dtype)

#for i in range(3):
#    t0 = time.time()
#    epsilon, sigma = lippmann_schwinger_isotropic(
#        mu, lmbda, epsilon_bar, grid_spec=grid_spec, use_cuda=False, verbose=0, depth=4,
#    )
#    print(f'{time.time()-t0} s')



#################################
## gradient w.r.t. epsilon_bar ##
#################################
def loss_fn(lmbda, mu, epsilon_bar):
    epsilon, sigma = lippmann_schwinger(
        compute_sigma_iso, 
        (lmbda, mu),
        epsilon_bar,
        ref_params={"lambda":lmbda0, "mu":mu0},
        grid_spec=grid_spec, 
        verbose=0, 
        depth=4,
    )
    return jnp.mean(sigma, axis=(1,2,3))
grad_fn = jax.jacrev(loss_fn, argnums=2)

epsilon_bar = np.array([0.01, 0, 0, 0, 0, 0], dtype=dtype)

for i in range(3):
    t0 = time.time()
    Cmacro = grad_fn(lmbda, mu, epsilon_bar)
    print(f'gradient took {time.time()-t0} s')

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


print(f'Cmacro=\n')
print_array(Cmacro)



########################
## gradient w.r.t. mu ##
########################
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


def loss_fn(lmbda, mu):
    epsilon_bars = jnp.eye(6)[:2]
    batched_fft_solve = jax.vmap(lambda eps: fft_solve(lmbda, mu, eps))
    Cmacro = batched_fft_solve(epsilon_bars)
    return -Cmacro[0,0]-Cmacro[1,1]-Cmacro[0,1]-Cmacro[1,0]  #bulk modulus
    
value_grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=False)


for i in range(3):
    t0 = time.time()
    val, grad = value_grad_fn(lmbda, mu)
    print(f'gradient took {time.time()-t0} s')


