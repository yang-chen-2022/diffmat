import jax
import jax.numpy as jnp
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger

# ============================================================================
# Constitutive Relations and Helper Functions
# ============================================================================

def compute_sigma_from_density(epsilon, params):
    """
    Compute isotropic stress from strain using density field.
    
    Parameters
    ----------
    epsilon : ndarray (6, nx, ny, nz)
        Strain field in Voigt notation
    params : tuple
        (rho, mat: MaterialParams)
            rho: (6, nx, ny, nz) density field
            mat: MaterialParams 
              E0, E1: Young's modulis of phase 1 and 2, respectively
              nu: scalar or ndarray (nx, ny, nz), Poisson ratio
              kk: scalar, numerical stablisation parameter
              penalty: scalar, penalty parameter for SIMP model

    Returns
    -------
    sigma : ndarray (6, nx, ny, nz)
        Stress field in Voigt notation
    """
    
    rho, mat = params

    E = mat['E0'] + (mat['E1'] - mat['E0']) * (rho + mat['kk']) ** mat['penalty']

    lmbda = E * mat['nu'] / (1. + mat['nu']) / (1. - 2. * mat['nu'])
    mu = E / (2.0 * (1. + mat['nu']))

    tr_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
    sigma = jnp.zeros_like(epsilon)
    sigma = sigma.at[:3].set((lmbda * tr_epsilon)[None, ...] + 2.0 * mu * epsilon[:3])
    sigma = sigma.at[3:].set(2.0 * mu * epsilon[3:])
    return sigma

# ============================================================================
# Objective Function and Sensitivity
# ============================================================================

def compute_c(rho, mat, grid_spec):
    """
    Compute macroscopic compliance (inverse stiffness).
    Using the energy based method, see Chen, et al. "Fft-based inverse homogenization for cellular material design." 
           International Journal of Mechanical Sciences 231 (2022): 107572.
    
    Parameters
    ----------
    epsilon_bar : ndarray (6,)
        Macroscopic strain (Voigt notation)
    rho : ndarray (nx, ny, nz)
        density field
    mat : Material parameters (E0, E1, nu, kk, penalty)
    grid_spce : grid specification
        
    Returns
    -------
    sigma_macro : ndarray (6,)
        Macroscopic stress (Voigt notation)
    """

    # Compute reference material parameters Lambda0, Mu0
    E = mat['E0'] + (mat['E1'] - mat['E0']) * (rho + mat['kk']) ** mat['penalty']

    lmbda = E * mat['nu'] / (1. + mat['nu']) / (1. - 2. * mat['nu'])
    mu = E / (2.0 * (1. + mat['nu']))

    lmbda0 = jax.lax.stop_gradient(0.5 * (jnp.max(lmbda) + jnp.min(lmbda)))
    mu0 = jax.lax.stop_gradient(0.5 * (jnp.max(mu) + jnp.min(mu)))
    
    # Solve linear elastic problem via Lippmann-Schwinger FFT solver.
    epsilon_bar = jnp.array([1.,1.,1.,0.,0.,0.])

    epsilon, sigma = lippmann_schwinger(
        compute_sigma_from_density,
        (rho, mat),
        epsilon_bar,
        ref_params={"lambda": lmbda0, "mu": mu0},
        grid_spec=grid_spec,
        tol=1.0e-4,
        maxits=2000,
        verbose=0,
        depth=8,
    )
    
    sigma_bar = jnp.mean(sigma, axis=[1, 2, 3])
    energy = jnp.sum( epsilon_bar[:3]*sigma_bar[:3] + 
                       epsilon_bar[3:]*sigma_bar[3:] * 2 )

    return -energy / 9

