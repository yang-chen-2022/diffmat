import jax
import jax.numpy as jnp
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger

# ============================================================================
# Constitutive Relations and Helper Functions
# ============================================================================

def lame_coefficients(rho, mat):
    E = mat["E0"] + (mat["E1"] - mat["E0"]) * (rho + mat["kk"]) ** mat["penalty"]
    lmbda = E * mat["nu"] / (1.0 + mat["nu"]) / (1.0 - 2.0 * mat["nu"])
    mu = E / (2.0 * (1.0 + mat["nu"]))
    return lmbda, mu


def compute_sigma_from_density(epsilon, params):
    lmbda, mu = lame_coefficients(*params)
    tr_epsilon = epsilon[0, ...] + epsilon[1, ...] + epsilon[2, ...]
    sigma = 2 * mu * epsilon + lmbda * jnp.stack(
        3 * [tr_epsilon] + 3 * [jnp.zeros(epsilon.shape[-3:], dtype=epsilon.dtype)]
    )
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

    lmbda, mu = lame_coefficients(rho, mat)
    params = {"lambda": lmbda, "mu": mu}
    ref_params = {
        key: jax.lax.stop_gradient(0.5 * (jnp.max(value) + jnp.min(value)))
        for key, value in params.items()
    }

    epsilon_bar = jnp.array([1.,1.,1.,0.,0.,0.])
    epsilon, sigma = lippmann_schwinger(
        compute_sigma_from_density,
        (rho, mat),
        epsilon_bar,
        ref_params,
        grid_spec=grid_spec,
        tol=1.0e-3,
        maxits=2000,
        verbose=1,
        depth=4,
    )
    
    sigma_bar = jnp.mean(sigma, axis=[1, 2, 3])
    energy = jnp.sum( epsilon_bar[:3]*sigma_bar[:3] + 
                      epsilon_bar[3:]*sigma_bar[3:] * 2 )

    return -energy / 9

