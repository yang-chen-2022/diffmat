
from jax import numpy as jnp
from diffmat.fracture.utilities import voigt_to_tensor, tensor_to_voigt

def compute_sigma_damaged(epsilon, params):
    """
    Compute stress with asymmetric degradation.

     :arg epsilon: strain in Voigt notation (6, Nx, Ny, Nz)
     :arg params:
          lmbda - spatially varying Lame parameter lambda
           mu - spatially varying Lame parameter mu
           d_phase - damage variable (Nx, Ny, Nz).
           k_stab -  Stabilisation parameter for the damage
    """
    lmbda, mu, d_phase, k_stab = params

    eps_tensor = voigt_to_tensor(epsilon)
    tr_eps = jnp.trace(eps_tensor, axis1=-2, axis2=-1)
    sigma = {}
    for sign, op in (("+", jnp.maximum), ("-", jnp.minimum)):
        tr_eps_signed = op(tr_eps, 0.0)

        eigvals, eigvecs = jnp.linalg.eigh(eps_tensor)
        eigvals_signed = op(eigvals, 0.0)

        eps_signed_tensor = jnp.einsum(
            "...ia,...a,...ja->...ij", eigvecs, eigvals_signed, eigvecs
        )

        eps_signed_v = tensor_to_voigt(eps_signed_tensor)

        sigma[sign] = 2.0 * mu * eps_signed_v
        sigma[sign] = sigma[sign].at[:3].add(lmbda * tr_eps_signed)

    return ((1.0 - d_phase[None, ...]) ** 2 + k_stab) * sigma["+"] + sigma["-"]


def compute_strain_energy(lmbda, mu, epsilon):
    """Compute the ONLY the positive/ tensile elastic strain energy to drive the fracture.

    :arg lmbda: spatially varying Lame parameter lambda
    :arg mu: spatially varying Lame parameter mu
    :arg epsilon: strain in Voigt notation [11,22,33,12,13,23], shape (6, Nx, Ny, Nz)
    """

    eps_tensor = voigt_to_tensor(epsilon)

    tr_eps = jnp.trace(eps_tensor, axis1=-2, axis2=-1)
    tr_eps_plus = jnp.maximum(tr_eps, 0.0)

    eigvals = jnp.linalg.eigvalsh(eps_tensor)

    eigvals_plus = jnp.maximum(eigvals, 0.0)
    eps_sq_plus = jnp.sum(eigvals_plus**2, axis=-1)

    psi_plus = 0.5 * lmbda * (tr_eps_plus**2) + mu * eps_sq_plus

    return psi_plus



