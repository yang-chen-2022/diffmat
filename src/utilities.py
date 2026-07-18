from jax import numpy as jnp


def voigt_to_tensor(v):
    """Convert 6-component Voigt notation (6, Nx, Ny, Nz) to 3x3 symmetric tensor (Nx, Ny, Nz, 3, 3)."""
    v0, v1, v2, v3, v4, v5 = v[0], v[1], v[2], v[3], v[4], v[5]
    
    row0 = jnp.stack([v0, v5, v4], axis=-1) #slots into the end
    row1 = jnp.stack([v5, v1, v3], axis=-1) 
    row2 = jnp.stack([v4, v3, v2], axis=-1)
    
    return jnp.stack([row0, row1, row2], axis=-2) #


def tensor_to_voigt(t):
    """Convert 3x3 symmetric tensor (Nx, Ny, Nz, 3, 3) back to 6-component Voigt (6, Nx, Ny, Nz)."""
    v0 = t[..., 0, 0]
    v1 = t[..., 1, 1]
    v2 = t[..., 2, 2]
    v3 = t[..., 1, 2]
    v4 = t[..., 0, 2]
    v5 = t[..., 0, 1]
    
    return jnp.stack([v0, v1, v2, v3, v4, v5], axis=0)



