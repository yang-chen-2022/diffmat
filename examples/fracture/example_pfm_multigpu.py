"""Domain-decomposed phase-field fracture run on several GPUs.

    python example_pfm_multigpu.py                 # all visible GPUs
    CUDA_VISIBLE_DEVICES=0,1 python example_pfm_multigpu.py
    # CPU emulation:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 python example_pfm_multigpu.py

Multi-node: call `jax.distributed.initialize()` first and launch one process per GPU.
"""

import time

import jax
import numpy as np
from jax import numpy as jnp
from jax.sharding import Mesh

from diffmat.fracture.rvegen import init_material
from diffmat.fracture.solve_one_step import MaterialParams, SolverConfig
from diffmat.fracture.solver_jit import solve_loading_history
from jaxmaterials.common import get_grid_spec

jax.config.update("jax_enable_x64", True)

N = 64
SPACING = 0.05

devices = jax.devices()
MESH_SHAPE = (len(devices),)  # e.g. (2, 2) for a pencil decomposition
mesh = Mesh(np.array(devices).reshape(MESH_SHAPE), ("x", "y")[: len(MESH_SHAPE)])
SHARD_AXES = ("x", None, None) if len(MESH_SHAPE) == 1 else ("x", "y", None)
print(f"{len(devices)} device(s): {devices}")

L = N * SPACING
grid = get_grid_spec(L, L, L, dx=SPACING, dy=SPACING, dz=SPACING)
idx = np.meshgrid(*[np.arange(N)] * 3, indexing="ij")
mat_id = (sum((i - N / 2) ** 2 for i in idx) < (N / 4) ** 2).astype(np.uint8)

lmbda, mu, gc, lc = init_material(
    mat_id, [60.0, 120.0], [40.0, 80.0], [2.7e-3] * 2, [0.1] * 2, jnp.float64
)
params = MaterialParams(lmbda=lmbda, mu=mu, gc=gc, lc=lc)
cfg = SolverConfig(
    grid=grid,
    lmbda0=float(0.5 * (lmbda.max() + lmbda.min())),
    mu0=float(0.5 * (mu.max() + mu.min())),
    maxiter_staggered=5,
)
E = jnp.stack([jnp.array([e, 0, 0, 0, 0, 0.0]) for e in np.arange(1, 9) * 1e-3])

for label, kwargs in (("single device", {}), ("sharded", {"mesh": mesh, "shard_axes": SHARD_AXES})):
    for run in ("compile+run", "run"):
        t0 = time.perf_counter()
        out = solve_loading_history(E, params, cfg, **kwargs)
        jax.block_until_ready(out)
        print(f"{label:14s} {run:12s} {time.perf_counter() - t0:8.2f} s")
    print("  sigma11 history:", np.asarray(out[1])[:, 0])
