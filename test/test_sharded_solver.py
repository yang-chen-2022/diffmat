"""Domain-decomposition test: the sharded solver must reproduce the single-device
result. Runs on virtual CPU devices, so no multi-GPU machine is required.

Run in its own process (the device count must be set before JAX is imported):
    pytest test/test_sharded_solver.py
"""

import os

os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from diffmat.fracture.rvegen import init_material
from diffmat.fracture.solve_one_step import MaterialParams, SolverConfig
from diffmat.fracture.solver_jit import solve_loading_history
from jaxmaterials.common import get_grid_spec
from jaxmaterials.solver.dfft import distributed_fft, fftn3, ifftn3

N = 16
SPACING = 0.05
EPS_STEPS = np.arange(1, 5) * 1e-3


@pytest.fixture(scope="module", autouse=True)
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def problem():
    L = N * SPACING
    grid = get_grid_spec(L, L, L, dx=SPACING, dy=SPACING, dz=SPACING)
    idx = np.meshgrid(*[np.arange(n) for n in (grid.nx, grid.ny, grid.nz)], indexing="ij")
    r2 = sum((i - n / 2) ** 2 for i, n in zip(idx, (grid.nx, grid.ny, grid.nz)))
    mat_id = (r2 < 16).astype(np.uint8)  # stiff spherical inclusion

    lmbda, mu, gc, lc = init_material(
        mat_id, [60.0, 120.0], [40.0, 80.0], [2.7e-3] * 2, [0.1] * 2, jnp.float64
    )
    params = MaterialParams(lmbda=lmbda, mu=mu, gc=gc, lc=lc)
    cfg = SolverConfig(
        grid=grid,
        lmbda0=float(0.5 * (lmbda.max() + lmbda.min())),
        mu0=float(0.5 * (mu.max() + mu.min())),
        maxiter_staggered=5,
        tol_PF=1e-8,
        tol_Elas=1e-6,
        tol_staggered=1e-6,
    )
    E = jnp.stack([jnp.array([e, 0.0, 0.0, 0.0, 0.0, 0.0]) for e in EPS_STEPS])
    return params, cfg, E


@pytest.fixture(scope="module")
def reference(problem):
    params, cfg, E = problem
    return jax.tree.map(np.asarray, solve_loading_history(E, params, cfg))


@pytest.mark.parametrize(
    "mesh_shape, axes",
    [
        ((2,), ("x", None, None)),
        ((4,), ("y", None, None)),  # mesh axis named "y" still splits grid axis 0
        ((2, 2), ("x", "y", None)),  # pencil decomposition
    ],
)
def test_sharded_matches_single_device(problem, reference, mesh_shape, axes):
    params, cfg, E = problem
    names = [a for a in axes if a is not None]
    devices = np.array(jax.devices()[: int(np.prod(mesh_shape))]).reshape(mesh_shape)
    mesh = Mesh(devices, tuple(names))

    out = solve_loading_history(E, params, cfg, mesh=mesh, shard_axes=axes)
    out = jax.tree.map(np.asarray, out)

    for got, ref in zip(out, reference):
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)


def test_indivisible_grid_is_rejected(problem):
    params, cfg, E = problem
    mesh = Mesh(np.array(jax.devices()[:3]), ("x",))  # 16 % 3 != 0
    with pytest.raises(ValueError, match="not divisible"):
        solve_loading_history(E, params, cfg, mesh=mesh)


@pytest.mark.parametrize("lead", [(), (6,)])
@pytest.mark.parametrize("mesh_shape", [(4,), (2, 2)])
def test_distributed_fft_matches_jnp_and_avoids_gather(lead, mesh_shape):
    names = ("x", "y")[: len(mesh_shape)]
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(mesh_shape), names)
    rng = np.random.default_rng(0)
    a = rng.standard_normal(lead + (16, 8, 12))
    spec = P(*([None] * len(lead)), *names, *([None] * (3 - len(names))))
    a_sh = jax.device_put(a, NamedSharding(mesh, spec))

    def roundtrip(x):
        with distributed_fft(mesh, names):
            xh = fftn3(x)
            return xh, jnp.real(ifftn3(xh))

    xh, back = jax.jit(roundtrip)(a_sh)
    np.testing.assert_allclose(xh, np.fft.fftn(a, axes=(-3, -2, -1)), atol=1e-10)
    np.testing.assert_allclose(back, a, atol=1e-12)

    hlo = jax.jit(roundtrip).lower(a_sh).compile().as_text()
    assert "all-to-all" in hlo
    assert "all-gather" not in hlo

    grad = jax.jit(jax.grad(lambda x: jnp.sum(roundtrip(x)[1] ** 2)))(a_sh)
    np.testing.assert_allclose(grad, 2 * a, atol=1e-10)
