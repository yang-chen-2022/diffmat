import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxmaterials.solver.dfft import distributed_fft

from diffmat.fracture.solve_one_step import (
    LoadConditions,
    MaterialParams,
    SolverConfig,
    StateVariables,
    split_solver_cfg,
    solve_one_load_step_with_info,
)
from diffmat.fracture.constitutive import compute_sigma_damaged, compute_strain_energy

from functools import partial
import os

import numpy as np

from diffmat.commons.io import save_arrays_to_vti


def _write_macro_line(out_dir, step, eps_macro, sig_macro, vtk_saved):
    path = os.path.join(out_dir, "macro_curve.txt")
    step = int(step)
    if step == 0:
        header = (
            f"{'step':>8}"
            f"{'e11':>15}{'e22':>15}{'e33':>15}"
            f"{'e12':>15}{'e13':>15}{'e23':>15}"
            f"{'s11':>15}{'s22':>15}{'s33':>15}"
            f"{'s12':>15}{'s13':>15}{'s23':>15}"
            f"{'vtk':>15}\n"
        )
        with open(path, "w") as f:
            f.write(header)
    line = (
        f"{step:8d}"
        + "".join(f"{x:15.6e}" for x in np.asarray(eps_macro))
        + "".join(f"{x:15.6e}" for x in np.asarray(sig_macro))
        + f"{int(vtk_saved):15d}"
        + "\n"
    )
    with open(path, "a") as f:
        f.write(line)


def _write_vtk(out_dir, spacing, step, epsilon, sigma, d):
    save_arrays_to_vti(
        filename=os.path.join(out_dir, f"fields_{int(step):04d}.vtk"),
        arrays=[np.asarray(epsilon), np.asarray(sigma), np.asarray(d)[None, ...]],
        names=["Strain", "Stress", "Damage"],
        spacing=spacing,
        origin=(0, 0, 0),
        stack_components=True,
    )


def _fft_mesh_axes(spec):
    """Mesh axes for the distributed FFT: ("a",) slab, ("a", "b") pencil, else ()."""
    spec = tuple(spec) + (None,) * (3 - len(tuple(spec)))
    if spec[0] is not None and spec[1:] == (None, None):
        return (spec[0],)
    if spec[0] is not None and spec[1] is not None and spec[2] is None:
        return (spec[0], spec[1])
    return ()


def solve_loading_history(
    Emean_steps,
    material_params: MaterialParams,
    solver_cfg: SolverConfig,
    out_dir=None,
    save_steps=(),
    save_unconverged=False,
    mesh=None,
    shard_axes=("x", None, None),
):
    """Incremental staggered solution over a sequence of macroscopic strains.

    If `out_dir` is given, the macroscopic strain/stress of every step is
    written to `out_dir/macro_curve.txt` and the strain, stress and damage fields
    of the load steps listed in `save_steps` to `out_dir/fields_XXXX.vtk`,
    from within the compiled loop. With `save_unconverged=True`, the fields are
    also saved at load steps where the staggered iterations hit
    `maxiter_staggered` without converging. Saving is a side effect only and does
    not enter the differentiated computation.

    If `mesh` (a `jax.sharding.Mesh`) is given, the spatial grid is decomposed over
    the devices: `shard_axes` names, for each of the three grid axes, the mesh axis
    it is split along (None = not split). The material fields are placed on the
    mesh. For a slab (`("a", None, None)`) or pencil (`("a", "b", None)`)
    decomposition the 3D FFTs use explicit transposes (`all_to_all`, see
    `jaxmaterials.solver.dfft`); XLA's own partitioner would all-gather the full
    field for every FFT. Finite differences and reductions are partitioned by XLA.
    Each split grid dimension must be divisible by its mesh axis size, and for
    the transposed FFT layout also grid axis 1 by the first mesh axis (and grid
    axis 2 by the second mesh axis for pencils).
    Field output (`save_steps`, `save_unconverged`) gathers data to the host, so
    prefer `out_dir=None` for large sharded runs.
    """
    shardings = None
    if mesh is not None:
        shardings = (
            NamedSharding(mesh, P(*shard_axes)),
            NamedSharding(mesh, P(None, *shard_axes)),
        )
        grid = solver_cfg.grid
        sizes = (grid.nx, grid.ny, grid.nz)
        for n, ax in zip(sizes, shard_axes):
            if ax is not None and n % mesh.shape[ax]:
                raise ValueError(
                    f"grid size {n} is not divisible by mesh axis '{ax}' "
                    f"of size {mesh.shape[ax]}"
                )
        fft_axes = _fft_mesh_axes(shard_axes)
        # Transposed (spectral) layout: grid axis 1 over a, grid axis 2 over b
        for n, ax in zip(sizes[1:], fft_axes):
            if n % mesh.shape[ax]:
                raise ValueError(
                    f"grid size {n} is not divisible by mesh axis '{ax}' "
                    f"of size {mesh.shape[ax]} (needed for the transposed FFT layout)"
                )
        material_params = jax.tree.map(
            lambda a: jax.device_put(a, shardings[0]), material_params
        )
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    save_steps = tuple(int(s) for s in np.asarray(save_steps).ravel())
    return _solve_loading_history(
        Emean_steps,
        material_params,
        solver_cfg,
        None if out_dir is None else str(out_dir),
        save_steps,
        bool(save_unconverged),
        shardings,
    )


# Incremental solution with staggered scheme
@partial(
    jax.jit,
    static_argnames=(
        "solver_cfg",
        "out_dir",
        "save_steps",
        "save_unconverged",
        "shardings",
    ),
)
def _solve_loading_history(
    Emean_steps,
    material_params: MaterialParams,
    solver_cfg: SolverConfig,
    out_dir,
    save_steps,
    save_unconverged,
    shardings=None,
):
    # Slab/pencil layouts use the transpose-based FFT; other layouts fall back to
    # the XLA-partitioned jnp.fft.
    mesh = shardings[0].mesh if shardings is not None else None
    fft_axes = _fft_mesh_axes(shardings[0].spec) if shardings is not None else ()
    with distributed_fft(mesh, fft_axes):
        return _solve_loading_history_impl(
            Emean_steps,
            material_params,
            solver_cfg,
            out_dir,
            save_steps,
            save_unconverged,
            shardings,
        )


def _solve_loading_history_impl(
    Emean_steps,
    material_params: MaterialParams,
    solver_cfg: SolverConfig,
    out_dir,
    save_steps,
    save_unconverged,
    shardings=None,
):
    def constrain(x):
        if shardings is None:
            return x
        return jax.lax.with_sharding_constraint(x, shardings[x.ndim == 4])

    staggered_loop_cfg, staggered_cfg = split_solver_cfg(solver_cfg)

    dtype = material_params.lmbda.dtype

    d0 = jnp.zeros(
        (solver_cfg.grid.nx, solver_cfg.grid.ny, solver_cfg.grid.nz),
        dtype=dtype,
    )

    HH0 = jnp.zeros_like(d0)

    epsilon0 = jnp.zeros(
        (6, solver_cfg.grid.nx, solver_cfg.grid.ny, solver_cfg.grid.nz),
        dtype=dtype,
    ) + Emean_steps[0][:, None, None, None]

    d0, HH0, epsilon0 = constrain(d0), constrain(HH0), constrain(epsilon0)
    depsilon0 = jnp.zeros_like(epsilon0)

    def step_fn(carry, xs):

        step, Emean = xs
        d, epsilon, HH, depsilon = carry

        if solver_cfg.verbose > 0:
            jax.debug.print("\n=== load step {i} ===\nEmean={E}", i=step, E=Emean, ordered=True)

        load_conditions = LoadConditions(Emean=Emean)
        state_variables = StateVariables(
                HH=HH,
                depsilon=depsilon,
            )

        (
            (
                d,
                epsilon,
            ),
            (
                _,
                converged,
            ),
        ) = solve_one_load_step_with_info(
            (
                d,
                epsilon,
            ),
            material_params,
            load_conditions,
            state_variables,
            staggered_cfg,
            staggered_loop_cfg,
        )

        sigma = compute_sigma_damaged(
                epsilon,
                (material_params.lmbda, material_params.mu, d, solver_cfg.k_stab),
            )

        psi = compute_strain_energy(
                material_params.lmbda,
                material_params.mu,
                epsilon,
            )
        HH = jax.lax.stop_gradient(jnp.maximum(HH, psi))

        depsilon = epsilon - Emean[:, None, None, None]
        depsilon = jax.lax.stop_gradient(depsilon)

        eps_macro = jnp.mean(epsilon, axis=(1, 2, 3))
        sig_macro = jnp.mean(sigma, axis=(1, 2, 3))

        if out_dir is not None:
            is_saved = (
                jnp.isin(step, jnp.asarray(save_steps, dtype=step.dtype))
                if save_steps
                else jnp.asarray(False)
            )
            if save_unconverged:
                is_saved = is_saved | ~converged
            eps_macro_, sig_macro_ = jax.lax.stop_gradient((eps_macro, sig_macro))
            jax.debug.callback(
                partial(_write_macro_line, out_dir),
                step,
                eps_macro_,
                sig_macro_,
                is_saved,
                ordered=True,
            )
            if save_steps or save_unconverged:
                fields = jax.lax.stop_gradient((epsilon, sigma, d))
                jax.lax.cond(
                    is_saved,
                    lambda: jax.debug.callback(
                        partial(_write_vtk, out_dir, solver_cfg.grid.grid_spacings),
                        step,
                        *fields,
                    ),
                    lambda: None,
                )

        d, epsilon, HH, depsilon = map(constrain, (d, epsilon, HH, depsilon))

        output = (
            eps_macro,
            sig_macro,
            epsilon,
            sigma,
            d,
        )

        return (d, epsilon, HH, depsilon), output

    _, outputs = jax.lax.scan(
        step_fn,
        (d0, epsilon0, HH0, depsilon0),
        (jnp.arange(Emean_steps.shape[0]), Emean_steps),
    )

    return outputs
