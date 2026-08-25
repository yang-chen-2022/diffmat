# diffmat

**Differentiable Materials Modelling**

`diffmat` is a research-oriented Python codebase for **differentiable materials modelling**.  
It uses an **[JAX](https://docs.jax.dev/en/latest/index.html#) based FFT numerical solver [JaxMaterials](https://github.com/eikehmueller/JaxMaterials#)**, enabling end-to-end differentiability of m[...]

This makes the framework suitable for gradient-based methods such as:
- Inverse material parameter identification
- Sensitivity analysis
- Optimisation
- Machine-learning-assisted constitutive modelling

---

## Key Ideas

- FFT-based solvers for computational materials modelling  
- JAX implementation for automatic differentiation  
- Designed for numerical simulation of material behaviour  
- Research-focused, modular, and extensible codebase  

---

## Installation

### 1. Set up a virtual environment

We recommend using conda to create an isolated environment:

```bash
conda create -n diffmat python=3.12
conda activate diffmat
```

### 2. Install JAX

Install JAX with CUDA 12 support (adjust for your setup if needed):

```bash
pip install -U "jax[cuda12]"
```

### 3. Install JaxMaterials

Clone and install JaxMaterials, which is a dependency of diffmat:

```bash
git clone git@github.com:eikehmueller/JaxMaterials.git
cd JaxMaterials/
pip install .
```

If you plan to modify JaxMaterials as part of your development, use the editable install instead:

```bash
pip install -e .
```

### 4. Install diffmat

Clone and install diffmat in editable mode for development:

```bash
git clone https://github.com/yang-chen-2022/diffmat.git
cd diffmat/
pip install -e .
```

---

## Test Cases

### Phase-Field Fracture Model (`test/test_pfm.py`)

This test demonstrates a phase-field fracture simulation applied to a composite material with particle inclusions. 
The differentiability is applied to the identification of the characteristic length of the inclusion material. In phase-field fracture model, this characteristic length cannot be measured experimentally, and often calculated based on 1D simplifications (neglecting Poisson's effect). Identification of this model parameter can be done with Newton's method, requiring sensitivity information of the model output to the parameter(s) to be identified. This sensitivity (derivatives) has been obtained in the literature using numerical perturbation method, see e.g. [Nguyen et al. 2016](https://www.sciencedirect.com/science/article/pii/S0022509616302563#s0060). At every Newton iteration, the perturbation method requires $1+p$ simulations for forward-finite-difference scheme or $1+ 2p$ solves for central-finite-difference scheme, with $p$ the number of parameters to be identified. It is clear that this method becomes impractical when $p$ is a large number. This highlights the advantage of differentiable modelling, which only requires $2$ simulations (1 forward solve $+$ 1 adjoint solve). Another advantage of differentiable modelling in this context is that it gives "exact" values of the derivatives up to the model accuracy (discretisation error and machine precision error), while the perturbation method suffers from the truncation error and round-off error of the finite difference scheme. The truncation error is $O(h)$ for forward difference and $O(h^2)$ for central difference (high if $h$ is large), whereas the round-off error becomes non-negligible when $h$ is too small.


1. **Geometry Setup**: Creates a 3D computational grid with randomly distributed spherical particles

![Initial microstructure of randomly distributed spherical inclusions in the RVE](examples/figures/pfm_initialgeom.png)

Figure: Initial geometry (input microstructure) used in the phase-field fracture test — randomly distributed spherical inclusions in the representative volume element (RVE).

RVE generation (brief): the microstructure was generated with a periodic random particle generator (see `diffmat.rvegen.generate_particles_periodic`). Typical parameters used in the example are: `box_size = [2.0, 2.0, 2.0]`, `spacing = [0.05, 0.05, 0.05]`, `n_particles = 20`, `radius_range = [0.1, 0.3]`, and `seed = 42` for reproducibility. Users can change `box_size`, `spacing`, `n_particles`, `radius_range`, and the random seed to control the particle density, size distribution, and repeatability.

2. **Material Association**: Assigns each voxel to either a particle inclusion or matrix phase based on proximity
3. **Material Properties**: Defines elastic (λ, μ) and fracture parameters (G_c, ℓ_c) for each phase
4. **Loading**: Applies monotonic uniaxial strain in the x-direction over 10 load steps
5. **Solver**: Runs the elastodamage phase-field solver with staggered iterations for elastic equilibrium and damage evolution
6. **Output**: Generates stress-strain curves and saves strain, stress, and damage fields in VTK format
7. **Differentiation**: Tests automatic differentiation for inverse parameter identification (under development)

### Topology Optimisation (`test/test_to.py`)

This test demonstrates gradient-based topology optimization using differentiable FFT solvers. It combines density-based topology optimization with automatic differentiation to design optimal mater[...]

1. **Domain Setup**: Creates a 2D rectangular domain (discretized as 99×99×1)
2. **Material Parameterization**: Uses SIMP (Solid Isotropic Material Penalization) model to interpolate material properties from density
3. **Compliance Computation**: Solves linear elasticity via Lippmann-Schwinger FFT solver with automatic differentiation
4. **Sensitivity Filtering**: Applies spatial filtering to sensitivities to prevent checkerboard patterns
5. **Optimality Criteria Update**: Iteratively updates density field using OC algorithm with volume constraint
6. **Visualization**: Plots convergence, optimized topology, and resulting strain/stress fields

This example showcases the powerful integration of:
- JAX's automatic differentiation through complex FFT-based solvers
- Gradient-based optimization for inverse design problems
- Real-time performance monitoring and visualization

**Reference**: Inspired by Mohit Pundir & David S. Kammer (2025), *Computer Methods in Applied Mechanics and Engineering*, 435, 117572.


We design periodic porous metamaterials that maximise the effective bulk modulus $K$ at prescribed solid volume fractions $\phi=0.1, 0.2, 0.3$. Each design was represented by a cubic representative volume element (RVE) of size $0.5 \times 0.5 \times 0.5 mm^3$. The solid phase had Young's modulus of $E_1=1$ GPa and a Poisson's ratio of 0.3, while the void phase was approximated bys a much softer material with Young's modulus of $E_0=10^{-6}$ GPa. 

The optimisation was performed using optimality criteria method [REF], requiring the computation of the gradient of the objective function (negative effective bulk modulus, $c=-K$) with respect to the density map $\rho$. The effective bulk modulus $K$ was computed from one homogenisation simulation subject to a macroscopic strain load $\overline{\boldsymbol{\varepsilon}}= (1,1,1,0,0,0)^T$ (energy-based method [REF]):


$$
\begin{aligned}
K &= \frac{1}{9} \sum_{i,j=1}^{3} C_{iijj}
   = \frac{1}{9} \overline{\boldsymbol{\varepsilon}}^T : \overline{\boldsymbol{\sigma}}
\qquad\text{with}\qquad
\overline{\boldsymbol{\varepsilon}} = (1,1,1,0,0,0)^T
\end{aligned}
$$


The complete objective was implemented as a differentiable JAX function. Within this function, JaxMaterials' `lippmann_schwinger()` routine solves the periodic equilibrium problem and returns the local strain and stress fields:

```Python 
def compute_c(rho, mat, grid_spec):
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
        tol=1.0e-3,
        maxits=2000,
        verbose=1,
        depth=4,
    )

    sigma_bar = jnp.mean(sigma, axis=[1, 2, 3])
    energy = jnp.sum( epsilon_bar[:3]*sigma_bar[:3] +
                      epsilon_bar[3:]*sigma_bar[3:] * 2 )

    return -energy / 9
```

The spatial density $\rho$ and material properties are passed to `lippmann_schwinger()` as constitutive parameters, while the user-defined function `compute_sigma_from_density()` maps the local strain and density fields to the local stress:


```Python
def compute_sigma_from_density(epsilon, params):
     rho, mat = params

    E = mat['E0'] + (mat['E1'] - mat['E0']) * (rho + mat['kk']) ** mat['penalty']

    lmbda = E * mat['nu'] / (1. + mat['nu']) / (1. - 2. * mat['nu'])
    mu = E / (2.0 * (1. + mat['nu']))

    tr_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
    sigma = jnp.zeros_like(epsilon)
    sigma = sigma.at[:3].set((lmbda * tr_epsilon)[None, ...] + 2.0 * mu * epsilon[:3])
    sigma = sigma.at[3:].set(2.0 * mu * epsilon[3:])
    return sigma
```

Because `lippmann_schwinger()` provides a custom reverse-mode derivative based on the adjoint-state method, the objective gradient with respect to every voxel in $\rho$ can be evaluated directly using the standard JAX interface:

```Python
value_grad_fn = jax.value_and_grad(compute_c, argnums=0, has_aux=False)
c, dc = value_grad_fn(rho, mat, grid_spec)
```

JaxMaterials therefore encapsulates both the forward equilibrium solution and its adjoint sensitivity calculation. The optimisation code does not need to differentiate explicitly through the solver iterations or implement a separate adjoint solver.

The combination of high porosity, complex pore morphology, and a stiffness contrast of $10^6$ makes these equilibrium problems numerically challenging. Some forward and adjoint solves required more than 2,000 iterations to satisfy the tolerance of $10^{-3}$, even with Anderson acceleration of depth four. We therefore limited each solve to 2,000 iterations. Despite this limit, the objective and sensitivity calculations remained sufficiently stable for the optimisation to converge.

To mitigate numerical instabilities (checker-boarding and mesh-dependency), a sensitivity filtering [REF] was applied to the sensitivity:
$$
\widetilde{\frac{\partial c}{\partial \rho}} = \frac{(\frac{\partial c}{\partial \rho}\rho) \odot \omega}{\overline{\omega} \rho}
$$
where $\odot$ denotes periodic convolution (the RVE is periodic). $\omega$ is the convolution kernel that was set to a $2\times 2\times 2$ array with all elements equal to 1, and $\overline{\omega}$ is the sum of all elements in the kernel.

For each target volume fraction, the density field was initialised using a spherical perturbation:

$$
\rho =
\begin{cases}
    \phi / 2, & x \in \mathcal{D}, \\
    \phi, & \text{otherwise}.
\end{cases}
$$
where $\mathcal{D}$ is a sphere with a diameter equal to two-thirds of the RVE size.

Figure 1 shows the evolution of the optimised topologies.

<figure style="margin: 0; text-align: center;">
  <img src="figures/to_animation_seq.gif" alt="Figure 1" width="700">
  <figcaption style="text-align: center;">
Figure 1: Evolution of the optimised porous structures at solid volume fractions of 10% (left), 20% (middle), and 30% (right).
</figcaption>
</figure>

Figure 2 presents the corresponding convergence histories. For all three volume fractions, the effective bulk modulus increases towards a stable plateau, demonstrating that the JaxMaterials forward and adjoint solutions provide sufficiently stable sensitivities for gradient-based topology optimisation.

<figure>
<div style="display: flex; justify-content: center; gap: 10px;">
<figure style="margin: 0; text-align: center;">
  <img src="figures/to_convergence.png" alt="Figure 2" width="400">
</figure>
</div>
<figcaption style="text-align: center;">
Figure 2: Evolution of the effective bulk modulus during topology optimisation at different solid volume fractions.
</figcaption>
</figure>

The computation time was not recorded, but all three cases were completed within approximately one hour using a Nvidia RTX A6000 GPU.






---

## Background and Acknowledgements

The initial development of `diffmat` builds upon and is inspired by **Eike Müller's** work:

- **JaxMaterials** by Eike Müller  
  <https://github.com/eikehmueller/JaxMaterials>

---

## Project Status

🚧 **Active research code**  
This project is under ongoing development. The API, numerical formulations, and features may change.

---

## License

