# diffmat

**Differentiable Materials Modelling**

`diffmat` is a research-oriented Python codebase for **differentiable materials modelling**.  
It uses an **[JAX](https://docs.jax.dev/en/latest/index.html#) based FFT numerical solver [JaxMaterials](https://github.com/eikehmueller/JaxMaterials#)**, enabling end-to-end differentiability of material response.

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

This test demonstrates a phase-field fracture simulation applied to a composite material with reinforcing fibres. The simulation:

1. **Geometry Setup**: Creates a 3D computational grid with an embedded cylindrical fibre aligned along the z-axis
2. **Material Association**: Assigns each voxel to either the fibre or matrix phase based on proximity
3. **Material Properties**: Defines elastic (λ, μ) and fracture parameters (G_c, ℓ_c) for each phase
4. **Loading**: Applies monotonic uniaxial strain in the x-direction over 10 load steps
5. **Solver**: Runs the elastodamage phase-field solver with staggered iterations for elastic equilibrium and damage evolution
6. **Output**: Generates stress-strain curves and saves strain, stress, and damage fields in VTK format

This example illustrates the framework's capability for simulating complex damage and fracture phenomena in heterogeneous materials.

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

