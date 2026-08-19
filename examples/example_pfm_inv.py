"""
Test case for elastodamage phase-field fracture simulation.
"""

import numpy as np
from matplotlib import pyplot as plt
import jax
from jax import numpy as jnp

from diffmat.fracture.solver import elastodamage_phasefield_solve
from diffmat.fracture.rvegen import generate_particles_periodic, voxelise_particles_periodic, init_material
from diffmat.commons.io import save_arrays_to_vti
from diffmat.commons.utilities import eng2lame

from jaxmaterials.common import get_grid_spec

import time
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')


# Output directories
out_dir = f"results/fracture/inverse"
os.makedirs(out_dir, exist_ok=True)


# ============================================================================
# Setup: Grid and RVE Geometry
# ============================================================================

# Create a 3D computational grid
box_size = [0.4, 0.4, 0.4] #physical length, mm
spacing = [0.005, 0.005, 0.005]
grid = get_grid_spec(
        box_size[0], 
        box_size[1], 
        box_size[2], 
        dx=spacing[0], 
        dy=spacing[1], 
        dz=spacing[2],
        )

# Random particle distribution
np.random.seed(42)
n_particles = 20
radius_range = [0.05, 0.1]
t0 = time.time()
positions, radii = generate_particles_periodic(
    n_particles,
    box_size,
    radius_range,
    min_gap=max(spacing),
)
print(f'  gerenate_random_particles took {time.time()-t0} s')

# Material Association: Assign voxels to particle or matrix
t0 = time.time()
matID = voxelise_particles_periodic(
    grid,
    positions,
    radii,
)
print(f'  creating matID took {time.time()-t0} s')

save_arrays_to_vti(
    filename=f"{out_dir}/matID.vtk",
    arrays=[matID[None, ...]],
    names=["matID"],
    spacing=spacing,
    origin=(0, 0, 0),
    stack_components=True,
)


# Define material properties
# NMC particle + LPSC matrix [Taghikhani et al. JMPS 2025]
E_particle = 175e3 #MPa
nu_particle = 0.28 
lmbda_particle, mu_particle = eng2lame(E_particle, nu_particle)
lc_particle = 0.01 #mm
gc_particle = 2.5e-3 #N/mm

E_matrix = 22e3 #MPa
nu_matrix = 0.37
lmbda_matrix, mu_matrix = eng2lame(E_matrix, nu_matrix)
lc_matrix = 0.01 #mm
gc_matrix = 2.8e-3 #N/mm

lmbda_list = [lmbda_matrix, lmbda_particle]    # Lame parameter
mu_list = [mu_matrix, mu_particle]         # Shear modulus
gc_list = [gc_matrix, gc_particle]    # Critical energy release rate
lc_list = [lc_matrix, lc_particle]      # Characteristic length

lmbda_grid, mu_grid, gc_grid, lc_grid = init_material(
        matID,
        lmbda_list, 
        mu_list, 
        gc_list, 
        lc_list, 
        jnp.float64,
        )

# Define monotonic uniaxial loading (strain in x-direction)
Emean = 0.002
nsteps = 1000

exx0 = Emean/nsteps/10
Emean_steps = [
    jnp.array([eps_xx, 0.0, 0.0, 0.0, 0.0, 0.0])
    for eps_xx in np.linspace(exx0, Emean, nsteps)
]

# Steps at which to save output fields
save_steps = np.arange(0, nsteps-1, 100)
if nsteps-1 not in save_steps:
    save_steps = np.append(save_steps, nsteps-1)


# Solve the elastodamage phase-field problem
t_start = time.time()
epsMacro, sigMacro = elastodamage_phasefield_solve(
    grid,
    lmbda_grid,
    mu_grid,
    gc_grid,
    lc_grid,
    Emean_steps,
    save_steps,
    k_stab=1e-6,
    maxiter_PF=2000,
    maxiter_Elas=2000,
    out_dir=out_dir,
    earlystop=0.2,
)
print(f"TOTAL TIME FOR PFM SOLVE: {(time.time()-t_start):.3f} s")

# ============================================================================
# Results Visualization and Output
# ============================================================================

# Plot stress-strain curve
filename = f"{out_dir}/macro_curve.txt"
data = np.genfromtxt(filename, names=True)
data = {name: data[name] for name in data.dtype.names}

save_steps = (data["step"][data["vtk"]==1]).astype(int)

plt.figure()
plt.plot(data["e11"], data["s11"], "-*", label="Stress-strain curve")
plt.plot(
    data["e11"][save_steps],
    data["s11"][save_steps],
    "o",
    label="Saved steps",
)
plt.xlabel(r"Strain ($\varepsilon_{11}$)")
plt.ylabel(r"Stress ($\sigma_{11}$)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



exit()

# ============================================ #
# Differentiation: parameter identification    #
# ============================================ #
data = {'eps':[], 'sig':[]}

data['eps'] = [0.001, 0.00311111, 0.00522222, 0.00733333, 0.00944444, 0.01155556,
        0.01366667, 0.01577778, 0.01788889, 0.02]
data['sig'] = [0.02962656, 0.09193908, 0.151002, 0.20303075, 0.24531511, 0.27640385,
        0.29608999, 0.30516536, 0.30505681, 0.29744624]
            

'''
data['eps'] = [0.0001, 0.00030101, 0.00050202, 0.00070303, 0.00090404, 0.00110505,
 0.00130606, 0.00150707, 0.00170808, 0.00190909, 0.0021101 , 0.00231111,
 0.00251212, 0.00271313, 0.00291414, 0.00311515, 0.00331616, 0.00351717,
 0.00371818, 0.00391919, 0.00412021, 0.00432121, 0.00452222, 0.00472323,
 0.00492424, 0.00512526, 0.00532626, 0.00552728, 0.00572829, 0.00592929,
 0.00613031, 0.00633131, 0.00653233, 0.00673334, 0.00693435, 0.00713536,
 0.00733637, 0.00753738, 0.00773839, 0.0079394 , 0.0081404 , 0.00834141,
 0.00854243, 0.00874342, 0.00894444, 0.00914546, 0.00934647, 0.00954749,
 0.00974848, 0.0099495 , 0.01015051, 0.01035152, 0.01055254, 0.01075354,
 0.01095455, 0.01115555, 0.01135657, 0.01155759, 0.0117586 , 0.0119596,
 0.01216059, 0.01236163, 0.01256263, 0.01276363, 0.01296465, 0.01316565,
 0.0133667,  0.01356771, 0.01376869, 0.01396972, 0.01417072, 0.01437174,
 0.01457273, 0.01477374, 0.01497478, 0.01517571, 0.0153768 , 0.01557773,
 0.01577877, 0.01597982, 0.01618087, 0.01638188, 0.01658287, 0.01678396,
 0.01698491, 0.01718574, 0.01738705, 0.01758745, 0.01778938, 0.01798984,
 0.01819209, 0.01839347, 0.01859374, 0.01880079, 0.01902117, 0.0191396]

data['sig'] = [0.00296266, 0.00891767, 0.01486973, 0.02081512, 0.0267502, 0.03267135,
 0.038575,   0.04445751, 0.05031538, 0.05614513, 0.0619432 , 0.06770629,
 0.07343099, 0.07911389, 0.08475184, 0.09034169, 0.09588008, 0.10136429,
 0.106791,   0.11215764, 0.11746117, 0.12269878, 0.12786812, 0.13296653,
 0.13799138, 0.14294066, 0.14781156, 0.15260261, 0.15731117, 0.16193555,
 0.166474,   0.17092423, 0.17528558, 0.17955543, 0.18373296, 0.18781681,
 0.19180568, 0.19569834, 0.1994939 , 0.20319141, 0.20678997, 0.21028902,
 0.21368833, 0.21698615, 0.22018345, 0.22327954, 0.22627339, 0.22916569,
 0.23195556, 0.23464435, 0.23723066, 0.23971573, 0.2420995 , 0.24438155,
 0.24656337, 0.24864446, 0.2506262 , 0.2525085 , 0.25429192, 0.25597695,
 0.2575646 , 0.2590565 , 0.26045087, 0.2617503 , 0.2629553 , 0.2640658,
 0.26508412, 0.26600832, 0.26683986, 0.26758134, 0.26822975, 0.2687875,
 0.26925296, 0.2696264 , 0.269907  , 0.2700893 , 0.2701774 , 0.2701565,
 0.27002743, 0.26977038, 0.2693627 , 0.2687598 , 0.26787916, 0.26652026,
 0.26423544, 0.26042548, 0.25429046, 0.24468686, 0.23014405, 0.20871651,
 0.17944252, 0.14445823, 0.10876302, 0.07427165, 0.04056253, 0.01643705]
'''

data['eps'] = jnp.array(data['eps'])
data['sig'] = jnp.array(data['sig'])

Emean_steps = [
    jnp.array([eps_xx, 0.0, 0.0, 0.0, 0.0, 0.0])
    for eps_xx in data['eps']
]
icomp = 0
print(Emean_steps)

#def loss_fn(params_norm, param0):
#
#    params = {k: params_norm[k]*params0[k] for k in params0}

def loss_fn(params):

    lmbda_particle = params.get("lmbda_particle", 100.)
    mu_particle = params.get("mu_particle", 80.)
    lc_particle = params.get("lc_particle", 0.08)
    gc_particle = params.get("gc_particle", 1.e-3)
    
    lmbda_matrix = params.get("lmbda_matrix", 10.)
    mu_matrix = params.get("mu_matrix", 8.)
    lc_matrix = params.get("lc_matrix", 0.08)
    gc_matrix = params.get("gc_matrix", 1.e-3)
    
    lmbda_list = [lmbda_matrix, lmbda_particle]      # Lame parameter
    mu_list = [mu_matrix, mu_particle]          # Shear modulus
    gc_list = [gc_matrix, gc_particle]        # Critical energy release rate
    lc_list = [lc_matrix, lc_particle]        # Characteristic length
    
    lmbda_grid, mu_grid, gc_grid, lc_grid = init_material(
            matID,
            lmbda_list, 
            mu_list,
            gc_list, 
            lc_list,
            jnp.float64
            )
    
    epsMacro, sigMacro = elastodamage_phasefield_solve(
        grid,
        lmbda_grid,
        mu_grid,
        gc_grid,
        lc_grid,
        Emean_steps,
        save_steps,
        k_stab=1e-6,
        maxiter_PF=1000,
        maxiter_Elas=2000,
    )
    
#    loss = jnp.mean( (((data['eps'] - epsMacro[:,icomp])/max(data['eps']))**2 + 
#                      ((data['sig'] - sigMacro[:,icomp])/max(data['sig']))**2)**0.5 )
    loss = jnp.mean( ((data['sig'] - sigMacro[:,icomp])/max(data['sig']))**2 )
    return loss


# initialisation
params0 = {
        'lc_particle': jnp.array(0.01),
#        'gc_particle': jnp.array(0.1e-3),
#        'lc_matrix': jnp.array(0.08),
}
params = {k: params0[k] for k in params0}
params_history = {k: [] for k in params}

params_norm = {k: params[k]/params0[k] for k in params0}

value_grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=False) 

for key in params_history.keys():
    params_history[key].append(params[key])

count = 0
tol = np.inf
tol_params = np.zeros((len(params),1),np.float64)
while (tol>1e-3) & (count<10):
    
    count += 1

    #
    #l, dl_dparams = value_grad_fn(params_norm, params0)
    l, dl_dparams = value_grad_fn(params)

    print(dl_dparams)
    print(l)

#    for i, key in enumerate(params_norm.keys()):
#        p_new = params_norm[key] - l / dl_dparams[key]
#        p_new = min(max(p_new, 1e-3), 1e3)
#    
#        print(f'  {key}_init: {params_norm[key]*params0[key]}')
#        print(f'  {key}_new: {p_new*params0[key]}')
#
#        tol_params[i] = abs(p_new - params_norm[key])
#
#        params_norm[key] = p_new
#        params_history[key].append(p_new*params0[key])

    for i, key in enumerate(params.keys()):
        p_new = params[key] - l / dl_dparams[key]
        p_new = min(max(p_new, 1e-3), 1e3)
    
        print(f'  {key}_init: {params[key]}')
        print(f'  {key}_new: {p_new}')

        tol_params[i] = abs(p_new - params[key])

        params[key] = p_new
        params_history[key].append(p_new)

    tol = np.sqrt(np.mean(tol_params**2))


#
nparams = len(params0)
fig, ax = plt.subplots(nparams,1)
if nparams==1:
    key = params0.keys()
    ax.plot(np.arange(0, len(params_history[key])), params_history[key], '-*')
else:
    for i, key in enumerate(params0.keys()):
        ax[i].plot(np.arange(0, len(params_history[key])), params_history[key], '-*')
        ax[i].set_title(f'{key}')
plt.show()


