import numpy as np
import jax
from jax import numpy as jnp

from jaxmaterials.common import GridSpec
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger

from diffmat.commons.io import save_arrays_to_vti
from diffmat.perf2amitex.io import saveMesh2VTK_amitex

import pyvista as pv

import os
import time
import threading
import psutil
from pynvml import *

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

class MemoryMonitor:
    def __init__(self, gpu_index=0, interval=1.0):
        self.gpu_index = gpu_index
        self.interval = interval
        self.running = False

        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(gpu_index)

    def snapshot(self):
        # CPU
        vm = psutil.virtual_memory()

        # GPU
        info = nvmlDeviceGetMemoryInfo(self.handle)

        return {
            "cpu_used_gb": vm.used / 1024**3,
            "cpu_pct": vm.percent,
            "gpu_used_gb": info.used / 1024**3,
            "gpu_total_gb": info.total / 1024**3,
        }

    def _loop(self):
        while self.running:
            s = self.snapshot()
            print(
                f"CPU {s['cpu_used_gb']:.2f}GB ({s['cpu_pct']}%) | "
                f"GPU {s['gpu_used_gb']:.2f}/{s['gpu_total_gb']:.2f}GB"
            )
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        nvmlShutdown()


monitor = MemoryMonitor(interval=0.5)
monitor.start()


def compute_sigma_iso(epsilon, params):
    lmbda, mu = params
    tr_epsilon = epsilon[0, ...] + epsilon[1, ...] + epsilon[2, ...]
    sigma = jnp.zeros_like(epsilon)
    sigma = sigma.at[:3].set( (lmbda*tr_epsilon)[None,...] + 
                               2.*mu*epsilon[:3] )
    sigma = sigma.at[3:].set(2.*mu*epsilon[3:])
    return sigma



contrast = 1e-9
d_factor = 0.2 #diameter = Lx * d_factor

# Domain
Lx = 0.5  #mm
Ly = 0.5
Lz = 0.5

nx = 100
ny = 100
nz = 100

grid_spec = GridSpec(Lx, Ly, Lz, nx, ny, nz)
dtype = np.float64

# mesh
dx, dy, dz = grid_spec.grid_spacings

x = np.linspace(dx/2, Lx - dx/2, nx)
y = np.linspace(dy/2, Ly - dy/2, ny)
z = np.linspace(dz/2, Lz - dz/2, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# inclusion / pore
hetero_type = "sphere"
hetero_type = "crack"

r = Lx * d_factor / 2 #mm

if hetero_type=="sphere":
    mask = (X-Lx/2)**2 + (Y-Ly/2)**2 + (Z-Lz/2)**2 <= r**2
elif hetero_type=="crack":
    mask = (( np.abs(X-Lx/2)<=dx ) &
            ( (Y-Ly/2)**2 + (Z-Lz/2)**2 <= r**2 ))


# material parameters
E_matrix = 200e3 #MPa
nu_matrix = 0.3

def eng2lame(E, nu):
    lmbda = E*nu / (1.+nu) / (1. - 2.*nu)
    mu = E / 2 / (1. + nu)
    return lmbda, mu

lmbda_matrix, mu_matrix = eng2lame(E_matrix, nu_matrix)
lmbda_inclusion = lmbda_matrix * contrast
mu_inclusion = mu_matrix * contrast
print(f'lambda_inclusion={lmbda_inclusion},  mu_inclusion={mu_inclusion}')
print(f'lambda_matrix={lmbda_matrix},  mu_matrix={mu_matrix}')

lmbda = np.zeros(mask.shape) + lmbda_matrix
mu = np.zeros(mask.shape) + mu_matrix

lmbda[mask] = lmbda_inclusion
mu[mask] = mu_inclusion

lmbda = lmbda.astype(dtype)
mu = mu.astype(dtype)
print(f'mu.dtype={mu.dtype}')
print(f'lmbda.dtype={lmbda.dtype}')

lmbda0 = 0.5 * (lmbda.max() + lmbda.min())
mu0 = 0.5 * (mu.max() + mu.min())
print(f'lambda0={lmbda0},  mu0={mu0}')
# 

saveMesh2VTK_amitex('matID.vtk', mask.astype(np.uint8), 'mask', origin=[0,0,0], spacing=[dx,dy,dz])


epsilon_bar = np.array([0., 0., 0.1, 0, 0, 0], dtype=dtype)
print(f'epsilon_bar.dtype={epsilon_bar.dtype}')
for i in range(3):
    t0 = time.time()
    epsilon, sigma = lippmann_schwinger(
        compute_sigma_iso, 
        (lmbda, mu), 
        epsilon_bar, 
        ref_params = {"lambda":lmbda0,"mu":mu0},
        grid_spec=grid_spec, 
        verbose=2, 
        depth=4,
    )
    print(f'{time.time()-t0} s')

#
sigma = np.asarray(sigma.block_until_ready())
save_arrays_to_vti(
        filename=f"stress.vtk",
        arrays=[sigma],
        names=["Stress"],
        spacing = (dx, dy, dz),
        origin = (0,0,0),
        stack_components=True,
    )


monitor.stop()
