config_path = "results/LED_collimator1220_pinhole/config.yaml"
import numpy as np; import matplotlib.pyplot as plt # type: ignore
from datetime import datetime
import src.RS as RS
import src.math_tool as mt
import src.img_tool as it
import src.load_config as lc
import src.utils as utils
from tqdm import tqdm
import src.ASM as ASM
import gc

c, π = lc.load(config_path), np.pi
Axs2D, Ays2D = np.meshgrid(c.sim.xs, c.sim.ys, indexing='ij')
Nθs = 64
θ0s = np.linspace(0, 2*π, Nθs, endpoint=False, dtype=np.float32)
n_L = 1.5 # refractive index in LED casing (epoxy)
n_G = 1.5 # refractive index in glass
edge_pinhole = it.block_edge_mask(c.sim.Nx, c.sim.Ny, c.sim.Nx*45//300)
edge = it.block_edge_mask(c.sim.Nx, c.sim.Ny, c.sim.Nx//2) # Nx, Ny # circle mask of Nx x Ny pixel with pixel radius

# they looks the same, but this verions have less noise
lens2 = it.img_2_array('results/LED_collimator1220/LED_-1220μm_2_ooμm_λ620nm_300.bmp')

lens = np.zeros(shape=(c.sim.Nx, c.sim.Ny), dtype = lens2.dtype)
for i in range(2):
    for j in range(2):
        lens[i::2, j::2] = lens2[:, :]
it.array_2_img(lens, 'results/LED_collimator1220/LED_-1220μm_2_ooμm_λ620nm_300_zoom2.bmp')
2
for x in np.linspace(0, 50, 10, endpoint=False, dtype=int):
    x_L_point = np.array([x])
    y_L_point = np.array([0])
    z_L_point = np.array([-220]) # N_point # location of electron hole recombination
    θ_L_point = np.array([0])
    magnitude_L_point = np.array([1])
    # Rayleigh Sommerfeld summation
    λ = c.λ/n_L
    k = c.k * n_L
    r = mt.dist(Axs2D[:, :, None], Ays2D[:, :, None], z=0, x_c = x_L_point, y_c=y_L_point, z_c = z_L_point) # Nx, Ny, N_point
    E_test_0 = ((-1j/λ * np.exp(k*1j*r)*z_L_point/r**2)@(np.exp(1j*θ_L_point)*magnitude_L_point)) * c.Δx * c.Δy# Nx, Ny
    E_test_0 *= edge_pinhole # set edge to block


    aperture = E_test_0
    pad_value = 600
    mul = 2*pad_value//len(aperture)+1
    ASM.ASM_3D_batch_E_Multi_Process(A_xy=it.img_pad(aperture, (pad_value,pad_value,pad_value,pad_value), pad_val=0), Lx = c.sim.Lx*mul, Ly=c.sim.Ly*mul, zs=c.sim.zs,\
                                    λ=c.λ/n_G, path=f'{c.path.E}/x={x}',\
                                    batch_size=4, xy_range_idx=(pad_value, -pad_value, pad_value, -pad_value), num_process=1)
    time = datetime.now()
    print(f'last run: {time.isoformat(timespec="minutes")}')