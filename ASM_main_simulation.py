config_paths = ["results/2fz_f30,40b1,1.5_edge_pass_1024_padx2/config.yaml"]
import numpy as np; import matplotlib.pyplot as plt
from datetime import datetime
import src.ASM as ASM
import src.math_tool as mt
import src.img_tool as it
import src.load_config as lc
for config_path in config_paths:
    c, π = lc.load(config_path), np.pi
    print(f'est memory spike: {c.sim.num_process * c.sim.z_batch_size * c.sim.Nx * c.sim.Ny * 2 * 128 / 8 / 1024**3}GB') # *2 cuz npfft is not in-place operation
    print(f'simulation size: sim.Nx={c.sim.Nx}, sim.Ny={c.sim.Ny}, sim.Nz={c.sim.Nz}\nfloat32 (intensity) requirement: {c.Nx * c.Ny * c.Nz * 32 / 8 / 1024 / 1024 / 1024} GB\nz-resolution: {(c.sim.z_end - c.sim.z_start)/c.sim.Nz}μm\nconfig file: {config_path}')
    if (np.array([c.sim.Nx, c.sim.Ny]) != np.array(c.aperture.shape)).any():
        raise Exception(f'error: loaded aperture dimension mismatch\nloaded file size: {c.aperture.shape}')
    # plt.imshow(c.aperture.T, cmap='plasma')
    # plt.show()
    # compute
    ASM.ASM_3D_batch_E2_Multi_Process(c.aperture, c.sim.Lx, c.sim.Ly, c.sim.zs, c.λ, c.path.E2, c.sim.z_batch_size, c.saved_xy_range_index, num_process=c.sim.num_process)
    time = datetime.now()
    print(f'last run: {time.isoformat(timespec="minutes")}')