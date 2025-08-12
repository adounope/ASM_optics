config_path_list = []
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import src.ASM as ASM
import src.math_tool as mt
import src.img_tool as it
from PIL import Image
import yaml
import src.apertures as apertures
from box import Box
for config_path in config_path_list:
    time_string = datetime.now().isoformat(timespec='minutes')
    ################################################## all code in unit of μm
    config = None
    with open(config_path, 'r') as f:
        config = Box(yaml.safe_load(f))
    π, λ = np.pi, eval(config.simulation.λ)
    k = 2*π/λ
    # simulation config
    s_res = config.simulation.resolution
    s_x_start, s_x_end, s_y_start, s_y_end, s_z_start, s_z_end = tuple(config.simulation.range)
    s_Lx, s_Ly, s_Lz = s_x_end - s_x_start, s_y_end - s_y_start, s_z_end - s_z_start
    s_Nx, s_Ny, s_Nz = eval(s_res[0]), eval(s_res[1]), eval(s_res[2]) # manufactured aperture have resolution of 1000 * 1000
    s_xs, s_ys, s_zs = np.linspace(s_x_start, s_x_end, s_Nx, endpoint=False), np.linspace(s_y_start, s_y_end, s_Ny, endpoint=False), np.linspace(s_z_start, s_z_end, s_Nz, endpoint=False)
    z_batch_size = eval(config.simulation.z_batch_size)

    # plot config
    x_start, x_end, y_start, y_end= np.array(tuple(config.save.range))
    z_start, z_end = s_z_start, s_z_end
    xs, ys, zs = s_xs[(x_start <= s_xs)*(s_xs <= x_end)], s_ys[(y_start <= s_ys)*(s_ys <= y_end)], s_zs
    x_start, x_end, y_start, y_end = xs[0], xs[-1], ys[0], ys[-1]
    Δx, Δy, Δz = xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]
    Nx, Ny, Nz = len(xs), len(ys), len(zs)
    Nx_ticks, Ny_ticks, Nz_ticks = tuple(config.plot.N_ticks)
    x_ticks, y_ticks, z_ticks = np.arange(0, Nx, Nx/Nx_ticks, dtype=int), np.arange(0, Ny, Ny/Ny_ticks, dtype=int), np.arange(0, Nz, Nz/Nz_ticks, dtype=int)
    xy_range_index = (np.argwhere(s_xs==x_start).flatten()[0], np.argwhere(s_xs==x_end).flatten()[0], np.argwhere(s_ys==y_start).flatten()[0], np.argwhere(s_ys==y_end).flatten()[0])
    save_path = f'{config.path.results}'
    save_path_E2 = f'{config.path.E2}'
    print(f'memory spike: {z_batch_size * s_Nx * s_Ny * 2 * 128 / 8 / 1024**3}GB') # *2 because np fft is not in-place operation
    print(f'simulation size: s_Nx={s_Nx}, s_Ny={s_Ny}, s_Nz={s_Nz}\nfloat32 (intensity) requirement: {Nx * Ny * Nz * 32 / 8 / 1024 / 1024 / 1024} GB\nz-resolution: {(s_z_end - s_z_start)/s_Nz}μm\nconfig file: {config_path}')
    #A_xy = np.load(f'{config.path.aperture}')
    A_xy = it.img_2_array(config.path.aperture)
    if (np.array([s_Nx, s_Ny]) != np.array(A_xy.shape)).any():
        print('error: loaded aperture dimension mismatch')
        exit()
    # plt.imshow(A_xy.T, cmap='plasma')
    # plt.show()
    # compute
    time_s = datetime.now()
    ASM.ASM_3D_batch_E2_Multi_Process(A_xy, s_Lx, s_Ly, s_zs, λ, save_path_E2, z_batch_size, xy_range_index, num_process=11)
    time_e = datetime.now()
    print(f'last run: {time_e.isoformat(timespec='minutes')}')
    print(f'duration: {time_e - time_s}')