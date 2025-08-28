import numpy as np
from box import Box
import yaml
import numbers
import src.img_tool as it

def f(xs):
    if isinstance(xs, numbers.Number):
        return xs
    elif isinstance(xs, str):
        return eval(str(xs))
    elif isinstance(xs, list):
        return [eval(str(x)) for x in xs]

def load(config_path):
    config = None
    with open(config_path, 'r') as file:
        config = Box(yaml.safe_load(file))
    class Config:
        π, λ = np.pi, f(config.simulation.λ)
        k = 2*π/λ

        # simulation config
        path = config.path
            # E2
            # results
            # aperture
        class sim:
            num_process = f(config.simulation.num_process)
            Nx, Ny, Nz = f(config.simulation.resolution)
            x_start, x_end, y_start, y_end, z_start, z_end = f(config.simulation.range)
            z_batch_size = f(config.simulation.z_batch_size)
            Lx, Ly, Lz = x_end - x_start, y_end - y_start, z_end - z_start
            xs, ys, zs = np.linspace(x_start, x_end, Nx, endpoint=False), np.linspace(y_start, y_end, Ny, endpoint=False), np.linspace(z_start, z_end, Nz, endpoint=False)
        aperture = None
        if path.aperture[-3:] == 'bmp' or path.aperture[-3:] == 'png':
            aperture = it.img_2_array(path.aperture)
        elif path.aperture[-3:] == 'npy':
            apreture = np.load(path.aperture)
        else:
            print('cannot load aperture file')
        x_start, x_end, y_start, y_end = f(config.save.range) # range
        z_start, z_end = sim.z_start, sim.z_end
        xs, ys, zs = sim.xs[(x_start <= sim.xs)*(sim.xs <= x_end)], sim.ys[(y_start <= sim.ys)*(sim.ys <= y_end)], sim.zs
        x_start, x_end, y_start, y_end = xs[0], xs[-1], ys[0], ys[-1]
        Δx, Δy, Δz = xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]
        Nx, Ny, Nz = len(xs), len(ys), len(zs)
        saved_xy_range_index = (np.argwhere(sim.xs==x_start).flatten()[0], np.argwhere(sim.xs==x_end).flatten()[0], np.argwhere(sim.ys==y_start).flatten()[0], np.argwhere(sim.ys==y_end).flatten()[0])

        # plot settings
        Nx_ticks, Ny_ticks, Nz_ticks = f(config.plot.N_ticks)
        x_tick_idxs, y_tick_idxs, z_tick_idxs = np.arange(0, Nx, Nx/Nx_ticks, dtype=int), np.arange(0, Ny, Ny/Ny_ticks, dtype=int), np.arange(0, Nz, Nz/Nz_ticks, dtype=int)
    return Config