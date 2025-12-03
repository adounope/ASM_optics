import numpy as np
import matplotlib.pyplot as plt
import time
import src.utils as utils
from tqdm import tqdm
import os
import gc
from multiprocessing import Process, shared_memory, Queue
import src.math_tool as mt

π = np.pi

def RS_batch(A_cfg, sim_cfg, method_cfg):
    '''
    Rayleigh Sommerfeld integral
    divide Aperture (light source) contribution into batchs, and sum,
    saving result E field to a directory path
        A_xy, Axs, Ays = A_cfg
        xs, ys, z, λ = sim_cfg
        batch_size, path = method_cfg
    '''
    A_xy, Axs, Ays = A_cfg
    xs, ys, z, λ = sim_cfg
    batch_size, path = method_cfg
    ΔAx = Axs[1] - Axs[0]
    ΔAy = Ays[1] - Ays[0]
    k = 2*π/λ

    xs2D, ys2D = np.meshgrid(xs, ys, indexing='ij')
    Axs2D, Ays2D = np.meshgrid(Axs, Ays, indexing='ij')
    A_xyF = A_xy.ravel()
    Axs2DF = Axs2D.ravel(); Ays2DF = Ays2D.ravel()

    batch_idx = utils.batch_idx(A_xy.size, batch_size)
    os.system(f'mkdir -p {path}')

    E = 0
    for idx in tqdm(batch_idx, desc='Rayleigh Sommerfeld'):
        s, e = idx
        # count = s//batch_size
        r = mt.dist(xs2D[:, :, None], ys2D[:, :, None], z, x_c=Axs2DF[s:e], y_c=Ays2DF[s:e], z_c=0) # sim.Nx, sim.Ny, A.Nx*A.Ny
        # Rayleigh Sommerfeld summation
        E += (z*np.exp(k*1j*r)/r**2)@A_xyF[s:e] # Nx, Ny
    E *= (-1j/λ) * ΔAx * ΔAy
    np.save(f"{path}/Z={z}.npy", E)

def RS_batch_Multi_Process(A_cfg, sim_cfg, method_cfg):
    
    A_xy_, Axs, Ays = A_cfg
    xs, ys, z, λ = sim_cfg
    batch_size, path, num_process = method_cfg
    ΔAx = Axs[1] - Axs[0]
    ΔAy = Ays[1] - Ays[0]

    k = 2*π/λ
    # create (parent)
    A_xy, A_xy_shm, A_xy_data = utils.create_shared_array(arr_shape=(len(Axs), len(Ays)), dtype=np.complex128)
    Axs2D, Axs2D_shm, Axs2D_data = utils.create_shared_array(arr_shape=(len(Axs), len(Ays)), dtype=float)
    Ays2D, Ays2D_shm, Ays2D_data = utils.create_shared_array(arr_shape=(len(Axs), len(Ays)), dtype=float)
    xs2D, xs2D_shm, xs2D_data = utils.create_shared_array(arr_shape=(len(xs), len(ys)), dtype=float)
    ys2D, ys2D_shm, ys2D_data = utils.create_shared_array(arr_shape=(len(xs), len(ys)), dtype=float)
    # End create (parent)

    A_xy[:, :] = A_xy_
    Axs2D[:, :], Ays2D[:, :] = np.meshgrid(Axs, Ays, indexing='ij')
    xs2D[:, :], ys2D[:, :] = np.meshgrid(xs, ys, indexing='ij')
    batch_idx = utils.batch_idx(A_xy.size, batch_size)

    os.system(f'mkdir -p {path}')

    def worker(q, p_idx):
        print(f'process {p_idx} started')
        # load (child)
        A_xy, A_xy_shm = utils.load_shared_array(*A_xy_data)
        Axs2D, Axs2D_shm = utils.load_shared_array(*Axs2D_data)
        Ays2D, Ays2D_shm = utils.load_shared_array(*Ays2D_data)
        xs2D, xs2D_shm = utils.load_shared_array(*xs2D_data)
        ys2D, ys2D_shm = utils.load_shared_array(*ys2D_data)
        # End load (child)
        Axs2DF = Axs2D.ravel()
        Ays2DF = Ays2D.ravel()
        A_xyF = A_xy.ravel()
        E_tmp = 0
        for idx in tqdm(batch_idx[p_idx::num_process], desc='Rayleigh Sommerfeld'):
            s, e = idx
            # count = s//batch_size
            r = mt.dist(xs2D[:, :, None], ys2D[:, :, None], z, x_c=Axs2DF[s:e], y_c=Ays2DF[s:e], z_c=0) # sim.Nx, sim.Ny, A.Nx*A.Ny
            # Rayleigh Sommerfeld summation
            E_tmp += (z*np.exp(k*1j*r)/r**2)@A_xyF[s:e] # Nx, Ny
            # del r
            # gc.collect()
        E_tmp *= (-1j/λ) * ΔAx * ΔAy
        print(E_tmp.shape)
        # close (child)
        A_xy_shm.close()
        Axs2D_shm.close()
        Ays2D_shm.close()
        xs2D_shm.close()
        ys2D_shm.close()
        # End close (child)
        q.put(E_tmp)
    processes = []
    qs = []
    for p_idx in range(num_process):
        qs.append(Queue())
        processes.append(Process(target=worker, args=(qs[p_idx], p_idx,) ) )
    for p_idx in range(num_process):
        processes[p_idx].start()
    E = 0
    for p_idx in range(num_process):
        E += qs[p_idx].get()
    for p_idx in range(num_process):
        processes[p_idx].join()

    # term (parent)
    A_xy_shm.close(); A_xy_shm.unlink()
    Axs2D_shm.close(); Axs2D_shm.unlink()
    Ays2D_shm.close(); Ays2D_shm.unlink()
    xs2D_shm.close(); xs2D_shm.unlink()
    ys2D_shm.close(); ys2D_shm.unlink()
    # End term (parent)

    np.save(f"{path}/Z={z}.npy", E)