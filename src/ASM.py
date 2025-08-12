#angular spectrum method
import numpy as np
import matplotlib.pyplot as plt
import time
import src.utils as utils
from tqdm import tqdm
import os
import gc
from multiprocessing import Process, shared_memory
# %load_ext autoreload
# %autoreload 2
# %matplotlib ipympl
π=np.pi

def ASM_3D(A_xy, Lx, Ly, zs, λ):
    '''
    if not enough RAM, use ASM_3D_batch instead
    Angular Spectrum Method
        A_xy: <2darray> aperture tensor
        Lx: <float>
        Ly: <float>
        zs: <1darray<float>> distance to screen
        λ: <float>
    '''
    Nx, Ny = A_xy.shape
    k=π*2/λ
    # x, y domain unit:
        # n * Lx / N = distance (m)
        # n = N*distance/Lx
    FFT_A = np.fft.fft2(A_xy)[:, :, None] # Nx, Ny # domain: 0 to N-1
    # FFT_A domain unit:
        # n (wavelength / array) = Lx (meter / array) / λ (meter / wavelength)
        # n = Lx / λ (wavelength / array)
    tmp_x = np.fft.fftfreq(Nx, 1/Nx).astype(complex) # domain of FFT # value [0,1,... N/2-1, -N/2, -N/2+1, ..., -1]
    tmp_y = np.fft.fftfreq(Ny, 1/Ny).astype(complex) # complex to avoid sqrt(-1) being nan
    # kx domain unit:
        # n = 2π/λ = FFT_A_n * 2π/Lx
    kx = tmp_x * 2*π/Lx
    ky = tmp_y * 2*π/Ly
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    # H = e^(i*sqrt(k**2 - kx**2 - ky**2 ) * z) # transfer function # domain: -(2π/L * N/2) to (2π/L * N/2)
    out = FFT_A * np.exp(1.0j*(k**2-kx[:, :, None]**2-ky[:, :, None]**2)**0.5 * zs) # memory spike
    return np.fft.ifft2(out, axes=(0, 1))

def ASM_3D_batch_E2(A_xy, Lx, Ly, zs, λ, path, batch_size, xy_range_idx=None):
    # only store intensity as float32, reduce storage
    '''
    xy_range_index <tuple<int, int, int, int>> x_start, x_end, y_start, y_end (ends are inclusive)
    '''
    Nx, Ny = A_xy.shape
    x1, x2, y1, y2 = 0, Nx-1, 0, Ny-1
    if xy_range_idx is not None:
        x1, x2, y1, y2 = xy_range_idx
    Nz = len(zs)
    os.system(f'mkdir -p {path}')
    batch_idx = utils.batch_idx(Nz, batch_size)

    k=π*2/λ
    FFT_A = np.fft.fft2(A_xy) # Nx, Ny # domain: 0 to N-1
    kx = np.fft.fftfreq(Nx, 1/Nx).astype(complex)* 2*π/Lx
    ky = np.fft.fftfreq(Ny, 1/Ny).astype(complex)* 2*π/Ly
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    kz = (k**2-kx**2-ky**2)**0.5
    del kx, ky
    gc.collect()

    for count, idx in enumerate(tqdm(batch_idx, desc='ASM simulation')):
        s, e = idx
        E = np.fft.ifft2( FFT_A[:, :, None] * np.exp(1.0j*kz[:, :, None]*zs[s:e]), axes=(0,1) )[x1: x2+1, y1:y2+1] #memory spike
        E2 = E; np.abs(E2, out=E2); E2 *= E2
        np.save(f'{path}/{count}.npy', E2.astype(np.float32))

def batch_E2_extract(Nx, Ny, Nz, path, batch_size):
    batch_idx = utils.batch_idx(Nz, batch_size)
    E2 = np.empty([Nx, Ny, Nz], dtype=np.float32)

    for count, idx in enumerate(tqdm(batch_idx, desc='extracting')):
        s, e = idx
        tmp = np.load(f'{path}/{count}.npy')
        E2[:, :, s:e] = tmp
    return E2

def create_shared_array(arr_shape, dtype):
    # remember to close when process end
    # and unlink to release memory
    shm = shared_memory.SharedMemory(create=True, size=np.prod(arr_shape)*np.dtype(dtype).itemsize)
    arr = np.ndarray(arr_shape, dtype=dtype, buffer=shm.buf)
    data = (shm.name, arr_shape, dtype)
    return arr, shm, data
def load_shared_array(arr_shm_name: str, shape, dtype):
    shm = shared_memory.SharedMemory(name=arr_shm_name)
    arr = np.ndarray(shape, dtype, buffer=shm.buf)
    return arr, shm

def __imulti_process_ifft2__(arr_shm_name: str, arr_shape, num_process: int):
    '''
    overall inplace operation of 2D ifft
    arr: <*, Nx, Ny<np.complex128>> last 2 dimension will go through 2Dfft
    arr.dtype must be complex128
    '''
    shared_arr, shm = load_shared_array(arr_shm_name, arr_shape, dtype=np.complex128)
    Nx, Ny = arr_shape[-2:]
    N = np.prod(arr_shape[:-2])
    shared_arr = shared_arr.reshape(-1, Nx, Ny)
    def worker(arr_shm_name, shape, s, e):
        shared_arr, shm = load_shared_array(arr_shm_name, shape, dtype=np.complex128)
        tmp = np.fft.ifft2(shared_arr[s: e, ...], axes=(-2, -1))
        shared_arr[s: e, ...] = tmp
        shm.close()
    process_index = utils.batch_idx(N, batch_size=int(np.ceil(N//num_process)))
    processes = []
    for s, e in process_index:
        processes.append(Process(target=worker, args=(shm.name, (N, Nx, Ny), s, e)))
    for i in range(len(process_index)):
        processes[i].start()
    for i in range(len(process_index)):
        processes[i].join()

def ASM_3D_batch_E2_Multi_Process(A_xy, Lx, Ly, zs, λ, path, batch_size, xy_range_idx=None, num_process=4):
    # only store intensity as float32, reduce storage
    '''
    xy_range_index <tuple<int, int, int, int>> x_start, x_end, y_start, y_end (ends are inclusive)
    '''
    Nx, Ny = A_xy.shape
    x1, x2, y1, y2 = 0, Nx-1, 0, Ny-1
    if xy_range_idx is not None:
        x1, x2, y1, y2 = xy_range_idx
    Nz = len(zs)
    os.system(f'mkdir -p {path}')
    batch_idx = utils.batch_idx(Nz, batch_size)

    k=π*2/λ
    kx = np.fft.fftfreq(Nx, 1/Nx).astype(complex)* 2*π/Lx
    ky = np.fft.fftfreq(Ny, 1/Ny).astype(complex)* 2*π/Ly
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    kz, kz_shm, kz_data = create_shared_array(arr_shape=(Nx, Ny), dtype=np.complex128)
    kz[:] = (k**2-kx**2-ky**2)**0.5 # Nx, Ny
    del kx, ky
    gc.collect()
    FFT_A, FFT_A_shm, FFT_A_data = create_shared_array(arr_shape=(Nx, Ny), dtype=np.complex128)
    FFT_A[:] = np.fft.fft2(A_xy) # Nx, Ny # domain: 0 to N-1

    def worker(E2_data, zs_data, s, e):
        E2, E2_shm = load_shared_array(*E2_data)
        FFT_A, FFT_A_shm = load_shared_array(*FFT_A_data)
        kz, kz_shm = load_shared_array(*kz_data)
        zs, zs_shm = load_shared_array(*zs_data)
        E2[..., s:e] = np.fft.ifft2( FFT_A[:, :, None] * np.exp(1.0j*kz[:, :, None]*zs[s:e]) , axes=(0, 1))
        np.abs(E2[x1: x2+1, y1:y2+1, s:e], out=E2[x1: x2+1, y1:y2+1, s:e])
        E2[x1: x2+1, y1:y2+1, s:e] *= E2[x1: x2+1, y1:y2+1, s:e]
        E2_shm.close()
        FFT_A_shm.close()
        kz_shm.close()
        zs_shm.close()

    for count, idx in enumerate(tqdm(batch_idx, desc='ASM simulation')):
        s, e = idx
        E2, E2_shm, E2_data = create_shared_array(arr_shape=(Nx, Ny, e-s), dtype=np.complex128)
        zs2, zs2_shm, zs2_data = create_shared_array(arr_shape=(e-s,), dtype=np.complex128)
        zs2[:] = zs[s:e]
        process_idx = utils.batch_idx(e-s, int(np.ceil((e-s)/num_process)))
        processes = []
        for p_s, p_e in process_idx:
            processes.append(Process(target=worker, args=( E2_data, zs2_data, p_s, p_e ) ) )
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        np.save(f'{path}/{count}.npy', E2[x1: x2+1, y1:y2+1].astype(np.float32))
        E2_shm.close(); E2_shm.unlink()
        zs2_shm.close(); zs2_shm.unlink()
    kz_shm.close(); kz_shm.unlink()
    FFT_A_shm.close(); FFT_A_shm.unlink()