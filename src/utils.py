import pickle
import numpy as np
from multiprocessing import Process, shared_memory

def save_obj(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def batch_idx(N, batch_size):
    tmp = np.linspace(0, np.ceil(N/batch_size), int(np.ceil(N/batch_size))+1, endpoint=True, dtype=int) * batch_size
    return [(tmp[i], np.minimum(tmp[i+1], N)) for i in range(len(tmp)-1)]

def create_shared_array(arr_shape: tuple, dtype):
    # remember to close when process end
    # and unlink to release memory
    '''
    usage:
    arr, arr_shm, arr_data = create_shared_array(arr_shape=  , dtype=  )
    '''
    shm = shared_memory.SharedMemory(create=True, size=np.prod(arr_shape)*np.dtype(dtype).itemsize)
    arr = np.ndarray(arr_shape, dtype=dtype, buffer=shm.buf)
    data = (shm.name, arr_shape, dtype)
    return arr, shm, data
def load_shared_array(arr_shm_name: str, shape, dtype):
    '''
    usage:
    arr, arr_shm = load_shared_array(*arr_data)
    '''
    shm = shared_memory.SharedMemory(name=arr_shm_name)
    arr = np.ndarray(shape, dtype, buffer=shm.buf)
    return arr, shm