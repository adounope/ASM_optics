import pickle
import numpy as np

def save_obj(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def batch_idx(N, batch_size):
    tmp = np.linspace(0, np.ceil(N/batch_size), int(np.ceil(N/batch_size))+1, endpoint=True, dtype=int) * batch_size
    return [(tmp[i], np.minimum(tmp[i+1], N)) for i in range(len(tmp)-1)]