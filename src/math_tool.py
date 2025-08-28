import numpy as np
def func_2_arr(f, Nx, Ny, Lx, Ly):
    x, y = np.meshgrid(np.linspace(-Lx/2, Lx/2, Nx, endpoint=False), np.linspace(-Ly/2, Ly/2, Ny, endpoint=False), indexing='ij')
    return f(x,y)

def r(x,y,x_c, y_c):
    '''
    x_c <float>
    x <2d-array<float>>
    '''
    return ((x-x_c)**2 + (y-y_c)**2)**0.5

def dist(x, y, z, x_c, y_c, z_c):
    return ((x-x_c)**2+(y-y_c)**2+(z-z_c)**2)**0.5

def FWHM(y, x=None, thres=1/2):
    '''
    y: <id array>
    x: <1d array>
    '''
    if x is None:
        x = np.arange(len(y))

    peak_idx = np.argmax(y)
    peak_y = y[peak_idx]
    thres_y = peak_y*thres

    left = y[:peak_idx+1] # include peak
    left_idx = np.where(left <= thres_y)[0]
    if len(left_idx) == 0:
        print('cannot find thres on left of peak')
        return
    else:
        left_idx = left_idx[-1]# right most element on left

    right = y[peak_idx:] # include peak
    right_idx = np.where(right <= thres_y)[0] 
    if len(right_idx) == 0:
        print('cannot find thres on right of peak')
        return
    else:
        right_idx = right_idx[0] # left most element on right

    left_idx2 = left_idx + 1
    left_y = left[left_idx]
    left_y2 = left[left_idx2]
    ########################
    left_x = x[left_idx]
    left_x2 = x[left_idx2]

    right_idx2 = right_idx - 1
    right_y = right[right_idx]
    right_y2 = right[right_idx2]
    #########################
    right_x = x[peak_idx + right_idx]
    right_x2 = x[peak_idx + right_idx2]

    left_xh = left_x2 + ((thres_y - left_y2) / (left_y - left_y2)) * (left_x - left_x2)
    right_xh = right_x2 + ((thres_y - right_y2) / (right_y - right_y2)) * (right_x - right_x2)
    width = right_xh - left_xh
    return width

# check for correctness
# π = np.pi
# x = np.linspace(-π, π, 1024, endpoint=False)
# y = np.cos(x) + 1
# width = mt.FWHM(y, x)
# print(width)


def neighbour(idx):
    i, j = idx
    return [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]

def connected_2D(arr, seed_idx):
    '''
    arr: <2darray<bool>>
    seed_idx: <tuple<int, int>> index of seed location
    find a set of point (i, j) s.t. its arr[i,j] == True and is connected to seed directly or by other point in the set
    '''
    register = np.zeros_like(arr, dtype=np.int8) # 2: explored, True, 1: exploring, 0: not explored, -1: explored, False
    exploring = [seed_idx]
    register[seed_idx] = 1
    while len(exploring) > 0:
        idx = exploring[0]
        nei = neighbour(idx)
        for i in nei:
            if arr[i]==False: # continue to explore unless False
                register[i] = -1
            elif register[i]==0: #True and not explored
                exploring.append(i)
                register[i] = 1
        register[idx] = 2
        exploring.pop(0)
    connected = (register == 2)
    connected_with_margin = connected + (register==-1)
    return connected, connected_with_margin

def FWHM_power(I, Δx, Δy, thres=1/2):
    '''
    I: <2darray> intensity
    Δx: <float> interval length of 1array unit of intensity array
    find a set of point s.t. its value > value.max()*thres and is connected to max
    then calculate its sum (integrate (intensity * d area) = power)
    '''
    idx = np.unravel_index(np.argmax(I), I.shape)
    connected, connected_with_margin = connected_2D(I>I.max()*thres, idx)
    
    lower_bound = (connected * I).sum()*Δx*Δy # exclude edge that below threshold
    upper_bound = (connected_with_margin * I).sum()*Δx*Δy # include immediate edge that below threshold
    # power = int{ intensity * dA }
    lower_bound_zone = connected
    upper_bound_zone = connected_with_margin

    return lower_bound, upper_bound, lower_bound_zone, upper_bound_zone