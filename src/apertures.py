# store all various aperture (analytical) function
import numpy as np
import src.math_tool as mt

π = np.pi
def double_slit(x, y, s, d):
    '''
    s: slit separation
    d: slit width
    '''
    x0, x1 = -s/2, s/2
    val = (np.abs(x-x0) < d/2) + (np.abs(x-x1) < d/2)
    return (val==True)*1

def concentric3(x, y):
    r = mt.r(x, y, 0, 0)
    w = np.array([6, 3, 2])
    rr = np.array([64, 128, 192]) # radius center of ring
    val = True
    for i in range(len(w)):
        val *= (np.abs(r-rr[i]) > w[i])
    return (val==True)*1

def concentric3_artifical_f10mm(x, y): # ring size: 64, 121.4, 159.3
    r = mt.r(x, y, 0, 0)
    rr = np.array([64, 121.4, 159.3]) # radius center of ring
    w = 10*64/rr
    val = True
    for i in range(len(w)):
        val *= (np.abs(r-rr[i]) > w[i])
    return (val==True)*1

def concentricN_artifical_f10mm(x, y):
    r = mt.r(x, y, 0, 0)
    rr = np.array([210, 183, 151, 110]) # radius center of ring
    w = 10*110/rr
    val = True
    for i in range(len(w)):
        val *= (np.abs(r-rr[i]) > w[i])
    return (val==True)*1

def angular_dots(x, y):
    r_d = 20 #radius of dot in μm

    rs = np.array([100, 100, 100]) #radius of dot center location
    θs = np.array([0, 1/3, 2/3]) * 2 * π # angular location of dot center locatio
    val = True
    for i in range(len(rs)):
        val *= (mt.r(x, y, rs[i] * np.cos(θs[i]), rs[i] * np.sin(θs[i])) > r_d)
    return (val==True)*1

def block_edge_mask(Nx, Ny, radius):
    '''
    radius <float> in unit of array size
    return:
        mask <2darray>
    '''
    center_x = (Nx-1)/2
    center_y = (Ny-1)/2
    x = np.linspace(0, Nx, Nx, endpoint=False)
    y = np.linspace(0, Ny, Ny, endpoint=False)
    x, y = np.meshgrid(x, y, indexing='ij')
    r = mt.r(x, y, center_x, center_y)
    mask = (r < radius)
    return mask