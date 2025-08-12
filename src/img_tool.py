
# input: image path
# return aperture array
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# modify ffmpeg_path here
matplotlib.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

def img_2_array(path):
    img_array = np.array(Image.open(path))
    return img_array
def array_2_img(arr, path):
    img = Image.fromarray(arr)
    img.save(path)

def img_pad(arr, N_pad_xy = (1, 1, 1, 1), pad_val = 0):
    Nx, Ny = arr.shape
    pad_x_plus, pad_x_minus, pad_y_plus, pad_y_minus = N_pad_xy
    padded_arr = np.ones((Nx+pad_x_plus+pad_x_minus, Ny+pad_y_plus+pad_y_minus), dtype=arr.dtype) * pad_val
    padded_arr[pad_x_plus:-pad_x_minus, pad_y_plus:-pad_y_minus] = arr
    return padded_arr
def arr_2_mp4(arr, path, fps = 30, title=lambda t: str(t), x_label='', y_label=''):
    '''
    arr <3d array>: t, x, y
    '''
    fig, ax = plt.subplots()

    heatmap = ax.imshow(arr[0].T, cmap='plasma', vmin=arr.min(), vmax=arr.max())
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.tight_layout(pad=3)
    print(f'animation_duration: {len(arr) / fps}s, fps: {fps}')
    # plt.colorbar(heatmap)
    # plt.title('Heatmap Animation from 3D Array')
    def update(frame):
        heatmap.set_array(arr[frame].T)
        ax.set_title(title(frame))
        return heatmap,
    ani = FuncAnimation(fig, update, frames=arr.shape[0], blit=True)
    ani.save(path, writer=FFMpegWriter(fps=fps))
    plt.close(fig)

def conv_lambda2D(arr, kernel_shape = (3,3), f = lambda x: x.mean()):
    shape = arr.shape # *, Nx, Ny
    kx, ky = kernel_shape
    out_dim = (shape[-2] - kx + 1, shape[-1] - ky + 1)
    conv_out = np.zeros((*shape[:-2], out_dim[0], out_dim[1])).astype(arr.dtype) # *, Nx+1-kx, Ny+1-ky
    for i in range(out_dim[0]):
        for j in range(out_dim[1]):
            conv_out[i, j] = f(arr[i:i+kx, j:j+ky])
    return conv_out