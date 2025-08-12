place a config.yaml file in ./results/<your_aperture>/

config the config.yaml file as here
  # config.yaml
  # all units in μm
  simulation:
    λ: 532e-3   # 532e-3 μm = 532 nm
    resolution: [2**10, 2**10, 5000*1] #Nx, Ny, Nz
    # x_start, x_end, y_start, y_end, z_start, z_end
    range: [-512, 512, -512, 512, 0, 20000]
    z_batch_size: 2**6
  save:
    # simulation data that are saved
    range: [-128, 128, -128, 128] # x_start, x_end, y_start, y_end range to save (in unit μm)

  path:
    results: ./results/f10_donut_100
    aperture: ./aperture/f10_donut_100_plus.bmp

  plot:
    N_ticks: [8, 8, 8]
    dp: 2 #decimal place for ticks

change config_path in ASM_main.ipynb
run all code in ASM_main.ipynb