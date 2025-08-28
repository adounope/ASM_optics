diffraction simulation using Angular Spectrum Method (ASM).

instruction (ASM simulation):
    place a config.yaml file in ./results/<your_aperture>/
    change config_path in ASM_main.ipynb
    run all code in ASM_main.ipynb

    if not enough RAM:
        lower z_batch_size or num_process in config.yaml

instruction (edit / create aperture):
    pull the needed file from ./make_aperture to ./
    then run all code inside

example:
    ./results/_sample_QR_f5_512 & ./results/_sample_QR_f5_512_zoomx2_padx2
    are provided as example,
    run with ASM_main_QR.ipynb, a QR code will be produced at Z=5mm or 5000μm

    both simulate from the same aperture (but different file)

    _sample_QR_f5_512_zoomx2_padx2 results are more reliable, (but take longer runtime)

    as zero padding aperture file can increase angular resolution (for ASM)
    (increase accuracy of ASM simulation for long range)
    while zooming aperture file can increase simulated angular range, and also target plane resolution

    notice the difference in config.yaml for both setup.