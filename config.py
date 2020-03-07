import numpy as np

METHOD = "ssf"
GRID_SIZE = 500
WAVELENGTH = 100
V_0_REL = 5 * 4 * 10 ** 0  # Convenient to measure potential in recoil units
M = 1
WAVEPACKET_CENTER_X = 104
WAVEPACKET_CENTER_Y = 104
TIME_STEP_REL = 0.02
NORM = 'ortho'
PLOT_SAVE_DIR_BASE = "/home/ubuntu/environment/quasicrystal/"

k = 2 * np.pi / WAVELENGTH
v_rec = k ** 2 / (2 * M)  # working in units of h_bar = 1
v_0 = V_0_REL * v_rec
omega = V_0_REL ** 0.5 * v_rec
sigma_square = 1 / (M * omega)
time_step = TIME_STEP_REL / omega
k_step = 2 * np.pi / GRID_SIZE
