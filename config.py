import numpy as np
from consts import *

METHOD = "ssf"
GRID_SIZE = 800
WAVELENGTH = 80
POTENTIAL_CHANGE_SPEED = 100
CUTOFF = 800
START = 100
PATH = MOVE_OPPOSITE_SQUARE
REPEATS = 2
NUMBER_OF_LASERS = 5
LASERS = (0, 2, 1, 3)
V_0_REL = 1.45 * NUMBER_OF_LASERS * 10 ** 0  # Convenient to measure potential in recoil units
M = 1
TIME_STEP_REL = 0.2
WAVEPACKET_CENTER_X = 1237
WAVEPACKET_CENTER_Y = 1157
TIME_STEP_REL = 0.2
NOISE = True
FIVE_FOLD = False
NORM = 'ortho'
PLOT_SAVE_DIR_BASE = "/home/gevorg/data/"
CODE_PATH = "/home/gevorg/quasicrystal/"

k = 2 * np.pi / WAVELENGTH
v_rec = k ** 2 / (2 * M)  # working in units of h_bar = 1
v_0 = V_0_REL * v_rec
omega = V_0_REL ** 0.5 * v_rec
sigma_square = 1 / (M * omega)
time_step = TIME_STEP_REL / omega
k_step = 2 * np.pi / GRID_SIZE
