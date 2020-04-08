import numpy as np
from consts import *

METHOD = "ssf"
GRID_SIZE = 300
WAVELENGTH = 80
POTENTIAL_CHANGE_SPEED = 20
CUTOFF = 20
START = 5
PATH = MOVE_OPPOSITE_SQUARE
REPEATS = 2
NUMBER_OF_LASERS = 5
LASERS = (0, 2, 1, 3)
V_0_REL = 3 * NUMBER_OF_LASERS * 10 ** 0  # Convenient to measure potential in recoil units
M = 1
WAVEPACKET_CENTER_X = 0
WAVEPACKET_CENTER_Y = 0
TIME_STEP_REL = 0.2
NOISE = False
FIVE_FOLD = True
NORM = 'ortho'
PLOT_SAVE_DIR_BASE = "/Users/gevorg/workspace/PartIII/Project/Plots/"
CODE_PATH = "/Users/gevorg/PycharmProjects/Quasicrystal/"

k = 2 * np.pi / WAVELENGTH
v_rec = k ** 2 / (2 * M)  # working in units of h_bar = 1
v_0 = V_0_REL * v_rec
omega = V_0_REL ** 0.5 * v_rec
sigma_square = 1 / (M * omega)
time_step = TIME_STEP_REL / omega
k_step = 2 * np.pi / GRID_SIZE
