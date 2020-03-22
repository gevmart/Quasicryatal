import numpy as np
from consts import *

METHOD = "ssf"
GRID_SIZE = 500
WAVELENGTH = 100
POTENTIAL_CHANGE_SPEED = 5
CUTOFF = 4
START = 3
PATH = MOVE_SQUARE
REPEATS = 2
NUMBER_OF_LASERS = 5
V_0_REL = 3 * 4 * 10 ** 0  # Convenient to measure potential in recoil units
M = 1
WAVEPACKET_CENTER_X = 62
WAVEPACKET_CENTER_Y = 62
TIME_STEP_REL = 0.1
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
