import numpy as np

from config import *
from plotting import heatmap
from utils import x, y

wavef = np.load("{}move_triangle_potential_2.0_x_762_y_1160_ssf_n_101_cutoff_800_grid_800_wavelength_80_timestep_0.2_lasernum_4_repeat_2_retroreflective_True/Modulation finished_wavefunction.npy".format(PLOT_SAVE_DIR_BASE), allow_pickle=True)
heatmap(np.abs(wavef) ** 2, x, y)