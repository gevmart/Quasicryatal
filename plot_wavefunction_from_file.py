import numpy as np

from config import *
from plotting import heatmap
from utils import x, y

wavef = np.load("{}move_opposite_square_x_0_y_0_ssf_n_500_cutoff_100_grid_300_wavelength_80_timestep_0.2_lasernum_4_repeat_2_retroreflective_True/Modulation finished_wavefunction.npy".format(PLOT_SAVE_DIR_BASE), allow_pickle=True)
heatmap(np.abs(wavef) ** 2, x, y)