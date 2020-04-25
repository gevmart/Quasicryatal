import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os

from config import *
from plotting import heatmap, annotate
from utils import x, y
from potential import generate_potential, POTENTIAL_CHANGE_SPEED, CUTOFF, minima, mins_only, path_map

# directory_to_save = "{}{}_potential_{}_x_{}_y_{}_{}_n_{}_cutoff_{}_grid_{}_wavelength_{}_timestep_{}_lasernum_{}_repeat_{}_retroreflective_{}/".format(
#     PLOT_SAVE_DIR_BASE, PATH, V_0_REL / NUMBER_OF_LASERS, WAVEPACKET_CENTER_X, WAVEPACKET_CENTER_Y, METHOD,
#     POTENTIAL_CHANGE_SPEED, CUTOFF, GRID_SIZE, WAVELENGTH, TIME_STEP_REL, NUMBER_OF_LASERS, REPEATS,
#     not NON_RETROREFLECTIVE)
#
# p = Path("{}otwell".format(directory_to_save))
#
# with p.open('rb') as f:
#     fsz = os.fstat(f.fileno()).st_size
#     out = np.load(f)
#     while f.tell() < fsz:
#         out = np.vstack((out, np.load(f)))
# print(out.reshape(out.shape[0] // 5, 5, 5))

wavef = np.load("{}move_parallelogram_potential_2.2_x_878_y_1160_cn_n_5_cutoff_10_grid_400_wavelength_80_timestep_0.2_lasernum_4_repeat_1_retroreflective_True/otwell".format(PLOT_SAVE_DIR_BASE), allow_pickle=True)
fig, ax = plt.subplots()
# im_pot = heatmap(generate_potential(0) / v_rec, x / WAVELENGTH, y / WAVELENGTH,
#                           ax, cbarlabel="Potential / Recoil Energy", cmap=plt.cm.gray)

heatmap(np.abs(wavef) ** 2, x / WAVELENGTH, y / WAVELENGTH, cbarlabel="Probability Distribution")
annotate(fig, ax, "Probability distribution at the finish of modulation", r"$x/\lambda$", r"$y/\lambda$")


# heatmap(np.abs(np.fft.fftshift(np.fft.fft2(wavef, norm=NORM))) ** 2, x, y)