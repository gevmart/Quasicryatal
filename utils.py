import os
import numpy as np
from shutil import copy

from grid import x, y
from config import PLOT_SAVE_DIR_BASE


def copy_code(directory, create=False):
    code_base_dir = "/home/ubuntu/environment/quasicrystal/"
    if not os.path.isdir(directory):
        os.mkdir(directory)
    for file in os.listdir(code_base_dir):
        if file.endswith(".py"):
            copy(code_base_dir + file, directory)


def calc_center_of_mass(wavefunction):
    return np.sum(np.abs(wavefunction) ** 2 * x), np.sum(np.abs(wavefunction) ** 2 * y)


def calc_root_mean_square(wavefunction):
    x_c, y_c = calc_center_of_mass(wavefunction)
    return np.sum(np.abs(wavefunction) ** 2 * ((x - x_c) ** 2 + (y - y_c) ** 2)) ** 0.5
