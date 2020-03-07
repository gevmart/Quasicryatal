import os
import numpy as np
from shutil import copy

from grid import x, y


def copy_code(directory, create=False):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    for file in os.listdir("./"):
        if file.endswith(".py"):
            copy(file, directory)


def calc_center_of_mass(wavefunction):
    return np.sum(np.abs(wavefunction) ** 2 * x), np.sum(np.abs(wavefunction) ** 2 * y)


def calc_root_mean_square(wavefunction):
    x_c, y_c = calc_center_of_mass(wavefunction)
    return np.sum(np.abs(wavefunction) ** 2 * ((x - x_c) ** 2 + (y - y_c) ** 2)) ** 0.5
