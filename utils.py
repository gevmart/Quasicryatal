import os
import numpy as np
from shutil import copy

from config import GRID_SIZE, PLOT_SAVE_DIR_BASE
from config import PLOT_SAVE_DIR_BASE


# %%
x, y = np.meshgrid(
    np.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, num=GRID_SIZE),
    np.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, num=GRID_SIZE))


def copy_code(directory, create=False):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    for file in os.listdir(PLOT_SAVE_DIR_BASE):
        if file.endswith(".py"):
            copy(PLOT_SAVE_DIR_BASE + file, directory)


def calc_center_of_mass(wavefunction):
    return np.sum(np.abs(wavefunction) ** 2 * x), np.sum(np.abs(wavefunction) ** 2 * y)


def calc_root_mean_square(wavefunction):
    x_c, y_c = calc_center_of_mass(wavefunction)
    return np.sum(np.abs(wavefunction) ** 2 * ((x - x_c) ** 2 + (y - y_c) ** 2)) ** 0.5


def default_notify(message):
    print("Notified with a message {}".format(message))
