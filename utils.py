import os
import numpy as np
from shutil import copy

from config import GRID_SIZE, WAVELENGTH, CODE_PATH


# %%
x, y = np.meshgrid(
    np.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, num=GRID_SIZE),
    np.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, num=GRID_SIZE))


def copy_code(directory, create=False):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    for file in os.listdir(CODE_PATH):
        if file.endswith(".py"):
            copy(CODE_PATH + file, directory)


def calc_center_of_mass(wavefunction):
    return np.sum(np.abs(wavefunction) ** 2 * x), np.sum(np.abs(wavefunction) ** 2 * y)


def calc_root_mean_square(wavefunction):
    x_c, y_c = calc_center_of_mass(wavefunction)
    return np.sum(np.abs(wavefunction) ** 2 * ((x - x_c) ** 2 + (y - y_c) ** 2)) ** 0.5


def default_notify(message):
    print("Notified with a message {}".format(message))


def apply(pair):
    """
    Applies the first element of tuple (a function) on the second one
    :param pair: a tuple in a form (fn, item)
    :return: fn(item)
    """
    return pair[0](pair[1])


def identity(x):
    return x


def probability_at_edges(wavefunction, edge_length=WAVELENGTH // 2):
    """
    Computes the probability of the wavefunction being in the edge of the box of thickness edge_length
    :param wavefunction: the wavefunction
    :param edge_length: the size of the edge
    :return: probability of being in the edge
    """
    return np.sum(np.abs(wavefunction) ** 2) - \
           np.sum(np.abs(wavefunction[edge_length:-edge_length, edge_length:-edge_length]) ** 2)


def probability_left_edge(wavefunction, edge_length=WAVELENGTH // 2):
    """
    Computes the probability of the wavefunction being in the left edge of the box of thickness edge_length
    :param wavefunction: the wavefunction
    :param edge_length: the size of the edge
    :return: probability of being in the left edge
    """
    return np.sum(np.abs(wavefunction) ** 2) - \
           np.sum(np.abs(wavefunction[edge_length:, :]) ** 2)


def probability_right_edge(wavefunction, edge_length=WAVELENGTH // 2):
    """
    Computes the probability of the wavefunction being in the right edge of the box of thickness edge_length
    :param wavefunction: the wavefunction
    :param edge_length: the size of the edge
    :return: probability of being in the right edge
    """
    return np.sum(np.abs(wavefunction) ** 2) - \
           np.sum(np.abs(wavefunction[:-edge_length, :]) ** 2)


def probability_upper_edge(wavefunction, edge_length=WAVELENGTH // 2):
    """
    Computes the probability of the wavefunction being in the upper edge of the box of thickness edge_length
    :param wavefunction: the wavefunction
    :param edge_length: the size of the edge
    :return: probability of being in the upper edge
    """
    return np.sum(np.abs(wavefunction) ** 2) - \
           np.sum(np.abs(wavefunction[:, edge_length:]) ** 2)


def probability_lower_edge(wavefunction, edge_length=WAVELENGTH // 2):
    """
    Computes the probability of the wavefunction being in the lower edge of the box of thickness edge_length
    :param wavefunction: the wavefunction
    :param edge_length: the size of the edge
    :return: probability of being in the lower edge
    """
    return np.sum(np.abs(wavefunction) ** 2) - \
           np.sum(np.abs(wavefunction[:, :-edge_length]) ** 2)


def second_derivative(grid, x_0, y_0, along_y=False):
    center_x = GRID_SIZE // 2 + x_0
    center_y = GRID_SIZE // 2 + y_0
    if along_y:
        return grid[center_x, center_y + 1] - 2 * grid[center_x, center_y] + grid[center_x, center_y - 1]
    return grid[center_x + 1, center_y] - 2 * grid[center_x, center_y] + grid[center_x - 1, center_y]
    # if along_y:
    #     return -1 / 12 * (grid[center_x, center_y + 2] - 16 * grid[center_x, center_y + 1]
    #                       + 30 * grid[center_x, center_y] - 16 * grid[center_x, center_y - 1] + grid[center_x, center_y - 2])
    #
    # return -1 / 12 * (grid[center_x + 2, center_y] - 16 * grid[center_x + 1, center_y]
    #                   + 30 * grid[center_x, center_y] - 16 * grid[center_x - 1, center_y] + grid[
    #                       center_x - 2, center_y])
