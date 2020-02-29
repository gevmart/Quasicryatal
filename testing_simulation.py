import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.stats import linregress
from shutil import copy

from config import time_step, PLOT_SAVE_DIR_BASE
from generate_wavepacket import wavepacket, sigma_square
from grid import x, y
from utils import copy_code


def expansion_without_potential(steps, method, propagate):
    t = 0
    var_t = np.zeros(steps)
    wavefunction = wavepacket
    for i in range(steps):
        wavefunction = propagate(wavefunction, t, 2)
        var_t[i] = np.sum(np.abs(wavefunction) ** 2 * (x ** 2 + y ** 2))
        t += time_step * 2

    slope, intercept, r_value, p_value, std_err = linregress(np.arange(steps), (var_t - sigma_square) ** 0.5)
    print(1 - r_value)

    plt.plot(np.arange(0, steps), (var_t - sigma_square) ** 0.5)
    plt.plot(np.arange(1, steps), np.arange(1, steps) * slope + intercept)
    directory_to_save = "{}no_potential_expansion_{}".format(PLOT_SAVE_DIR_BASE, method)
    copy_code(directory_to_save)
    plt.savefig("{}/plt.png".format(directory_to_save))
    plt.show()
