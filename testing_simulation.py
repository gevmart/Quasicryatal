import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.stats import linregress

from config import time_step
from generate_wavepacket import wavepacket, sigma_square


def expansion_without_potential(steps, propagate, method):
    t = 0
    var_t = np.zeros(steps)
    wavefunction = wavepacket
    for i in range(steps):
        wavefunction, avg, rms = propagate(wavefunction, t, 2, v=0)
        var_t[i] = rms ** 2
        t += time_step * 2

    slope, intercept, r_value, p_value, std_err = linregress(np.arange(steps), (var_t - sigma_square) ** 0.5)
    # print(1 - r_value)

    plt.plot(np.arange(0, steps), (var_t - sigma_square) ** 0.5)
    plt.plot(np.arange(1, steps), np.arange(1, steps) * slope + intercept)
    # directory_to_save = "{}no_potential_expansion_{}".format(PLOT_SAVE_DIR_BASE, method)
    # copy_code(directory_to_save)
    # plt.savefig("{}/plt.png".format(directory_to_save))
    plt.show()


def expansion_with_different_potential_strengths(steps, propagate, method='ssf'):
    limit = 6
    var_t = []
    size = lambda j: steps * (max(0, j - 1) + max(0, j - 2) - 2 * max(0, j - 4) + 1)
    for j in np.arange(0, limit + 1):
        t = 0
        wavefunction = wavepacket
        current_rms = np.zeros(size(j))
        for i in np.arange(size(j)):
            wavefunction, avg, rms = propagate(wavefunction, t, 2, v=j * 2)
            current_rms[i] = rms ** 2
            print(rms)
            t += time_step * 2
        var_t.append(current_rms)
    var_t = np.array(var_t)
    for j in np.arange(0, limit + 1):
        plt.plot(np.arange(size(j)), (var_t[j] - sigma_square) ** 0.5,
                 label="Potential: {} recoil".format(1.0 * j / 2))

    plt.legend(loc='upper left')
    plt.title("Expansion of the wavefunction at different potentials")
    plt.xlabel("Time in arbitrary units")
    plt.ylabel("Root mean square * 100 / Wavelength")
