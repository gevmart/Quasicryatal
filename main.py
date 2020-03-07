import numpy as np
import os
import time
from matplotlib import pyplot as plt, colors, animation

import plotting
from config import time_step, WAVELENGTH, PLOT_SAVE_DIR_BASE, V_0_REL, v_rec
from grid import potential, x, y, minima, generate_potential, mins_only
from generate_wavepacket import wavepacket
from time_propagation_ssf import propagate_ssf
from time_propagation_cn import propagate_cn
from utils import copy_code
from testing_simulation import expansion_with_different_potential_strengths, expansion_without_potential

# %%
METHOD = "ssf"

propagate = propagate_cn if METHOD == "cn" else propagate_ssf

wavefunction = wavepacket
t = 0
var_t = np.zeros(100)


def calcualte_and_plot(v=float('nan')):
    global ani

    fig, ax = plt.subplots()
    im_pot = plotting.heatmap(generate_potential(0, v) / v_rec, x / WAVELENGTH, y / WAVELENGTH,
                              ax, cbarlabel="Potential / Recoil Energy", cmap=plt.cm.gray)
    im_wave = plotting.heatmap(np.absolute(wavepacket) ** 2, x / WAVELENGTH, y / WAVELENGTH, ax,
                               cbarlabel="Normalized Wavefunction", cmap='alpha')
    plotting.annotate(fig, ax, "Wavefunction evolution", r"$x/\lambda$", r"$y/\lambda$")
    # plotting.heatmap(minima, x / WAVELENGTH, y / WAVELENGTH, ax, cmap='alpha')
    n = 10

    def animate(i):
        global wavefunction, t
        wavefunction, com, rms = propagate(wavefunction, t, n, v=12)
        t += time_step * n
        im_pot.set_data(generate_potential(t, v) / v_rec)
        probability_density = np.absolute(wavefunction) ** 2
        im_wave.set_data(probability_density)
        im_wave.set_clim(np.max(probability_density), np.min(probability_density))
        print(com, rms)
        return im_wave

    ani = animation.FuncAnimation(fig, animate, frames=200, repeat=True, interval=100)

    # plt.show()


def benchmark(n):
    start_time = time.time()
    propagate(wavepacket, 0, n)
    print("Time taken to complete {} steps is {} seconds".format(n, time.time() - start_time))


def save_com_to_file(steps, v=float('nan')):
    avg = np.array([0.0, 0.0])
    wavef = wavepacket
    n = 10
    t = 0

    directory_to_save = "{}square_single_phases_side_minimum_n=15_{}/".format(PLOT_SAVE_DIR_BASE, METHOD)
    copy_code(directory_to_save)

    for i in np.arange(steps):
        wavef, com, rms = propagate(wavef, t, n, v)
        t += n * time_step
        avg = (avg * (i % 10) + com) / (i % 10 + 1)
        if i % 10 == 9:
            with open("{}data.txt".format(directory_to_save), 'a') as file:
                file.write(str(avg) + "   " + str(t) + "  " + str(time.time()) + os.linesep)
            avg = np.array([0.0, 0.0])


print(time.time())
# save_com_to_file(1200)
# calcualte_and_plot()
expansion_with_different_potential_strengths(100, propagate, METHOD)
print(time.time())
# ani.save("{}animation.gif".format(directory), writer='imagemagick', fps=10)
