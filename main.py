import numpy as np
import os
import time
from matplotlib import pyplot as plt, colors, animation

import plotting
from config import *
from potential import generate_potential, POTENTIAL_CHANGE_SPEED, CUTOFF, minima, mins_only
from generate_wavepacket import wavepacket
from time_propagation_ssf import propagate_ssf
from time_propagation_cn import propagate_cn
from utils import *
from testing_simulation import expansion_with_different_potential_strengths, expansion_without_potential

# %%
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
        print(probability_at_edges(wavefunction, 50), "left", probability_left_edge(wavefunction), "right",
              probability_right_edge(wavefunction), "up", probability_upper_edge(wavefunction), "down",
              probability_lower_edge(wavefunction))
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

    directory_to_save = "{}{}_potential_{}_x_{}_y_{}_{}_n_{}_cutoff_{}_grid_{}_wavelength_{}_timestep_{}_lasernum_{}_repeat_{}/".format(PLOT_SAVE_DIR_BASE, PATH, V_0_REL / NUMBER_OF_LASERS, WAVEPACKET_CENTER_X, WAVEPACKET_CENTER_Y, METHOD, POTENTIAL_CHANGE_SPEED, CUTOFF, GRID_SIZE, WAVELENGTH, TIME_STEP_REL, NUMBER_OF_LASERS, REPEATS)
    copy_code(directory_to_save)

    def write_notification_to_file(message):
        with open("{}data.txt".format(directory_to_save), 'a') as file:
            file.write(message + os.linesep)

    for i in np.arange(steps):
        wavef, com, rms = propagate(wavef, t, n, v, notify=write_notification_to_file)
        t += n * time_step
        avg = (avg * (i % 10) + com) / (i % 10 + 1)
        if i % 10 == 9:
            with open("{}data.txt".format(directory_to_save), 'a') as file:
                file.write(str(avg) + "   " + str(rms) + "   " + str(t) + "  " + str(time.time()) + "   " +
                           str(probability_at_edges(wavef)) + "   " + str(probability_left_edge(wavef)) + "   " +
                           str(probability_right_edge(wavef)) + "   " + str(probability_upper_edge(wavef)) + "   "
                           + str(probability_lower_edge(wavef)) + os.linesep)
            avg = np.array([0.0, 0.0])


print(time.time())
save_com_to_file(4000)
# calcualte_and_plot()
# expansion_with_different_potential_strengths(100, propagate, METHOD)
print(time.time())
# ani.save("{}animation.gif".format(directory), writer='imagemagick', fps=10)
