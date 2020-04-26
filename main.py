import numpy as np
from pathlib import Path
import os
import time
from matplotlib import pyplot as plt, colors, animation

import plotting
from config import *
from potential import generate_potential, POTENTIAL_CHANGE_SPEED, CUTOFF, minima, mins_only, path_map
from generate_wavepacket import wavepacket
from time_propagation_ssf import propagate_ssf
from time_propagation_cn import propagate_cn
from utils import *
from testing_simulation import expansion_with_different_potential_strengths, expansion_without_potential
from hermites import generate_hermite
from generate_eigenstates import get_eigenstates

# %%
propagate = propagate_cn if METHOD == "cn" else propagate_ssf

wavefunction = wavepacket
t = 0
var_t = np.zeros(100)


def calcualte_and_plot(v=float('nan')):
    global ani

    fig, ax = plt.subplots()
    im_pot = plotting.heatmap(generate_potential(0, v) / v_rec, x / WAVELENGTH, y / WAVELENGTH,
                              ax, cbarlabel="Potential / Recoil Energy", cmap=plt.cm.gray, fontsize=22)
    # im_wave = plotting.heatmap(np.absolute(wavepacket) ** 2, x / WAVELENGTH, y / WAVELENGTH, ax, cmap='alpha')
    plotting.annotate(fig, ax, "Starting point of modulation", r"$x/\lambda$", r"$y/\lambda$", fontsize=22)
    # start_point = np.zeros((GRID_SIZE, GRID_SIZE))
    # center_x = GRID_SIZE // 2 - (929 - WAVEPACKET_CENTER_X)
    # center_y = GRID_SIZE // 2 - (1093 - WAVEPACKET_CENTER_Y)
    # center_x = GRID_SIZE // 2 + WAVEPACKET_CENTER_X
    # center_y = GRID_SIZE // 2 + WAVEPACKET_CENTER_Y
    # start_point[center_y - 3:center_y + 3, center_x - 3:center_x + 3] = 1
    # plotting.heatmap(start_point, x / WAVELENGTH, y / WAVELENGTH, ax, cmap='alpha')
    # plotting.heatmap(minima, x / WAVELENGTH, y / WAVELENGTH, ax, cmap='alpha')
    n = 10

    num_of_eigenstates = 10
    eigenstates_orwell = get_eigenstates()
    eigenstates_otwell = get_eigenstates(center_x=OTHER_MIN_Y, center_y=OTHER_MIN_X)
    herm_otwell = np.zeros(num_of_eigenstates)
    herm_orwell = np.zeros(num_of_eigenstates)
    for i in np.arange(num_of_eigenstates):
        herm_orwell[i] = np.abs(np.sum(eigenstates_orwell[i] * wavefunction)) ** 2
        herm_otwell[i] = np.abs(np.sum(eigenstates_otwell[i] * wavefunction)) ** 2
    im_wave = plotting.heatmap(np.absolute(eigenstates_otwell[0]) ** 2, x / WAVELENGTH, y / WAVELENGTH, ax, cmap='alpha')
    print(herm_orwell)
    print(herm_otwell)
    def animate(i):
        global wavefunction, t
        wavefunction, com, rms = propagate(wavefunction, t, n)
        t += time_step * n
        im_pot.set_data(generate_potential(t, v) / v_rec)
        probability_density = np.absolute(wavefunction) ** 2
        im_wave.set_data(probability_density)
        # im_wave.set_clim(np.max(probability_density), np.min(probability_density))
        print(com, rms)
        print(np.sum(np.abs(wavefunction) ** 2))
        print("Overlap with itself" + str(np.sum(np.abs(wavefunction ** 2))))
        print("Overlap with (0, 0) Hermite" + str(
            np.abs(np.sum(generate_hermite(0, 0, 128, 0) * wavefunction)) ** 2))

        print(probability_at_edges(wavefunction, 50), "left", probability_left_edge(wavefunction), "right",
              probability_right_edge(wavefunction), "up", probability_upper_edge(wavefunction), "down",
              probability_lower_edge(wavefunction))
        return

    # ani = animation.FuncAnimation(fig, animate, frames=200, repeat=True, interval=100)
    # ani.save("{}animation.gif".format("./"), writer='imagemagick', fps=10)

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

    num_of_eigenstates = 10
    eigenstates_orwell = get_eigenstates()
    eigenstates_otwell = get_eigenstates(center_x=OTHER_MIN_Y, center_y=OTHER_MIN_X)

    directory_to_save = "{}{}_potential_{}_x_{}_y_{}_{}_n_{}_cutoff_{}_grid_{}_wavelength_{}_timestep_{}_lasernum_{}_repeat_{}_retroreflective_{}/".format(PLOT_SAVE_DIR_BASE, PATH, V_0_REL / NUMBER_OF_LASERS, WAVEPACKET_CENTER_X, WAVEPACKET_CENTER_Y, METHOD,POTENTIAL_CHANGE_SPEED, CUTOFF, GRID_SIZE, WAVELENGTH, TIME_STEP_REL, NUMBER_OF_LASERS, REPEATS, not NON_RETROREFLECTIVE)
    copy_code(directory_to_save)

    def write_notification_to_file(message, write_wavefunction=True):
        with open("{}data.txt".format(directory_to_save), 'a') as file:
            file.write(message + os.linesep)
        print(message)
        if write_wavefunction:
            np.save("{}{}_wavefunction".format(directory_to_save, message), wavef)

    for i in np.arange(steps):
        k = 0
        wavef, com, rms = propagate(wavef, t, n, v, notify=write_notification_to_file)
        t += n * time_step
        avg = (avg * (i % 10) + com) / (i % 10 + 1)
        if i % 10 == 9:
            with open("{}data.txt".format(directory_to_save), 'a') as file:
                file.write(str(avg) + "   " + str(rms) + "   " + str(t) + "  " + str(time.time()) + "   " +
                           str(probability_at_edges(wavef)) + "   " + str(probability_left_edge(wavef)) + "   " +
                           str(probability_right_edge(wavef)) + "   " + str(probability_upper_edge(wavef)) + "   "
                           + str(probability_lower_edge(wavef)) + os.linesep)

            if t * omega - START > REPEATS * CUTOFF * path_map[PATH][1]:
                herm_otwell = np.zeros(num_of_eigenstates)
                herm_orwell = np.zeros(num_of_eigenstates)
                for i in np.arange(num_of_eigenstates):
                    herm_orwell[i] = np.abs(np.sum(eigenstates_orwell[i] * wavef)) ** 2
                    herm_otwell[i] = np.abs(np.sum(eigenstates_otwell[i] * wavef)) ** 2
                p = Path("{}otwell".format(directory_to_save))
                with p.open('ab') as f:
                    np.save(f, herm_otwell)
                p = Path("{}orwell".format(directory_to_save))
                with p.open('ab') as f:
                    np.save(f, herm_orwell)

            avg = np.array([0.0, 0.0])

        if t * omega - START > REPEATS * CUTOFF * path_map[PATH][1] and i % 100 == 99:
            np.save("{}{}_wavefunction".format(directory_to_save, k), wavef)
            k += 1


print(time.time())
save_com_to_file(7000)
# calcualte_and_plot()
# expansion_with_different_potential_strengths(100, propagate, METHOD)
print(time.time())
