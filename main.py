import numpy as np
import time
from matplotlib import pyplot as plt, colors, animation

import plotting
from config import time_step, WAVELENGTH, PLOT_SAVE_DIR_BASE, V_0_REL, v_rec
from grid import potential, x, y, minima, generate_potential, mins_only
from generate_wavepacket import wavepacket
from time_propagation_ssf import propagate_ssf
from time_propagation_cn import propagate_cn
from utils import copy_code, calc_center_of_mass

# %%
METHOD = "cn"

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

    def animate(_):
        global wavefunction, t
        wavefunction = propagate(wavefunction, t, n, v)
        t += time_step * n
        im_pot.set_data(generate_potential(t, v) / v_rec)
        probability_density = np.absolute(wavefunction) ** 2
        im_wave.set_data(probability_density)
        im_wave.set_clim(np.max(probability_density), np.min(probability_density))
        print(calc_center_of_mass(wavefunction))
        return im_wave

    ani = animation.FuncAnimation(fig, animate, frames=200, repeat=True, interval=100)

    # plt.show()


def benchmark(n):
    start_time = time.time()
    propagate(wavepacket, 0, n)
    print("Time taken to complete {} steps is {} seconds".format(n, time.time() - start_time))


calcualte_and_plot()
# directory = "{}locality_check_with_potential_{}_{}/".format(PLOT_SAVE_DIR_BASE, 1.0 * i / 4, METHOD)
# copy_code(directory)
# ani.save("{}animation.gif".format(directory), writer='imagemagick', fps=10)
