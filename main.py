from matplotlib import pyplot as plt, colors, animation
import numpy as np
import time

import plotting
from config import time_step, WAVELENGTH
from grid import potential, x, y, minima, generate_potential
from generate_wavepacket import wavepacket
from time_propagation_ssf import propagate_ssf
from time_propagation_cn import propagate_cn

# %%
METHOD = "cn"

propagate = propagate_cn if METHOD == "cn" else propagate_ssf

wavefunction = wavepacket
t = 0


def calcualte_and_plot():
    global ani

    fig, ax = plt.subplots()
    im_pot = plotting.heatmap(potential, x / WAVELENGTH, y / WAVELENGTH, ax, cbarlabel="Potential", cmap=plt.cm.gray)
    im_wave = plotting.heatmap(np.absolute(wavepacket) ** 2, x / WAVELENGTH, y / WAVELENGTH, ax,
                               cbarlabel="Normalized Wavefunction", cmap='alpha')
    plotting.annotate(fig, ax, "Potential", r"$x/\lambda$", r"$y/\lambda$")
    # plotting.heatmap(minima, x / WAVELENGTH, y / WAVELENGTH, ax, cmap='alpha')
    n = 10

    def animate(i):
        global wavefunction, t
        wavefunction = propagate(wavefunction, t, n)
        t += time_step * n
        im_pot.set_data(generate_potential(t))
        probability_density = np.absolute(wavefunction) ** 2
        im_wave.set_data(probability_density)
        im_wave.set_clim(np.max(probability_density), np.min(probability_density))
        return im_wave

    ani = animation.FuncAnimation(fig, animate, frames=40, repeat=True, interval=100)

    plt.show()


def benchmark(n):
    start_time = time.time()
    wavefunction = propagate(wavepacket, 0, n)
    print("Time taken to complete {} steps is {} seconds".format(n, time.time() - start_time))


calcualte_and_plot()
# ani.save('./animation.gif', writer='imagemagick', fps=10)
