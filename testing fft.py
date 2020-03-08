import numpy as np
from matplotlib import pyplot as plt
from generate_wavepacket import wavepacket
from plotting import heatmap
from potential import x, y
from config import GRID_SIZE, k, WAVELENGTH, k_step

NORM = 'ortho'

print(np.sum(np.absolute(wavepacket) ** 2))
fourier = np.fft.fftshift(np.fft.fft2(wavepacket, norm=NORM))
print(np.sum(np.absolute(fourier) ** 2))
inversed = np.fft.ifft2(fourier, norm=NORM).real
print(np.sum(np.absolute(inversed) ** 2))

k_step = 2 * np.pi / GRID_SIZE

fig, ax = plt.subplots()
heatmap(inversed, x * k_step, y * k_step, ax=ax, cbarlabel="s")

plt.show()
