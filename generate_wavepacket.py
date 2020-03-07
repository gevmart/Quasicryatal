import numpy as np

from config import WAVEPACKET_CENTER_X, WAVEPACKET_CENTER_Y, sigma_square, GRID_SIZE, WAVELENGTH
from grid import x, y, mins_only

r_square = (x - 0) ** 2 + (y - 0) ** 2
wavepacket = (1 / ((np.pi * sigma_square) ** 0.5) * np.exp(-(r_square / (2 * sigma_square))))\
    .astype(complex)
