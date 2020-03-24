import numpy as np

from config import WAVEPACKET_CENTER_X, WAVEPACKET_CENTER_Y, sigma_square, GRID_SIZE, WAVELENGTH
from utils import x, y

r_square = x ** 2 + y ** 2
wavepacket = (1 / ((np.pi * sigma_square) ** 0.5) * np.exp(-(r_square / (2 * sigma_square))))\
    .astype(complex)
