import numpy as np

from config import WAVEPACKET_CENTER_X, WAVEPACKET_CENTER_Y, sigma_square, GRID_SIZE, WAVELENGTH
from utils import x, y


# def mapThatShit(dir):
#     special_guy = 82 * 2 ** 0.5
#
#     center_x = dir[0] * special_guy
#     center_y = dir[1] * special_guy
#     r_square = (x - center_x) ** 2 + (y - center_y) ** 2
#
#     return 1 / ((np.pi * sigma_square) ** 0.5) * np.exp(-(r_square / (2 * sigma_square))).astype(complex)
#
# # wavepacket = np.array((GRID_SIZE, GRID_SIZE), dtype=complex)
# dirs = np.array(list(map(mapThatShit, [[1, 0], [0, 1], [-1, 0], [0, -1], [1 / 2 ** 0.5, 1 / 2 ** 0.5],
#                                        [-1 / 2 ** 0.5, 1 / 2 ** 0.5], [1 / 2 ** 0.5, -1 / 2 ** 0.5], [-1 / 2 ** 0.5, -1 / 2 ** 0.5]])))
# # lloxer = np.array((8, GRID_SIZE, GRID_SIZE)).astype(complex)
# # for i in np.arange(8):
# #     center_x = dirs[i][0]
# #     center_y = dirs[i][1]
# #     r_square = (x - center_x) ** 2 + (y - center_y) ** 2
# #
# #     lloxer[i] = np.exp(-(r_square / (2 * sigma_square))).astype(complex)
#
# wavepacket = np.sum(dirs, axis=0) / np.sqrt(8)
# print(np.sum(np.abs(wavepacket) ** 2))

r_square = x ** 2 + y ** 2
wavepacket = (1 / ((np.pi * sigma_square) ** 0.5) * np.exp(-(r_square / (2 * sigma_square))))\
    .astype(complex)
