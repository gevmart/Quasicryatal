import numpy as np
from matplotlib import pyplot as plt

from config import PLOT_SAVE_DIR_BASE


def read_and_parse_input(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f]

    return lines


lines = read_and_parse_input("{}square_single_phases_x_147_y_0_ssf_n=300_relaxed_10_low_timestep/data.txt"
                             .format(PLOT_SAVE_DIR_BASE))

idx_list = [idx + 1 for idx, val in enumerate(lines) if len(val.split("[")) == 1]
res = [lines[i:j] for i, j in zip([0] + idx_list, idx_list + [None])]
for i in np.arange(len(res) - 1):
    res[i] = res[i][:-1]
colors = ['red', 'blue', 'black']
for i in np.arange(len(res)):
    movement = np.array([list(map(float, list(filter(None, li.split("[")[1].split("]")[0].split(" "))))) for li in res[i]])
    plt.quiver(movement[:-1, 0], movement[:-1, 1], movement[1:, 0] - movement[:-1, 0], movement[1:, 1] - movement[:-1, 1],
               scale_units='xy', angles='xy', scale=1, color=colors[i])

plt.show()
