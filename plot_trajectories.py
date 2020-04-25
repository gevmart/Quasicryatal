import numpy as np
from matplotlib import pyplot as plt

from config import PLOT_SAVE_DIR_BASE


def read_and_parse_input(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f]

    return lines


lines = read_and_parse_input("{}move_parallelogram_potential_2.5_x_0_y_0_cn_n_70_cutoff_490_grid_700_wavelength_80_timestep_0.2_lasernum_4_repeat_6_retroreflective_True_extended/data.txt"
                             .format(PLOT_SAVE_DIR_BASE))

idx_list = [idx + 1 for idx, val in enumerate(lines) if len(val.split("[")) == 1]
res = [lines[i:j] for i, j in zip([0] + idx_list, idx_list + [None])]
for i in np.arange(len(res) - 1):
    res[i] = res[i][:-1]


def over_wavelength(s):
    return float(s) / 80


colors = ['red', 'blue', 'orange', 'black', 'green', 'purple']
fig, ax = plt.subplots()
ax.tick_params(axis='both', which='major', labelsize=18)
for i in np.arange(len(res)):
    movement = np.array([list(map(over_wavelength, list(filter(None, li.split("[")[1].split("]")[0]
                                                     .split(" "))))) for li in res[i]])

    movement[:, 1] = -movement[:, 1]
    plt.quiver(movement[:-1, 0], movement[:-1, 1], movement[1:, 0] - movement[:-1, 0],
               movement[1:, 1] - movement[:-1, 1],
               scale_units='xy', angles='xy', scale=1, color=colors[i])

plt.title("Square modulations starting at symmetry center, 10 fold", fontsize=18)
plt.legend(["Before modulation", "First cycle", "Second cycle", "After modulation"], fontsize=18)
plt.xlabel(r"$x/\lambda$", fontsize=18)
plt.ylabel(r"$y/\lambda$", fontsize=18)
# ax.set_ylim([-0.2, 0.2])
plt.show()
