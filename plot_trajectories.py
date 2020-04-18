import numpy as np
from matplotlib import pyplot as plt

from config import PLOT_SAVE_DIR_BASE


def read_and_parse_input(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f]

    return lines


lines = read_and_parse_input("{}move_triangle_potential_2.0_x_762_y_1160_ssf_n_100_cutoff_800_grid_800_wavelength_80_timestep_0.2_lasernum_4_repeat_2/data.txt"
                             .format(PLOT_SAVE_DIR_BASE))

idx_list = [idx + 1 for idx, val in enumerate(lines) if len(val.split("[")) == 1]
res = [lines[i:j] for i, j in zip([0] + idx_list, idx_list + [None])]
for i in np.arange(len(res) - 1):
    res[i] = res[i][:-1]

# first_repeat_data = res[1][len(res[1]) // 2]
# mod_end_data = res[1][-1]
# end_data = res[2][-1]
# print(first_repeat_data)
# print(mod_end_data)
# print(end_data)


def over_wavelength(s):
    return float(s) / 80


colors = ['red', 'blue', 'orange', 'black', 'green', 'purple']
for i in np.arange(len(res)):
    movement = np.array([list(map(over_wavelength, list(filter(None, li.split("[")[1].split("]")[0]
                                                     .split(" "))))) for li in res[i]])
    plt.quiver(movement[:-1, 0], movement[:-1, 1], movement[1:, 0] - movement[:-1, 0],
               movement[1:, 1] - movement[:-1, 1],
               scale_units='xy', angles='xy', scale=1, color=colors[i])

plt.title("Triangle movement repeated twice with modulation speed 100")
plt.legend(np.arange(6))
plt.xlabel(r"$x/\lambda$")
plt.ylabel(r"$y/\lambda$")
plt.show()
