import os
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from config import PLOT_SAVE_DIR_BASE


p_orwell = Path("{}move_triangle_potential_2.0_x_878_y_1160_cn_n_100_cutoff_300_grid_700_wavelength_80_"
         "timestep_0.2_lasernum_4_repeat_1_retroreflective_True/orwell".format(PLOT_SAVE_DIR_BASE))
p_otwell = Path("{}move_triangle_potential_2.0_x_878_y_1160_cn_n_100_cutoff_300_grid_700_wavelength_80_"
         "timestep_0.2_lasernum_4_repeat_1_retroreflective_True/otwell".format(PLOT_SAVE_DIR_BASE))


def load(path, num_of_hermites):
    with path.open('rb') as f:
        fsz = os.fstat(f.fileno()).st_size
        out = np.load(f)
        while f.tell() < fsz:
            out = np.vstack((out, np.load(f)))
    return out.reshape(out.shape[0] // num_of_hermites, num_of_hermites, num_of_hermites)

num_of_hermites = 7
orwell = load(p_orwell, num_of_hermites)
otwell = load(p_otwell, num_of_hermites)
orwell_first = orwell[:, 0, 0]
otwell_first = otwell[:, 0, 0]
print(otwell_first)

plt.plot(np.arange(orwell_first.size), orwell_first)
plt.plot(np.arange(orwell_first.size), otwell_first)
plt.show()