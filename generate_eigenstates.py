import time

import matplotlib.pyplot as plt
import scipy.sparse.linalg
from numpy import pi
from scipy.sparse import csr_matrix as cs

from config import *
from eigenstate_helpers import hamiltonian, one_d_index_to_two_d_index, generate_potential, two_d_to_one_d_index

start = time.perf_counter()

## Physical parameters
hbar_ev = 6.582119570e-16  # [eV s]
wavelength = 1  # 726*1e-9 #Distances are normalized by wavelength
k0 = 2 * pi / wavelength  # wave number
E_recoil = 9.7 * 1e3 * hbar_ev * 2 * pi  # [ev]9.7*1e3*hbar*2*pi
hbar = hbar_ev / E_recoil  # normalized by recoil energy, [E_r s]
mass = 2 * pi ** 2 * hbar ** 2  # mass normalized by E_recoil

# systems parameters
plaquette_length = 1
up = plaquette_length / 2
low = -plaquette_length / 2
left = -plaquette_length / 2
right = plaquette_length / 2
w = 1 / (2 * pi ** 2 * hbar)  # oscillator frequency so that oscillator length is 1
n_x = 80
n_y = n_x
x_array = np.linspace(low, up, n_x)
y_array = np.linspace(left, right, n_y)
dx = x_array[1] - x_array[0]
dy = y_array[1] - y_array[0]
X, Y = np.meshgrid(x_array, y_array)


def coarse_grain(vector, i, j, coarse_graining_size):
    avg = 0
    for ind1 in np.arange(coarse_graining_size):
        for ind2 in np.arange(coarse_graining_size):
            avg += vector[two_d_to_one_d_index(i + ind1, j + ind2, n_x)]

    return avg


def calc_eigenstates(n_states, center_x, center_y):
    V_mat = generate_potential(X, Y, 2 * NUMBER_OF_LASERS, center_x, center_y)
    H, L, V = hamiltonian(x_array, y_array, V_mat)
    val, vec = scipy.sparse.linalg.eigsh(cs(H), k=n_states, return_eigenvectors=True, which='SA')
    sorted_indices = np.argsort(val)
    beautified_eigenstates = np.zeros((n_states, GRID_SIZE, GRID_SIZE), dtype=complex)

    print("center_x" + str(center_y))
    coarse_graining_size = WAVELENGTH // n_x
    for j in range(n_states):
        eigvec_mat = np.zeros((GRID_SIZE, GRID_SIZE), dtype=complex)
        for index in range(n_x * n_y // coarse_graining_size ** 2):
            (i_x, j_y) = one_d_index_to_two_d_index(coarse_graining_size ** 2 * index, n_x)
            eigvec_mat[i_x - n_x // 2 + GRID_SIZE // 2 + center_x, j_y - n_y // 2 + GRID_SIZE // 2 + center_y] = \
                coarse_grain(vec[:, j], i_x, j_y, coarse_graining_size)
        beautified_eigenstates[j] = eigvec_mat

    return beautified_eigenstates


def get_eigenstates(n=10, center_x=0, center_y=0):
    """
    Returns the numerically calculated eigenstates near a potential well by considering half wavelength regions
    in each direction only.
    :param n: number of eigenstates to be computed
    :return: eigenstates as a third rank numpy array
    """
    return calc_eigenstates(n, center_x, center_y)

# plt.figure(1)
# plt.clf()
# plt.pcolor(x, y, np.square(np.abs(get_eigenstates()[0])), cmap='magma')
# plt.colorbar()
# plt.xlabel('x')
# plt.ylabel('y')
# # plt.suptitle('State number: ' + str(j) + 'Energy: ' + str(energies[j]))
# plt.savefig("BIG_FDS_State number_ " + str(123) + ".png")
# plt.clf()