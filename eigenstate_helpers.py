# from potential_functions import *
# from generate_lattice import *
# import config
# from config import *
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix as cs
import numpy as np

from config import *


def two_d_to_one_d_index(i_x, j_y, nx):
    # input: (i,j) 2D index to 1D index
    index = i_x + nx * j_y
    return index


def one_d_index_to_two_d_index(index1, nx):
    # input: 1D index input to 2D indexed positions
    j_y = np.floor_divide(index1, nx)
    i_x = np.remainder(index1, nx)
    return i_x, j_y


def laplacian(nx, ny, dx):
    """
    Generates the discretized Laplacian Matrix using 9 point stencil (precondition dx = dy)
    9 point stencil on our two dimensional grid says that laplacian is not zero for item to itself with weight -3,
    item to nearest neighbours with weight 0.5, and item to its diagonal neighbour with weight 0.25
    :param nx: number of points in x direction
    :param ny: number of points in y direction
    :param dx: discretized space size
    :return: a laplacian matrix in lil_matrix format of scipy
    """
    laplace = lil_matrix((nx * ny, nx * ny), dtype=complex)

    for j_y in range(ny):
        for i_x in range(nx):
            index = two_d_to_one_d_index(i_x, j_y, nx)
            neighbours_next = [(i_x + 1, j_y, nx), (i_x - 1, j_y, nx), (i_x, j_y + 1, nx), (i_x, j_y - 1, nx)]
            for ind in neighbours_next:
                if nx > ind[0] > 0 and ny > ind[1] > 0:
                    index2 = two_d_to_one_d_index(ind[0], ind[1], nx)
                    laplace[index, index2] = 0.5 / dx ** 2
            neighbours_diag = [(i_x + 1, j_y + 1, nx), (i_x - 1, j_y - 1, nx), (i_x - 1, j_y + 1, nx), (i_x + 1, j_y - 1, nx)]
            for ind in neighbours_diag:
                if nx > ind[0] > 0 and ny > ind[1] > 0:
                    index2 = two_d_to_one_d_index(ind[0], ind[1], nx)
                    laplace[index, index2] = 0.25 / dx ** 2
            laplace[index, index] = -3 / dx ** 2

    return laplace


def laplace_banded(nx, ny, dx):
    n = nx * ny
    laplace = np.zeros((nx + 2, n))
    laplace[-1, :] = np.repeat(-3 / dx ** 2, n)
    laplace[-2, :] = np.repeat(0.5 / dx ** 2, n)
    laplace[-2, ::nx] = 0
    laplace[1, :] = np.repeat(0.5 / dx ** 2, n)
    laplace[1, nx-1::nx] = 0
    laplace[0, :] = np.repeat(0.25 / dx ** 2, n)
    laplace[0, ::nx] = 0
    laplace[2, :] = np.repeat(0.25 / dx ** 2, n)
    laplace[2, ::nx] = 0

    return laplace


def hamiltonian_banded(x_array, y_array, v_mat):
    nx = len(x_array)
    ny = len(y_array)
    dx = x_array[1] - x_array[0]
    hamilt = -1 / (4 * np.pi ** 2) * laplace_banded(nx, ny, dx)
    hamilt[-1, :] += v_mat.reshape(nx * ny)

    return hamilt


def hamiltonian(x_array, y_array, V_mat):
    """
    Generates the discretized Hamiltonian H = -1/(4pi^2)*p^2+V(x,y) (using the appropriate dimensions).
    Laplacian is obtained by laplace method above and the potential part is purely diagonal.
    :param x_array: x coordinates of discretized points
    :param y_array: y coordinates of discretized points
    :param V_mat: the potential matrix which is just put on the diagonals of the matrix
    :return: the Hamiltonian in lil_matrix format, as well as kinetic (laplace) and potential parts in the same format
    """
    nx = len(x_array)
    ny = len(y_array)
    dx = x_array[1] - x_array[0]
    H = lil_matrix((nx * ny, nx * ny), dtype=complex)
    laplace = laplacian(nx, ny, dx)
    V = lil_matrix((nx * ny, nx * ny), dtype=complex)
    for i_x in range(nx):
        for j_y in range(ny):
            index = two_d_to_one_d_index(i_x, j_y, nx)
            # print(i_x,j_y,index)
            V[index, index] = V_mat[i_x, j_y]
    H.setdiag(V_mat.reshape(nx * ny))

    # H = -config.hbar**2/(2*config.mass)*laplace+V
    H = -1 / (4 * np.pi ** 2) * laplace + V

    return H, laplace, V


def harm_potential(x, y, w, mass):
    # returns harmonic potential
    return 0.5 * mass * w ** 2 * (x ** 2 + y ** 2)


def generate_potential(x, y, v, center_x=0, center_y=0):
    """
    Generates the potential at time t
    :param t: time
    :param v: optionally takes the strength of the potential, or uses the default from config
    :param notify: a function to be called to notify of an event such as square movement started or finished
    :return: the potential grid at time t
    """
    phases = default_phases(x, y, center_x, center_y)

    return -v / NUMBER_OF_LASERS * np.sum(np.cos(phases) ** 2, axis=0) if not NON_RETROREFLECTIVE else\
        -v / NUMBER_OF_LASERS * np.abs(np.sum(np.exp(1j * phases), axis=0)) ** 2


def default_phases(x_t, y_t, center_x, center_y):
    """
    :return: the default phases of laser if there was no modulation
    """
    angles = np.arange(NUMBER_OF_LASERS) * np.pi / NUMBER_OF_LASERS
    if NON_RETROREFLECTIVE:
        angles *= 2

    kxs, kys = 2 * np.pi * np.cos(angles), 2 * np.pi * np.sin(angles)

    return (np.outer(kxs, x_t - (WAVEPACKET_CENTER_X - center_x) / WAVELENGTH)
            + np.outer(kys, y_t - (WAVEPACKET_CENTER_Y - center_y) / WAVELENGTH))\
        .reshape(NUMBER_OF_LASERS, x_t.shape[0], x_t.shape[0])

