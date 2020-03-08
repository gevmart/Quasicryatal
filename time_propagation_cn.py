from scipy.linalg import inv, solve_banded

from config import *
from grid import generate_potential
from utils import calc_center_of_mass, calc_root_mean_square, default_notify


#%%
# define the elements which will go into matrices
future_coupling = -1j * time_step / 4 / M
present_coupling = -future_coupling


def make_cn_matrix(diag, off_diag):
    """
    Creates a tridiagonal, symmetric matrix using scipy convention of banded matrices.
    For the details on the convention,
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html

    :param diag: the element to be put on the diagonal
    :param off_diag: the element to be put off diagonal
    :return: tridiagonal matrix in scipy banded matrix convention
    """

    sup = np.repeat(off_diag, GRID_SIZE ** 2)  # above the main diagonal
    sub = np.repeat(off_diag, GRID_SIZE ** 2)  # below the main diagonal
    # very important as not doing this results in making not neighbours become neighbours
    # which results in loss of unitarity near the edges:q
    sup[::GRID_SIZE] = 0
    sub[GRID_SIZE-1::GRID_SIZE] = 0
    return np.array([sup, np.repeat(diag, GRID_SIZE ** 2), sub])


def multiply_tridiagonal(matrix, arr):
    """
    Multiplies a tridiagonal matrix given in scipy banded matrix convention and a vector

    :param matrix: a tridiagonal matrix
    :param arr: an array of the same dimension as the matrix
    :return: an array resulting from the multiplication
    """

    return np.roll(arr * matrix[0], -1) + arr * matrix[1] + np.roll(arr * matrix[2], 1)


base_tridiagonal_future = make_cn_matrix(1, future_coupling)
base_tridiagonal_present = make_cn_matrix(1, present_coupling)


def diag_time_dependent(t, v, notify, transpose=False):
    potential = generate_potential(t, v, notify) if not transpose else generate_potential(t, v, notify).T
    return 1j * time_step / 2 / M + 1j * time_step * potential / 4


def apply_half_step(wavefunction, t, v, notify, transpose=False):
    """
    Applies half step propagation in Crank-Nicholson scheme. If transpose is true, it is the second half being applied

    :param wavefunction: the initial wavefunction
    :param t: time
    :param v: relative strength of the potential
    :param notify: function to be called when a notifiable action takes place
    :param transpose: indicates whether it's the first or the second step of the process
    :return: wavefunction after half step propagation
    """

    present_diag = 1 - diag_time_dependent(t, v, notify, transpose)  # present diagonal
    future_diag = 1 + diag_time_dependent(t + time_step / 2, v, notify, transpose)  # future diagonal
    base_tridiagonal_future[1] = np.reshape(future_diag, GRID_SIZE ** 2)
    base_tridiagonal_present[1] = np.reshape(present_diag, GRID_SIZE ** 2)
    wavefunction = solve_banded((1, 1), base_tridiagonal_future,
                                multiply_tridiagonal(base_tridiagonal_present, wavefunction))
    return np.reshape(wavefunction, (GRID_SIZE, GRID_SIZE))


def propagate_cn(wavefunction, t, n, v=float('nan'), notify=default_notify):
    avg_com, avg_rms = np.array([0.0, 0.0]), 0.0
    for i in np.arange(n):
        wavefunction = np.reshape(wavefunction, GRID_SIZE ** 2)
        wavefunction = apply_half_step(wavefunction, t + i * time_step, v, notify)

        wavefunction = np.reshape(wavefunction.T, GRID_SIZE ** 2)
        wavefunction = apply_half_step(wavefunction, t + (i + 1 / 2) * time_step, v, notify, True).T
        avg_com = (avg_com * i + calc_center_of_mass(wavefunction)) / (i + 1)
        avg_rms = (avg_rms * i + calc_root_mean_square(wavefunction)) / (i + 1)

    return wavefunction, avg_com, avg_rms
