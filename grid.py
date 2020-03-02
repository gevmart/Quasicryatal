import numpy as np
import math

from config import GRID_SIZE, WAVELENGTH, v_0, omega, k, v_rec

# %%
x, y = np.meshgrid(
    np.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, num=GRID_SIZE),
    np.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, num=GRID_SIZE))


# %%
def generate_potential(t, v=float('nan')):
    """
    Generates the potential at time t
    :param t: time
    :param v: optionally takes the strength of the potential, or uses the default from config
    :return: the potential grid at time t
    """
    n = 10

    v = v_0 if math.isnan(v) else v * v_rec

    p1, p2, p3, p4 = phase_single_square(t, n)

    return -v / 4 * (
            np.cos(p1) ** 2 +
            np.cos(p2) ** 2 +
            np.cos(p3) ** 2 +
            np.cos(p4) ** 2)


def propagate_sliding(t, n):
    """
    Slides the potential to have a new center keeping the form of the potential
    :param t: time
    :param n: quantifies the rate of change of the potential
    :return: the phases of the lasers
    """
    x_t, y_t = slide_x(t, n)

    return k * x_t, k * y_t, k / 2 ** 0.5 * (x_t + y_t), k / 2 ** 0.5 * (x_t - y_t)


def slide_x(t, n):
    """
    Slides the potential along negative x direction
    :param t: time
    :param n: slide speed measure
    :return: new center position
    """
    return x + t * omega / n / k, y


def propagate_square(t, n):
    """
    Slides the potential in a square (left, up, right, down)
    :param t: time
    :param n: slide speed measure
    :return: new center position
    """
    return square_movement(t, n)


def phase_single_laser(t, n):
    """
    Changes the phase of the laser which is along x direction
    :param t: time
    :param n: potential change measure
    :return: phases of the lasers
    """
    return k * (x + t * omega / n / k), k * y, k / 2 ** 0.5 * (x + y), k / 2 ** 0.5 * (x - y)


def phase_single_square(t, n, cutoff=30, start=0):
    """
    Changes the phases of the lasers along x and y direction in a square fashion
    :param t: time
    :param n: potential change measure
    :param cutoff: optional size of the square
    :param start: start time of the square movement
    :return: phases of the lasers
    """
    p1_rel, p2_rel = square_movement(t, n, cutoff, start)

    return k * p1_rel, k * p2_rel, k / 2 ** 0.5 * (x + y), k / 2 ** 0.5 * (x - y)


def phase_square_and_reverse(t, n, cutoff=30):
    """
    Changes the phases of the lasers along x and y directions in a square fashion, then unwinds the square back.
    It is expected that this action should results in no net movement
    :param t: time
    :param n: potential change measure
    :param cutoff: optional size of the square
    :return: phases of the lasers
    """
    if t * omega < 4 * cutoff:
        return phase_single_square(t, n, cutoff)

    p1, p2, p3, p4 = phase_single_square(t, n, cutoff, 4 * cutoff)
    return p2 - k * y + k * x, p1 - k * x + k * y, p3, p4


def square_movement(t, n, cutoff=10, start=0):
    """
    Makes a square movement in some coordinates
    :param t: time
    :param n: potential change measure
    :param cutoff: indicates thw size of the square
    :param start: time from which to start the square motion of the potential
    :return: final values of moving coordinates
    """
    x_t, y_t = x, y
    t = t - start / omega
    if t * omega < cutoff:
        x_t = x + t * omega / n / k
        y_t = y
    elif t * omega < 2 * cutoff:
        x_t = x + cutoff / n / k
        y_t = y + (t * omega - cutoff) / n / k
    elif t * omega < 3 * cutoff:
        x_t = x + (3 * cutoff - t * omega) / n / k
        y_t = y + cutoff / n / k
    elif t * omega < 4 * cutoff:
        x_t = x
        y_t = y + (4 * cutoff - t * omega) / n / k

    return x_t, y_t


def potential_ten_fold(t, v=float('nan')):
    v = v_0 if math.isnan(v) else v * v_rec

    lasers_number = 5
    angles = np.arange(lasers_number) * np.pi / lasers_number

    kxs, kys = k * np.cos(angles), k * np.sin(angles)

    return -v / lasers_number * np.sum(np.cos(np.outer(x, kxs) + np.outer(y, kys)) ** 2, axis=1).reshape((GRID_SIZE, GRID_SIZE))


# %%
potential = generate_potential(0)
minima = ((potential <= np.roll(potential, 1, 0)) &
          (potential <= np.roll(potential, -1, 0)) &
          (potential <= np.roll(potential, 1, 1)) &
          (potential <= np.roll(potential, -1, 1)) &
          (potential <= np.roll(np.roll(potential, 1, 0), 1, 1)) &
          (potential <= np.roll(np.roll(potential, 1, 0), -1, 1)) &
          (potential <= np.roll(np.roll(potential, -1, 0), 1, 1)) &
          (potential <= np.roll(np.roll(potential, -1, 0), -1, 1))).astype(int)

mins_only = np.argwhere(minima)

minima = ((1 <= np.roll(minima, 1, 0)) |
          (1 <= np.roll(minima, -1, 0)) |
          (1 <= np.roll(minima, 1, 1)) |
          (1 <= np.roll(minima, -1, 1)) |
          (1 <= np.roll(np.roll(minima, 1, 0), 1, 1)) |
          (1 <= np.roll(np.roll(minima, 1, 0), -1, 1)) |
          (1 <= np.roll(np.roll(minima, -1, 0), 1, 1)) |
          (1 <= np.roll(np.roll(minima, -1, 0), -1, 1))).astype(int)
