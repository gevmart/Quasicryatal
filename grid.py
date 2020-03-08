import numpy as np
import math

from config import *
from utils import default_notify, x, y


# %%
POTENTIAL_CHANGE_SPEED = 8
CUTOFF = 20
started = False
finished = False


def generate_potential(t, v=float('nan'), notify=default_notify):
    """
    Generates the potential at time t
    :param t: time
    :param v: optionally takes the strength of the potential, or uses the default from config
    :param notify: a function to be called to notify of an event such as square movement started or finished
    :return: the potential grid at time t
    """
    v = v_0 if math.isnan(v) else v * v_rec

    p1, p2, p3, p4 = phase_single_square(t, start=20, cutoff=CUTOFF, notify=notify)

    return -v / 4 * (
            np.cos(p1) ** 2 +
            np.cos(p2) ** 2 +
            np.cos(p3) ** 2 +
            np.cos(p4) ** 2)


def propagate_sliding(t):
    """
    Slides the potential to have a new center keeping the form of the potential
    :param t: time
    :param n: quantifies the rate of change of the potential
    :return: the phases of the lasers
    """
    x_t, y_t = slide_x(t)

    return k * x_t, k * y_t, k / 2 ** 0.5 * (x_t + y_t), k / 2 ** 0.5 * (x_t - y_t)


def slide_x(t):
    """
    Slides the potential along negative x direction
    :param t: time
    :param n: slide speed measure
    :return: new center position
    """
    return x + t * omega / POTENTIAL_CHANGE_SPEED / k, y


def propagate_square(t):
    """
    Slides the potential in a square (left, up, right, down)
    :param t: time
    :param n: slide speed measure
    :return: new center position
    """
    return square_movement(t)


def phase_single_laser(t):
    """
    Changes the phase of the laser which is along x direction
    :param t: time
    :param n: potential change measure
    :return: phases of the lasers
    """
    return k * (x + t * omega / POTENTIAL_CHANGE_SPEED / k), k * y, k / 2 ** 0.5 * (x + y), k / 2 ** 0.5 * (x - y)


def phase_single_square(t, cutoff=30, start=0, notify=default_notify):
    """
    Changes the phases of the lasers along x and y direction in a square fashion
    :param t: time
    :param n: potential change measure
    :param cutoff: optional size of the square
    :param start: start time of the square movement
    :param notify: an optional method to be called when there is a need to notify
    :return: phases of the lasers
    """
    p1_rel, p2_rel = square_movement(t, cutoff, start, notify=notify)

    return k * p1_rel, k * p2_rel, k / 2 ** 0.5 * (x + y), k / 2 ** 0.5 * (x - y)


def phase_square_multiple(t, number_of_squares, start=0, cutoff=CUTOFF):
    """
    Changes the phases of the lasers along x and y directions in a square fashion number_of_squares times
    :param t: time
    :param number_of_squares: number of squares to be made in parameter space
    :param start: start time of the square movement
    :param cutoff: size of the square
    :return: phases of the lasers
    """
    return phase_single_square(t, cutoff=CUTOFF, start=4 * CUTOFF * max(
        0, min((omega * t - start) / 4 // CUTOFF, number_of_squares - 1)) + start)


def phase_square_and_reverse(t, cutoff=30):
    """
    Changes the phases of the lasers along x and y directions in a square fashion, then unwinds the square back.
    It is expected that this action should results in no net movement
    :param t: time
    :param n: potential change measure
    :param cutoff: optional size of the square
    :return: phases of the lasers
    """
    if t * omega < 4 * cutoff:
        return phase_single_square(t, cutoff)

    p1, p2, p3, p4 = phase_single_square(t, cutoff, 4 * cutoff)
    return p2 - k * y + k * x, p1 - k * x + k * y, p3, p4


def square_movement(t, cutoff=10, start=0, notify=default_notify):
    """
    Makes a square movement in some coordinates
    :param t: time
    :param n: potential change measure
    :param cutoff: indicates thw size of the square
    :param start: time from which to start the square motion of the potential
    :param notify: an optional method to be called when there is a need to notify
    :return: final values of moving coordinates
    """
    global started, finished
    x_t, y_t = x, y
    t = t - start / omega
    if t < 0:
        x_t = x
        y_t = y
    elif t * omega < cutoff:
        if not started:
            notify("Movement started")
            started = True
        x_t = x + t * omega / POTENTIAL_CHANGE_SPEED / k
        y_t = y
    elif t * omega < 2 * cutoff:
        x_t = x + cutoff / POTENTIAL_CHANGE_SPEED / k
        y_t = y + (t * omega - cutoff) / POTENTIAL_CHANGE_SPEED / k
    elif t * omega < 3 * cutoff:
        x_t = x + (3 * cutoff - t * omega) / POTENTIAL_CHANGE_SPEED / k
        y_t = y + cutoff / POTENTIAL_CHANGE_SPEED / k
    elif t * omega < 4 * cutoff:
        x_t = x
        y_t = y + (4 * cutoff - t * omega) / POTENTIAL_CHANGE_SPEED / k
    elif not finished:
        notify("Movement ended")
        finished = True

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
