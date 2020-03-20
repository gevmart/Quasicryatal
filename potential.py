import numpy as np
import math

from config import *
from consts import *
from utils import default_notify, x, y, apply, identity


# %%
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

    t -= START / omega
    p1, p2, p3, p4 = default_phases() if t < 0 else make_modulation(t, notify)

    return -v / 4 * (
            np.cos(p1) ** 2 +
            np.cos(p2) ** 2 +
            np.cos(p3) ** 2 +
            np.cos(p4) ** 2)


def make_modulation(t, notify):
    """
    Makes a general modulation motion
    :param t: time
    :param notify: notify function
    :return: phases of the lasers
    """
    fn, duration = path_map[PATH]
    return fn(t - duration * CUTOFF * max(0, min(
        (omega * t) / duration // CUTOFF, REPEATS - 1)) / omega, cutoff=CUTOFF, notify=notify)


def propagate_sliding(t, cutoff=CUTOFF, notify=default_notify):
    """
    Slides the potential to have a new center keeping the form of the potential
    :param t: time
    :param cutoff: the size
    :param notify: function for notifying
    :return: the phases of the lasers
    """
    x_t, y_t = propagate_square(t, cutoff=cutoff, notify=notify)

    return k * x_t, k * y_t, k / 2 ** 0.5 * (x_t + y_t), k / 2 ** 0.5 * (x_t - y_t)


def slide_x(t):
    """
    Slides the potential along negative x direction
    :param t: time
    :return: new center position
    """
    return x + t * omega / POTENTIAL_CHANGE_SPEED / k, y


def propagate_square(t, cutoff=CUTOFF, notify=default_notify):
    """
    Slides the potential in a square (left, up, right, down)
    :param t: time
    :param cutoff: the size of the square
    :param notify: notify function
    :return: new center position
    """
    p1, p2, p3, p4 = square_movement(t, cutoff=cutoff, notify=notify)
    return p1 / k, p2 / k


def phase_single_laser(t):
    """
    Changes the phase of the laser which is along x direction
    :param t: time
    :return: phases of the lasers
    """
    return k * (x + t * omega / POTENTIAL_CHANGE_SPEED / k), k * y, k / 2 ** 0.5 * (x + y), k / 2 ** 0.5 * (x - y)


def phase_single_square(t, cutoff=30, notify=default_notify, notify_bool=True):
    """
    Changes the phases of the lasers along x and y direction in a square fashion
    :param t: time
    :param cutoff: optional size of the square
    :param notify: an optional method to be called when there is a need to notify
    :param notify_bool: whether to notify
    :return: phases of the lasers
    """
    return square_movement(t, cutoff, notify=notify, notify_bool=notify_bool)


def phase_square_and_reverse(t, cutoff=30, notify=default_notify):
    """
    Changes the phases of the lasers along x and y directions in a square fashion, then unwinds the square back.
    It is expected that this action should results in no net movement
    :param t: time
    :param cutoff: optional size of the square
    :param notify: notify function
    :return: phases of the lasers
    """
    if t * omega < 4 * cutoff:
        return phase_single_square(t, cutoff)

    p1, p2, p3, p4 = phase_single_square(t - 4 * cutoff / omega, cutoff, notify=notify)
    return p2 - k * y + k * x, p1 - k * x + k * y, p3, p4


def triangle_xy(t, cutoff=30, notify=default_notify):
    """
    Changes the phases of x and y directed lasers in an equilateral triangle
       #########
       -#-----#-
       --#---#--
       ---#-#---
       ----#---
    :param t: time
    :param cutoff: size of the triangle
    :param notify: a function to notify when the cycle starts and ends
    :return: phases of the lasers
    """
    triangle_list_fns = [
        (lambda x: x - t * omega / POTENTIAL_CHANGE_SPEED / 2,
         lambda x: x + t * omega / POTENTIAL_CHANGE_SPEED * 3 ** 0.5 / 2, identity, identity),
        (lambda x: x + (2 * t * omega - 3 * cutoff) / POTENTIAL_CHANGE_SPEED / 2,
         lambda x: x + cutoff / POTENTIAL_CHANGE_SPEED * 3 ** 0.5 / 2, identity, identity),
        (lambda x: x - (t * omega - 3 * cutoff) / POTENTIAL_CHANGE_SPEED / 2,
         lambda x: x - (t * omega - 3 * cutoff) / POTENTIAL_CHANGE_SPEED * 3 ** 0.5 / 2, identity, identity)]

    return closed_loop_steps(t, len(triangle_list_fns), triangle_list_fns, cutoff, notify)


def parallelogram_xy(t, cutoff=30, notify=default_notify):
    """
    Changes the phases of x and y directed lasers in a counterclockwise parallelogram with pi/3 angle
       ----#---
       ---#-#---
       --#---#--
       -#-----#-
       #-------#
       -#-----#-
       --#---#--
       ---#-#---
       ----#---
    :param t: time
    :param cutoff: size of the triangle
    :param notify: a function to notify when the cycle starts and ends
    :return: phases of the lasers
    """
    parallelogram_fns_list = [
        (lambda x: x - t * omega / POTENTIAL_CHANGE_SPEED / 2,
         lambda x: x + t * omega / POTENTIAL_CHANGE_SPEED * 3 ** 0.5 / 2, identity, identity),
        (lambda x: x + (t * omega - 2 * cutoff) / POTENTIAL_CHANGE_SPEED / 2,
         lambda x: x + t * omega / POTENTIAL_CHANGE_SPEED * 3 ** 0.5 / 2, identity, identity),
        (lambda x: x + (t * omega - 2 * cutoff) / POTENTIAL_CHANGE_SPEED / 2,
         lambda x: x + (4 * cutoff - t * omega) / POTENTIAL_CHANGE_SPEED * 3 ** 0.5 / 2, identity, identity),
        (lambda x: x + (4 * cutoff - t * omega) / POTENTIAL_CHANGE_SPEED / 2,
         lambda x: x + (4 * cutoff - t * omega) / POTENTIAL_CHANGE_SPEED * 3 ** 0.5 / 2, identity, identity)]

    return closed_loop_steps(t, len(parallelogram_fns_list), parallelogram_fns_list, cutoff, notify)


def semicircle_xy(t, cutoff=30, notify=default_notify):
    """
    Changes the phases of x and y directed lasers in a counterclockwise semicircle
       --#####-
       -#-----#
       #-------#
       #########
    :param t: time
    :param cutoff: size of the semicircle
    :param notify: a function to notify when the cycle starts and ends
    :return: phases of the lasers
    """
    radius = cutoff / POTENTIAL_CHANGE_SPEED
    semicircle_list_fns = [
        (lambda x: x - t * omega / POTENTIAL_CHANGE_SPEED, identity, identity, identity),
        (lambda x: x - radius * np.cos((t * omega - cutoff) / cutoff),
         lambda x: x + radius * np.sin((t * omega - cutoff) / cutoff), identity, identity),
        (lambda x: x - (t * omega - (2 + np.pi) * cutoff) / POTENTIAL_CHANGE_SPEED, identity, identity, identity)]

    return closed_loop_steps(t, len(semicircle_list_fns), semicircle_list_fns, cutoff, notify, intervals=[1, np.pi, 1])


def circle_xy(t, cutoff=30, notify=default_notify):
    """
    Changes the phases of x and y directed lasers in a counterclockwise circle starting, but not centered at the origin
       --#####-
       -#-----#
       #-------#
       #-------#
       #-------#
       -#-----#
       --#####-
    :param t: time
    :param cutoff: size of the circle
    :param notify: a function to notify when the cycle starts and ends
    :return: phases of the lasers
    """
    notify_started(t, notify)
    radius = cutoff / POTENTIAL_CHANGE_SPEED
    if 0 < t * omega < 2 * np.pi * cutoff:
        return circle(t, radius, 0, 1, cutoff)

    notify_finished(t, notify)

    return default_phases()


def down_and_circle_xy(t, cutoff=30, notify=default_notify):
    """
    Changes the phases of x and y directed lasers in a counterclockwise circle centered at origin by
     first going down, then making the circle
       --#####-
       -#-----#
       #-------#
       #---#---#
       #---#---#
       -#--#--#
       --#####-
    :param t: time
    :param cutoff: size of the circle
    :param notify: a function to notify when the cycle starts and ends
    :return: phases of the lasers
    """
    radius = cutoff / POTENTIAL_CHANGE_SPEED
    down_circle_fns_list = [
        (identity, lambda x: x - omega * t / POTENTIAL_CHANGE_SPEED, identity, identity),
        (lambda x: x + radius * (-np.sin((t * omega - cutoff) / cutoff)),
         lambda x: x + radius * (-np.cos((t * omega - cutoff) / cutoff)), identity, identity),
        (identity, lambda x: x + (omega * t - (2 + 2 * np.pi) * cutoff) / POTENTIAL_CHANGE_SPEED, identity, identity)]

    return closed_loop_steps(t, len(down_circle_fns_list), down_circle_fns_list, cutoff, notify,
                             intervals=[1, 2 * np.pi, 1])


def circle(t, radius, center_x=0, center_y=0, cutoff=30):
    """
    Makes a circular motion assuming that we start at 3/2 pi angle
    :param t: time
    :param radius: circle radius
    :param center_x: center of circle in x
    :param center_y: center of circle in y
    :param cutoff: modulation size measure
    :return: phases of the lasers
    """
    p1, p2, p3, p4 = default_phases()
    return p1 + (center_x - np.sin(t * omega / cutoff)) * radius,\
           p2 + (center_y - np.cos(t * omega / cutoff)) * radius, p3, p4


def square_movement(t, cutoff=CUTOFF, notify=default_notify, notify_bool=True):
    """
    Makes a square movement in some coordinates
    :param t: time
    :param cutoff: indicates thw size of the square
    :param notify: an optional method to be called when there is a need to notify
    :param notify_bool: whether to notify
    :return: final values of moving coordinates
    """
    if t < 0:
        return default_phases()

    square_list_fns = [(lambda x: x + t * omega / POTENTIAL_CHANGE_SPEED, identity, identity, identity),
                       (lambda x: x + cutoff / POTENTIAL_CHANGE_SPEED,
                        lambda x: x + (t * omega - cutoff) / POTENTIAL_CHANGE_SPEED, identity, identity),
                       (lambda x: x + (3 * cutoff - t * omega) / POTENTIAL_CHANGE_SPEED,
                        lambda x: x + cutoff / POTENTIAL_CHANGE_SPEED, identity, identity),
                       (identity, lambda x: x + (4 * cutoff - t * omega) / POTENTIAL_CHANGE_SPEED, identity, identity)]

    return closed_loop_steps(t, 4, square_list_fns, cutoff, notify, notify_bool=notify_bool)


def closed_loop_steps(t, steps, fns_list, cutoff, notify, intervals=None, notify_bool=True):
    """
    Returns the phases of a general closed loop of a shape given by fns_list
    :param t: time
    :param steps: number of segments in the loop
    :param fns_list: a list of 4-tuples specifying the progression of a phase of a laser in a segment
    :param cutoff: size of the loop
    :param notify: function to be called when modulation starts/stops
    :param intervals: optionally, provide the length of each interval in cutoff units
    :param notify_bool: whether to notify
    :return: phases of the lasers
    """
    global started, finished
    if t < 0:
        return default_phases()
    if notify_bool:
        notify_started(t, notify)

    intervals = intervals if intervals else list(np.repeat(1, steps))
    prev_sum = 0

    for i in np.arange(steps):
        if t * omega < (prev_sum + intervals[i]) * cutoff:
            return tuple(map(apply, list(zip(fns_list[i], default_phases()))))
        prev_sum += intervals[i]

    if notify_bool:
        notify_finished(t, notify)

    return default_phases()


def default_phases():
    """
    :return: the default phases of laser if there was no modulation
    """
    return k * x, k * y, k / 2 ** 0.5 * (x + y), k / 2 ** 0.5 * (x - y)


def notify_started(t, notify):
    global started
    if t > 0 and not started:
        notify(START_MESSAGE)
        started = True


def notify_finished(t, notify):
    global finished
    if not finished:
        notify(FINISH_MESSAGE)
        finished = True


def potential_ten_fold(t, v=float('nan')):
    v = v_0 if math.isnan(v) else v * v_rec

    lasers_number = 5
    angles = np.arange(lasers_number) * np.pi / lasers_number

    kxs, kys = k * np.cos(angles), k * np.sin(angles)

    return -v / lasers_number * np.sum(np.cos(np.outer(x, kxs) + np.outer(y, kys)) ** 2, axis=1).reshape((GRID_SIZE, GRID_SIZE))


# a map from path keyword to the function and duration of the cutoff
path_map = {
    MOVE_SQUARE: (propagate_sliding, 4),
    SQUARE: (phase_single_square, 4),
    TRIANGLE: (triangle_xy, 3),
    PARALLELOGRAM: (parallelogram_xy, 4),
    SEMICIRCLE: (semicircle_xy, 2 + np.pi),
    CIRCLE: (circle_xy, 2 * np.pi),
    DOWN_CIRCLE: (down_and_circle_xy, 2 + 2 * np.pi)
}


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
