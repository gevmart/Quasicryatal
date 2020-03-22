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
    phases = default_phases() if t < 0 else make_modulation(t, notify)

    return -v / 4 * np.sum(np.cos(phases) ** 2, axis=0)


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

    return default_phases(x_t, y_t)


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
    l1, l2 = 0, 1
    ps = np.array(square_movement(t, cutoff=cutoff, notify=notify, laser_one=l1, laser_two=l2))
    defaults = default_phases()
    return (ps[l1, :] - defaults[l1, :]) / k + x, (ps[l2, :] - defaults[l2, :]) / k + y


def phase_single_laser(t):
    """
    Changes the phase of the laser which is along x direction
    :param t: time
    :return: phases of the lasers
    """
    return k * (x + t * omega / POTENTIAL_CHANGE_SPEED / k), k * y, k / 2 ** 0.5 * (x + y), k / 2 ** 0.5 * (x - y)


def phase_single_square(t, cutoff=30, notify=default_notify, notify_bool=True, laser_one=0, laser_two=2):
    """
    Changes the phases of the lasers along x and y direction in a square fashion
    :param t: time
    :param cutoff: optional size of the square
    :param notify: an optional method to be called when there is a need to notify
    :param notify_bool: whether to notify
    :param laser_one: number of the first laser in the plane of movement in parameter space
    :param laser_two: number of the second in the plane of movement in parameter space
    :return: phases of the lasers
    """
    return square_movement(t, cutoff, notify=notify, notify_bool=notify_bool, laser_one=laser_one, laser_two=laser_two)


def phase_square_and_reverse(t, cutoff=30, notify=default_notify, laser_one=0, laser_two=2):
    """
    Changes the phases of the lasers along x and y directions in a square fashion, then unwinds the square back.
    It is expected that this action should results in no net movement
    :param t: time
    :param cutoff: optional size of the square
    :param notify: notify function
    :param laser_one: number of the first laser in the plane of movement in parameter space
    :param laser_two: number of the second in the plane of movement in parameter space
    :return: phases of the lasers
    """
    if t * omega < 4 * cutoff:
        return phase_single_square(t, cutoff)

    p1, p2, p3, p4 = phase_single_square(t - 4 * cutoff / omega, cutoff, notify=notify,
                                         laser_one=laser_one, laser_two=laser_two)
    return p2 - k * y + k * x, p1 - k * x + k * y, p3, p4


def triangle(t, cutoff=30, notify=default_notify, laser_one=0, laser_two=2):
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
    :param laser_one: number of the first laser in the plane of movement in parameter space
    :param laser_two: number of the second in the plane of movement in parameter space
    :return: phases of the lasers
    """
    triangle_list_fns = np.repeat(identity, NUMBER_OF_LASERS * 3).reshape(3, NUMBER_OF_LASERS)
    triangle_list_fns[0, laser_one] = lambda x: x - t * omega / POTENTIAL_CHANGE_SPEED / 2
    triangle_list_fns[0, laser_two] = lambda x: x + t * omega / POTENTIAL_CHANGE_SPEED * 3 ** 0.5 / 2
    triangle_list_fns[1, laser_one] = lambda x: x + (2 * t * omega - 3 * cutoff) / POTENTIAL_CHANGE_SPEED / 2
    triangle_list_fns[1, laser_two] = lambda x: x + cutoff / POTENTIAL_CHANGE_SPEED * 3 ** 0.5 / 2
    triangle_list_fns[2, laser_one] = lambda x: x - (t * omega - 3 * cutoff) / POTENTIAL_CHANGE_SPEED / 2
    triangle_list_fns[2, laser_two] = lambda x: x - (t * omega - 3 * cutoff) / POTENTIAL_CHANGE_SPEED * 3 ** 0.5 / 2

    return closed_loop_steps(t, len(triangle_list_fns), triangle_list_fns, cutoff, notify)


def parallelogram(t, cutoff=30, notify=default_notify, laser_one=0, laser_two=2):
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
    :param laser_one: number of the first laser in the plane of movement in parameter space
    :param laser_two: number of the second in the plane of movement in parameter space
    :return: phases of the lasers
    """
    parallelogram_fns_list = np.repeat(identity, NUMBER_OF_LASERS * 4).reshape(4, NUMBER_OF_LASERS)
    parallelogram_fns_list[0, laser_one] = lambda x: x - t * omega / POTENTIAL_CHANGE_SPEED / 2
    parallelogram_fns_list[0, laser_two] = lambda x: x + t * omega / POTENTIAL_CHANGE_SPEED * 3 ** 0.5 / 2
    parallelogram_fns_list[1, laser_one] = lambda x: x + (t * omega - 2 * cutoff) / POTENTIAL_CHANGE_SPEED / 2
    parallelogram_fns_list[1, laser_two] = lambda x: x + t * omega / POTENTIAL_CHANGE_SPEED * 3 ** 0.5 / 2
    parallelogram_fns_list[2, laser_one] = lambda x: x + (t * omega - 2 * cutoff) / POTENTIAL_CHANGE_SPEED / 2
    parallelogram_fns_list[2, laser_two] = lambda x: x + (4 * cutoff - t * omega) / POTENTIAL_CHANGE_SPEED * 3 ** 0.5 / 2
    parallelogram_fns_list[3, laser_one] = lambda x: x + (4 * cutoff - t * omega) / POTENTIAL_CHANGE_SPEED / 2
    parallelogram_fns_list[3, laser_two] = lambda x: x + (4 * cutoff - t * omega) / POTENTIAL_CHANGE_SPEED * 3 ** 0.5 / 2

    return closed_loop_steps(t, len(parallelogram_fns_list), parallelogram_fns_list, cutoff, notify)


def semicircle(t, cutoff=30, notify=default_notify, laser_one=0, laser_two=2):
    """
    Changes the phases of x and y directed lasers in a counterclockwise semicircle
       --#####-
       -#-----#
       #-------#
       #########
    :param t: time
    :param cutoff: size of the semicircle
    :param notify: a function to notify when the cycle starts and ends
    :param laser_one: number of the first laser in the plane of movement in parameter space
    :param laser_two: number of the second in the plane of movement in parameter space
    :return: phases of the lasers
    """
    radius = cutoff / POTENTIAL_CHANGE_SPEED
    semicircle_list_fns = np.repeat(identity, NUMBER_OF_LASERS * 3).reshape(3, NUMBER_OF_LASERS)
    semicircle_list_fns[0, laser_one] = lambda x: x - t * omega / POTENTIAL_CHANGE_SPEED
    semicircle_list_fns[1, laser_one] = lambda x: x - radius * np.cos((t * omega - cutoff) / cutoff)
    semicircle_list_fns[1, laser_two] = lambda x: x + radius * np.sin((t * omega - cutoff) / cutoff)
    semicircle_list_fns[2, laser_one] = lambda x: x - (t * omega - (2 + np.pi) * cutoff) / POTENTIAL_CHANGE_SPEED

    return closed_loop_steps(t, len(semicircle_list_fns), semicircle_list_fns, cutoff, notify, intervals=[1, np.pi, 1])


def circle(t, cutoff=30, notify=default_notify, laser_one=0, laser_two=2):
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
    :param laser_one: number of the first laser in the plane of movement in parameter space
    :param laser_two: number of the second in the plane of movement in parameter space
    :return: phases of the lasers
    """
    notify_started(t, notify)
    radius = cutoff / POTENTIAL_CHANGE_SPEED
    if 0 < t * omega < 2 * np.pi * cutoff:
        return circle_movement(t, radius, laser_one=laser_one, laser_two=laser_two,
                               center_one=0, center_two=1, cutoff=cutoff)

    notify_finished(t, notify)

    return default_phases()


def down_and_circle(t, cutoff=30, notify=default_notify, laser_one=0, laser_two=2):
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
    :param laser_one: number of the first laser in the plane of movement in parameter space
    :param laser_two: number of the second in the plane of movement in parameter space
    :return: phases of the lasers
    """
    radius = cutoff / POTENTIAL_CHANGE_SPEED
    down_circle_fns_list = np.repeat(identity, NUMBER_OF_LASERS * 3).reshape(3, NUMBER_OF_LASERS)
    down_circle_fns_list[0, laser_two] = lambda x: x - omega * t / POTENTIAL_CHANGE_SPEED
    down_circle_fns_list[1, laser_one] = lambda x: x + radius * (-np.sin((t * omega - cutoff) / cutoff))
    down_circle_fns_list[1, laser_two] = lambda x: x + radius * (-np.cos((t * omega - cutoff) / cutoff))
    down_circle_fns_list[2, laser_two] = lambda x: x + (omega * t - (2 + 2 * np.pi) * cutoff) / POTENTIAL_CHANGE_SPEED

    return closed_loop_steps(t, len(down_circle_fns_list), down_circle_fns_list, cutoff, notify,
                             intervals=[1, 2 * np.pi, 1])


def circle_movement(t, radius, laser_one=0, laser_two=2, center_one=0, center_two=0, cutoff=CUTOFF):
    """
    Makes a circular motion assuming that we start at 3/2 pi angle
    :param t: time
    :param radius: circle radius
    :param laser_one: number of the first laser in the plane of movement in parameter space
    :param laser_two: number of the second in the plane of movement in parameter space
    :param center_one: center of circle in laser_one
    :param center_two: center of circle in laser_two
    :param cutoff: modulation size measure
    :return: phases of the lasers
    """
    ps = default_phases()
    ps[laser_one, :] += (center_one - np.sin(t * omega / cutoff)) * radius
    ps[laser_two, :] += (center_two - np.cos(t * omega / cutoff)) * radius
    return ps


def square_movement(t, cutoff=CUTOFF, notify=default_notify, notify_bool=True, laser_one=0, laser_two=2):
    """
    Makes a square movement in some coordinates
    :param cutoff: indicates thw size of the square
    :param notify: an optional method to be called when there is a need to notify
    :param notify_bool: whether to notify
    :return: final values of moving coordinates
    :param t: time
    :param laser_one: number of the first laser in the plane of movement in parameter space
    :param laser_two: number of the second in the plane of movement in parameter space
    """
    if t < 0:
        return default_phases()

    square_list_fns = np.repeat(identity, NUMBER_OF_LASERS * 4).reshape(4, NUMBER_OF_LASERS)
    square_list_fns[0, laser_one] = lambda x: x + t * omega / POTENTIAL_CHANGE_SPEED
    square_list_fns[1, laser_one] = lambda x: x + cutoff / POTENTIAL_CHANGE_SPEED
    square_list_fns[1, laser_two] = lambda x: x + (t * omega - cutoff) / POTENTIAL_CHANGE_SPEED
    square_list_fns[2, laser_one] = lambda x: x + (3 * cutoff - t * omega) / POTENTIAL_CHANGE_SPEED
    square_list_fns[2, laser_two] = lambda x: x + cutoff / POTENTIAL_CHANGE_SPEED
    square_list_fns[3, laser_two] = lambda x: x + (4 * cutoff - t * omega) / POTENTIAL_CHANGE_SPEED

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


def default_phases(x_t=x, y_t=y):
    """
    :return: the default phases of laser if there was no modulation
    """
    angles = np.arange(NUMBER_OF_LASERS) * np.pi / NUMBER_OF_LASERS

    kxs, kys = k * np.cos(angles), k * np.sin(angles)

    return (np.outer(kxs, x_t) + np.outer(kys, y_t)).reshape(NUMBER_OF_LASERS, GRID_SIZE, GRID_SIZE)


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


# a map from path keyword to the function and duration of the cutoff
path_map = {
    MOVE_SQUARE: (propagate_sliding, 4),
    SQUARE: (phase_single_square, 4),
    TRIANGLE: (triangle, 3),
    PARALLELOGRAM: (parallelogram, 4),
    SEMICIRCLE: (semicircle, 2 + np.pi),
    CIRCLE: (circle, 2 * np.pi),
    DOWN_CIRCLE: (down_and_circle, 2 + 2 * np.pi)
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
