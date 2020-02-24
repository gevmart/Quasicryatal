import numpy as np

from config import GRID_SIZE, WAVELENGTH, v_0, omega, k

# %%
x, y = np.meshgrid(
    np.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, num=GRID_SIZE),
    np.linspace(-GRID_SIZE / 2, GRID_SIZE / 2, num=GRID_SIZE))


# %%
def generate_potential(t):
    n = 10

    return phase_single_laser(t, n)


def propagate_sliding(t, n):
    x_t, y_t = propagate_square(t, n)

    return -v_0 / 4 * (
            np.cos(k * x_t) ** 2 +
            np.cos(k * y_t) ** 2 +
            np.cos(k / 2 ** 0.5 * (x_t + y_t)) ** 2 +
            np.cos(k / 2 ** 0.5 * (x_t - y_t)) ** 2)


def slide_x(t, n):
    return x + t * omega / n / k, y


def propagate_square(t, n):
    cutoff = n
    x_t, y_t = x, y
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


def phase_single_laser(t, n):
    return -v_0 / 4 * (
            np.cos(k * (x + t * omega / n / k)) ** 2 +
            np.cos(k * y) ** 2 +
            np.cos(k / 2 ** 0.5 * (x + y)) ** 2 +
            np.cos(k / 2 ** 0.5 * (x - y)) ** 2)


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
