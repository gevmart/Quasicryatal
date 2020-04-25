import numpy as np
from scipy.special import factorial
from config import *
from utils import x, y, second_derivative
from potential import potential


def generate_hermite(n, m, x_0, y_0):
    omega_x = (second_derivative(potential, x_0, y_0, False) / 2 / M) ** 0.5
    omega_y = (second_derivative(potential, x_0, y_0, True) / 2 / M) ** 0.5
    hermite_x = generic_hermite(n, x - x_0, omega_x)
    hermite_y = generic_hermite(m, y - y_0, omega_y)
    return hermite_x * hermite_y


def generic_hermite(k, z, omega):
    sigma = 1 / (M * omega) ** 0.5
    return 1 / (2 ** k * factorial(k)) ** 0.5 * (1 / np.pi / sigma ** 2) ** 0.25 * np.exp(-z ** 2 / 2 / sigma ** 2) *\
                hermite_polinomial(k, z / sigma)


def hermite_polinomial(k, z):
    # hardcode Hermites as they have no closed form (could alternatively do recursively)
    if k == 0:
        return 1
    if k == 1:
        return 2 * z
    if k == 2:
        return 4 * z ** 2 - 2
    if k == 3:
        return 8 * z ** 3 - 12 * z
    if k == 4:
        return 16 * z ** 4 - 48 * z ** 2 + 12
    if k == 5:
        return 32 * z ** 5 - 160 * z ** 3 + 120 * z
    if k == 6:
        return 64 * z ** 6 - 480 * z ** 4 + 720 * z ** 2 - 120

    # only hardcoded up to 6, should never get here!
    print("Shouldn't have come here, use Hermites only up to 6!")
    return 1
