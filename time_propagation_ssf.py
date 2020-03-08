from config import *
from grid import generate_potential
from utils import x, y, calc_center_of_mass, calc_root_mean_square, default_notify


def propagate_ssf(wavefunction, t, n, v=float('nan'), notify=default_notify):
    avg_com, avg_rms = np.array([0.0, 0.0]), 0.0

    for i in np.arange(n):
        wavefunction *= np.exp(-1j * generate_potential(t + i * time_step, v, notify=notify) * time_step / 2)
        wavefunction = np.fft.fftshift(np.fft.fft2(wavefunction, norm=NORM))
        wavefunction *= np.exp(-1j * time_step * k_step ** 2 * (x ** 2 + y ** 2) / (2 * M))
        wavefunction = np.fft.ifft2(np.fft.fftshift(wavefunction), norm=NORM)
        wavefunction *= np.exp(-1j * generate_potential(t + i * time_step + time_step / 2,
                                                        v, notify=notify) * time_step / 2)
        avg_com = (avg_com * i + calc_center_of_mass(wavefunction)) / (i + 1)
        avg_rms = (avg_rms * i + calc_root_mean_square(wavefunction)) / (i + 1)

    return wavefunction, avg_com, avg_rms
