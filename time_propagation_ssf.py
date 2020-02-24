from config import *
from grid import x, y, generate_potential


def propagate_ssf(wavefunction, t, n):
    for i in np.arange(n):
        wavefunction *= np.exp(-1j * generate_potential(t + i * time_step) * time_step / 2)
        wavefunction = np.fft.fftshift(np.fft.fft2(wavefunction, norm=NORM))
        wavefunction *= np.exp(-1j * time_step * k_step ** 2 * (x ** 2 + y ** 2) / (2 * M))
        wavefunction = np.fft.ifft2(np.fft.fftshift(wavefunction), norm=NORM)
        wavefunction *= np.exp(-1j * generate_potential(t + i * time_step + time_step / 2) * time_step / 2)

    return wavefunction
