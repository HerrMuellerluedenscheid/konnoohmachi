import konnoohmachi
import numpy as np


def test_lengths():
    frequencies = np.linspace(0, 10, 9)
    amplitudes = np.random.rand(10)
    bandwidth = 40
    konnoohmachi.smooth(frequencies, amplitudes, bandwidth)
