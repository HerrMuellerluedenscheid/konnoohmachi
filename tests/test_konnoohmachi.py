import konnoohmachi
import numpy as np
import time
import pytest

from reference import konnoohmachi as konnoohmachi_pyrocko
from reference import window as window_pyrocko


def test_konnoohmachi_version():
    assert isinstance(konnoohmachi.__version__(), str)


def test_konnoohmachi():
    b = 1
    n = 1000
    freqs = np.arange(n)
    amps = np.random.rand(n)
    smoothed = konnoohmachi.smooth(freqs, amps, b)
    print(type(smoothed))


def test_window():
    fc = 1.0
    n_values = 3
    b = 20
    frequencies = np.arange(n_values, dtype=float)

    w_pyrocko = window_pyrocko(frequencies, fc=fc, b=b)
    w_konnoohmachi = konnoohmachi.window(frequencies, fc, b)

    assert w_pyrocko[1] == w_konnoohmachi[1]


def test_konnoohmachi_benchmark():
    n_values = 20000
    b = 20
    frequencies = np.arange(n_values, dtype=float)
    spectra = np.random.rand(n_values)

    t1 = time.time()
    smoothed_pyrocko = konnoohmachi_pyrocko(spectra, frequencies, b)
    print(time.time() - t1)

    t1 = time.time()
    smoothed = konnoohmachi.smooth(frequencies, spectra, b)
    print(time.time() - t1)

    for s1, s2 in zip(smoothed, smoothed_pyrocko):
        assert s2 == pytest.approx(s1, 1e-6)

    assert len(smoothed) == len(smoothed_pyrocko)
