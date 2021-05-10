import konnoohmachi
import numpy as np
import time

_wins = {}


def test_konnoohmachi_version():
    assert isinstance(konnoohmachi.__version__(), str)


def test_konnoohmachi():
    b = 1

    n = 1000
    freqs = np.arange(n)
    amps = np.random.rand(n)
    smoothed = konnoohmachi.smooth(freqs, amps, b)
    print(type(smoothed))
    print(smoothed)


def test_konnoohmachi_benchmark():

    n_values = 1000
    b = 20
    frequencies = np.arange(n_values, dtype=float)
    spectra = np.random.rand(n_values)
    t1 = time.time()
    smoothed_pyrocko = konnoohmachi_pyrocko(spectra, frequencies, b)
    print(time.time() - t1)

    frequencies = np.arange(1, n_values + 1, dtype=float)
    t1 = time.time()
    smoothed = konnoohmachi.smooth(frequencies, spectra, b)

    assert smoothed == smoothed_pyrocko
    print(time.time() - t1)

    print(smoothed)


def window(freqs, fc, b):
    if fc == 0.0:
        w = np.zeros(len(freqs))
        w[freqs == 0] = 1.0
        return w

    T = np.log10(freqs / fc) * b
    w = (np.sin(T) / T) ** 4
    w[freqs == fc] = 1.0
    w[freqs == 0.0] = 0.0
    w /= np.sum(w)
    return w


def konnoohmachi_pyrocko(amps, freqs, b):
    smooth = np.zeros(len(freqs), dtype=freqs.dtype)
    amps = np.array(amps)
    global wins
    for i, fc in enumerate(freqs):
        fkey = tuple((b, fc, freqs[0], freqs[1], freqs[-1]))
        if fkey in _wins.keys():
            win = _wins[fkey]
        else:
            win = window(freqs, fc, b)
            _wins[fkey] = win
        smooth[i] = np.sum(win * amps)

    return smooth
