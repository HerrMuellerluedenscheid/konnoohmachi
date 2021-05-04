import string_sum
import numpy as np
import pytest


def test_konnoohmachi_version():
    assert isinstance(string_sum.__version__(), str)


def test_konnoohmachi():
    n_values = 3
    b = 1
    smoothed = string_sum.sum_as_string(
        np.arange(n_values), np.ones(n_values), b
    )
    print(smoothed)
    print(type(smoothed))


@pytest.mark.skip
def test_konnoohmachi_against_obspy():
    import obspy
    n_values = 3
    b = 1
    frequencies = num.arange(n_values)
    spectra = num.ones(n_values)
    smoothed = obspy.signal.konnoohmachismoothing.konno_ohmachi_smoothing(
        spectra, frequencies, bandwidth=b)

    print(smoothed)
