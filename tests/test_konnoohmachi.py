import konnoohmachi
import numpy as np
import pytest


def test_random_data():
    frequencies = np.linspace(0, 10, 10)
    amplitudes = np.random.rand(10)
    bandwidth = 40
    smoothed_spectrum = konnoohmachi.smooth(frequencies, amplitudes, bandwidth)

    assert smoothed_spectrum.shape == frequencies.shape


# def test_no_zero_frequency():
#     frequencies = np.linspace(1, 10, 10)
#     amplitudes = np.random.rand(10)
#     bandwidth = 40
#     smoothed_spectrum = konnoohmachi.smooth(frequencies, amplitudes, bandwidth)

#     assert smoothed_spectrum.shape == frequencies.shape


@pytest.mark.benchmark
def test_benchmark():
    import time
    from numpy.fft import rfft, rfftfreq
    from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing

    """Benchmarks Konnoohmachi rust implementation.
    We take the obspy implementation as a reference.
    """

    nrepeat = 2
    bandwidth = 40
    times_python = []
    times_rust = []

    print(f"\nnsamples | Rust \t| Python \t| Gain")
    print("-" * 60)

    for nsamples_exp in range(8, 16):
        nsamples = 2**nsamples_exp
        freqs = np.linspace(0, 100, nsamples, dtype=np.float64)
        amps = np.random.rand(nsamples).astype(np.float64)

        spec, freqs = rfft(amps), rfftfreq(nsamples, 0.001)
        spec = np.abs(spec)

        t1 = time.time()
        for i in range(nrepeat):
            rust_konno = konnoohmachi.smooth(freqs, spec, bandwidth)
        t2 = time.time()
        for i in range(nrepeat):
            obspy_konno = konno_ohmachi_smoothing(spec, freqs, normalize=True)
        t3 = time.time()
        trust = (t2 - t1) / nrepeat
        tpython = (t3 - t2) / nrepeat
        times_rust.append(t2)
        times_python.append(t3)
        print(
            f"{nsamples}\t | {trust:10.5f}\t| {tpython:10.5f}\t| {(tpython/trust):10.5f} "
        )


@pytest.mark.plot
def test_against_obspy():
    # c.f. https://github.com/obspy/obspy/issues/2215#issuecomment-422748341
    import obspy
    from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing
    import obspy.signal.util
    from numpy.fft import rfft, rfftfreq
    import matplotlib.pyplot as plt

    tr = obspy.read()[0]

    spec, freqs = rfft(tr.data), rfftfreq(tr.stats.npts, tr.stats.delta)

    plt.figure(figsize=(12, 6))
    plt.loglog(freqs, np.abs(spec), label="raw", color="lightgrey")
    plt.loglog(
        freqs,
        konnoohmachi.smooth(freqs, np.abs(spec), 40),
        label="konno ohmachi (rust)",
    )
    plt.legend()
    plt.show()
