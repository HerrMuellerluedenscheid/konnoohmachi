import konnoohmachi
import numpy as np
import pytest


def test_random_data():
    frequencies = np.linspace(0, 10, 10)
    amplitudes = np.random.rand(10)
    bandwidth = 40
    smoothed_spectrum = konnoohmachi.smooth(frequencies, amplitudes, bandwidth)

    assert smoothed_spectrum.shape == frequencies.shape


def test_no_zero_frequency():
    frequencies = np.linspace(1, 10, 10)
    amplitudes = np.random.rand(10)
    bandwidth = 40
    smoothed_spectrum = konnoohmachi.smooth(frequencies, amplitudes, bandwidth)

    assert smoothed_spectrum.shape == frequencies.shape


def test_smoother_matches_smooth():
    frequencies = np.linspace(1, 100, 128)
    amplitudes = np.random.rand(128)
    bandwidth = 40

    smoother = konnoohmachi.Smoother(frequencies, bandwidth)

    np.testing.assert_allclose(
        smoother.smooth(amplitudes),
        konnoohmachi.smooth(frequencies, amplitudes, bandwidth),
        rtol=1e-12,
    )


def test_smoother_is_reusable():
    frequencies = np.linspace(1, 100, 64)
    bandwidth = 40
    smoother = konnoohmachi.Smoother(frequencies, bandwidth)

    for _ in range(3):
        amplitudes = np.random.rand(64)
        np.testing.assert_allclose(
            smoother.smooth(amplitudes),
            konnoohmachi.smooth(frequencies, amplitudes, bandwidth),
            rtol=1e-12,
        )


def test_smoother_reports_size():
    smoother = konnoohmachi.Smoother(np.linspace(1, 100, 100), 40)

    assert len(smoother) == 100
    # The windows are cached as a dense n x n matrix of float64.
    assert smoother.nbytes == 100 * 100 * 8


def test_smoother_rejects_bad_bandwidth():
    with pytest.raises(ValueError):
        konnoohmachi.Smoother(np.linspace(1, 100, 10), 0.0)


def test_smoother_rejects_length_mismatch():
    smoother = konnoohmachi.Smoother(np.linspace(1, 100, 10), 40)

    with pytest.raises(ValueError):
        smoother.smooth(np.random.rand(11))


@pytest.mark.benchmark
def test_benchmark_cached():
    """Compares repeated smoothing on a shared frequency axis with and without caching."""
    import time

    bandwidth = 40
    nspectra = 20

    print(f"\nnsamples | uncached \t| cached \t| Gain")
    print("-" * 60)

    for nsamples_exp in range(8, 12):
        nsamples = 2**nsamples_exp
        freqs = np.linspace(1, 100, nsamples, dtype=np.float64)
        spectra = [np.random.rand(nsamples) for _ in range(nspectra)]

        t1 = time.perf_counter()
        for spec in spectra:
            konnoohmachi.smooth(freqs, spec, bandwidth)
        t2 = time.perf_counter()

        smoother = konnoohmachi.Smoother(freqs, bandwidth)
        for spec in spectra:
            smoother.smooth(spec)
        t3 = time.perf_counter()

        uncached, cached = t2 - t1, t3 - t2
        print(
            f"{nsamples}\t | {uncached:10.5f}\t| {cached:10.5f}\t| {(uncached/cached):10.5f} "
        )


@pytest.mark.benchmark
def test_benchmark():
    import time
    from numpy.fft import rfft, rfftfreq
    from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing

    """Benchmqarks Konnoohmachi rust implementation.
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

    plt.figure(figsize=(8, 4))
    ax = plt.subplot(111)
    ax.loglog(freqs, np.abs(spec), label="raw", color="lightgrey")
    ax.loglog(
        freqs,
        konnoohmachi.smooth(freqs, np.abs(spec), 40),
        label="konno-ohmachi (rust)",
    )

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.legend()
    plt.savefig("tests/konno-ohmachi-demo.png")
    plt.show()
