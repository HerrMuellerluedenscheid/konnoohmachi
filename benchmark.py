import time
import konnoohmachi
import matplotlib.pyplot as plt

import numpy as np
from numpy.fft import rfft, rfftfreq

from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing

"""Benchmarks Konnoohmachi rust implementation.
We take the obspy implementation as a reference.
"""

nrepeat = 2
bandwidth = 40
times_python = []
times_rust = []
want_plot = False

print(f"nsamples | Rust \t| Python \t| Gain")
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

if want_plot:
    plt.figure(figsize=(12, 6))
    plt.plot(times_python, label="python")
    plt.plot(times_rust, label="rust")
    plt.legend()

    plt.figure(figsize=(12, 6))
    plt.loglog(freqs, np.abs(spec), label="raw", color="grey")
    plt.loglog(freqs, obspy_konno, label="python")
    plt.loglog(freqs, rust_konno, "-", label="rust")
    plt.legend()

    plt.show()
