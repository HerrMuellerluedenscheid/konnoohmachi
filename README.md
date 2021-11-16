Konno-Ohmachi spectral smoothing
================================

Implemented in rust with a python interface.

## Dependencies

You need the rust compiler installed:

[Rust](https://www.rust-lang.org/tools/install)

## Installation

Installation from pypi:

    pip install konnoohmachi

Installation from source:

    pip install .

## Usage

This smoothes some random numbers:

    import konnoohmachi

    b = 10
    n = 1000
    freqs = np.arange(n)
    amplitudes = np.random.rand(n)
    smoothed = konnoohmachi.smooth(freqs, amplitudes, b)
