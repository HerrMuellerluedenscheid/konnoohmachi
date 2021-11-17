[![PyPI](https://img.shields.io/pypi/v/konnoohmachi.svg)](https://pypi.python.org/pypi)
[![PyPI](https://img.shields.io/pypi/dm/konnoohmachi.svg)](https://pypi.python.org/pypi)

Konno-Ohmachi spectral smoothing
================================

Implemented in rust with a python interface.

## Dependencies

You need the rust compiler installed:

[Rust](https://www.rust-lang.org/tools/install)

## Installation

Installation from pypi:

```bash
pip install konnoohmachi
```

Installation from source:

```bash
pip install .
```

## Usage

This smoothes some random numbers:

```python
import konnoohmachi

b = 10
n = 1000
freqs = np.arange(n)
amplitudes = np.random.rand(n)
smoothed = konnoohmachi.smooth(freqs, amplitudes, b)
```