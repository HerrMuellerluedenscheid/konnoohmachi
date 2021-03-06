[![PyPI](https://img.shields.io/pypi/v/konnoohmachi.svg)](https://pypi.org/project/konnoohmachi/)
[![PyPI](https://img.shields.io/pypi/dm/konnoohmachi.svg)](https://pypi.org/project/konnoohmachi/)
[![Python](https://github.com/HerrMuellerluedenscheid/konnoohmachi/actions/workflows/python.yml/badge.svg)](https://github.com/HerrMuellerluedenscheid/konnoohmachi/actions/workflows/python.yml)
[![Rust](https://github.com/HerrMuellerluedenscheid/konnoohmachi/actions/workflows/rust.yml/badge.svg)](https://github.com/HerrMuellerluedenscheid/konnoohmachi/actions/workflows/rust.yml)

Fast Konno-Ohmachi Spectral Smoothing
=====================================

Implemented in Rust with a Python interface. The performance gain measured against the widely used Python/numpy implementation that comes with [obspy](https://docs.obspy.org/packages/autogen/obspy.signal.konnoohmachismoothing.konno_ohmachi_smoothing.html#obspy.signal.konnoohmachismoothing.konno_ohmachi_smoothing) approaches approximately a factor of 2.5 for large and 10 for small vectors (see [Benchmarks](#Benchmarks)).

![konno-ohmachi demo](./tests/konno-ohmachi-demo.png)

## Installation

Installation from [pypi](https://pypi.org/project/konnoohmachi/):

```bash
pip install konnoohmachi
```

Installation from source:

```bash
pip install .
```

## Usage

This smoothes some random numbers:

### Python

```python
import konnoohmachi

bandwidth = 40

# using fake random data
frequencies = np.arange(1000)
amplitudes = np.random.rand(1000)

smoothed_amplitudes = konnoohmachi.smooth(frequencies, amplitudes, bandwidth)
```

### Rust

```rust
use konnoohmachi;

let frequencies = Array1::<f64>::zeros(10);
let amplitudes = Array1::<f64>::ones(10);
let bandwidth = 40.0;
konnoohmachi_smooth(
    frequencies.view().into_dyn(),
    amplitudes.view().into_dyn(),
    bandwidth,
);
```

## Benchmarks

Measuring the execution time based of increasing sized spectra yields:

```
❯ python3 benchmark.py
nsamples |    Rust      |    Python     | Performance Gain
----------------------------------------------------------
256      |    0.00017   |    0.00192    |   11.30802
512      |    0.00054   |    0.00431    |    7.97596
1024     |    0.00198   |    0.01117    |    5.63623
2048     |    0.00775   |    0.03143    |    4.05371
4096     |    0.03067   |    0.10024    |    3.26844
8192     |    0.12212   |    0.35058    |    2.87080
16384    |    0.49391   |    1.29653    |    2.62506
32768    |    1.98499   |    5.05335    |    2.54578
```
