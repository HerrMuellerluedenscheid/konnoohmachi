import numpy as np

_wins = {}


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


def konnoohmachi(amps, freqs, b):
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
