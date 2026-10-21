"""
The internal FFT (_fft, _ifft, _rfft, _irfft) against numpy.fft, plus Parseval and impulse properties.

Lengths cover powers of two (radix-2), lengths with small prime factors (mixed radix) and lengths with a large
prime factor (Bluestein). The tolerance per output is 16 eps log2(n) (||x|| + |X_k|): for noise-like signals the
||x|| term dominates, and measured agreement with numpy is within 2 eps ||x|| log2(n). Plant signals often sit on
a large offset, which makes the zero-frequency term n times the offset; the |X_k| term covers the rounding that
both numpy and this FFT make on such a large term.
"""

import cmath
import math

import numpy as np

from harness import EPS, Case, industrial, pick

FN = "_fft"


def complex_input(re, im):
    if len(re) != len(im):
        raise ValueError("real and imaginary parts must have the same length")
    return np.asarray(re, float) + 1j * np.asarray(im, float)


def as_pair(z):
    return [z.real.tolist(), z.imag.tolist()]


def ref_fft(re, im):
    return as_pair(np.fft.fft(complex_input(re, im)))


def ref_ifft(re, im):
    return as_pair(np.fft.ifft(complex_input(re, im)))


def ref_rfft(x):
    return as_pair(np.fft.rfft(np.asarray(x, float)))


def ref_irfft(re, im, n=None):
    return np.fft.irfft(complex_input(re, im), n).tolist()


REFERENCES = {"_fft": ref_fft, "_ifft": ref_ifft, "_rfft": ref_rfft, "_irfft": ref_irfft}
reference = ref_fft


def length(rng, max_len):
    r = rng.random()
    if r < 0.2:
        return 2 ** int(rng.integers(0, 15))
    if r < 0.35:
        return pick(rng, [2, 3, 5, 7, 11, 13, 97, 101, 1009, 4099, 10007, 65537])
    if r < 0.7:
        return int(rng.integers(1, 300))
    return int(round(math.exp(rng.uniform(math.log(300), math.log(max(max_len, 301))))))


def norm(*parts):
    return math.sqrt(sum(v * v for p in parts for v in p))


def tol(scale, n, total=0.0):
    """
    16 eps log2(n) ||x||, plus 8 eps |sum x| in every output. The second term is numpy's: for lengths with a large
    prime factor, numpy's Bluestein FFT smears rounding from a large zero-frequency term into every bin (on a
    92682-point signal sitting on an offset, numpy was 1.1e-13 from the exact transform and this FFT, which takes
    the mean out first, 1.3e-14). The comparison must allow for the less accurate side.
    """
    return 16 * EPS * scale * math.log2(n + 1) + 8 * EPS * total


def rtol(n):
    return 16 * EPS * math.log2(n + 1)


def signal(rng, n):
    return industrial(rng, n) if rng.random() < 0.5 else [float(v) for v in rng.normal(0, 1, n)]


def generate(rng, n, max_len):
    for _ in range(n):
        size = length(rng, max_len)
        which = pick(rng, ["_fft", "_ifft", "_rfft", "_irfft"])
        re = signal(rng, size)
        im = signal(rng, size)
        total = abs(complex(math.fsum(re), math.fsum(im)))
        if which == "_fft":
            yield Case(which, [re, im], {}, tol(norm(re, im), size, total), rtol=rtol(size))
        elif which == "_ifft":
            yield Case(which, [re, im], {}, tol(norm(re, im) / size, size, total / size), rtol=rtol(size))
        elif which == "_rfft":
            yield Case(which, [re], {}, tol(norm(re), size, abs(math.fsum(re))), rtol=rtol(size))
        else:
            spectrum = np.fft.rfft(re)
            out_n = size if rng.random() < 0.7 else int(rng.integers(1, 2 * size + 2))
            kwargs = {"n": out_n} if rng.random() < 0.8 else {}
            if not kwargs:
                out_n = 2 * (len(spectrum) - 1)
            X = [spectrum.real.tolist(), spectrum.imag.tolist()]
            yield Case(which, X, kwargs, tol(norm(*X) / max(out_n, 1), max(out_n, 1)), rtol=rtol(max(out_n, 1)))
    for bad in ([[], []], [[1.0, 2.0], [1.0]]):
        yield Case("_fft", bad, {}, 0.0, "invalid")
    yield Case("_irfft", [[1.0], [0.0]], {}, 0.0, "invalid")


def properties(rng, n, max_len):
    """Parseval's theorem, and the transform of a shifted impulse."""
    for i in range(n):
        size = length(rng, min(max_len, 20000))
        if i % 2 == 0:
            re, im = signal(rng, size), signal(rng, size)
            energy = norm(re, im) ** 2

            def parseval(r, energy=energy, size=size):
                return abs(sum(a * a + b * b for a, b in zip(r[0], r[1])) / size - energy) / max(energy, 1e-300)

            yield Case("_fft", [re, im], {}, 64 * EPS * math.log2(size + 1), "Parseval: energy is preserved",
                       check=parseval)
        else:
            j = int(rng.integers(0, size))
            re = [0.0] * size
            re[j] = 1.0

            def impulse(r, j=j, size=size):
                worst = 0.0
                for k in range(size):
                    e = cmath.exp(-2j * math.pi * ((j * k) % size) / size)
                    worst = max(worst, abs(complex(r[0][k], r[1][k]) - e))
                return worst

            yield Case("_fft", [re, [0.0] * size], {}, 32 * EPS * math.log2(size + 1),
                       "impulse at j transforms to exp(-2 pi i j k / n)", check=impulse)
