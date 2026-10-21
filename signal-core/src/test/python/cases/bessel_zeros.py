"""
The internal Bessel pole finder (_bessel_zeros) against scipy.signal._filter_design._bessel_zeros.

scipy evaluates kve with the AMOS library; this module evaluates the same closed-form sum exactly in decimal
arithmetic (a double-precision sum has a rounding floor above scipy's 1e-15 stopping test from order 7). Measured
agreement is within 8 eps of each zero's magnitude for orders 0 to 80; the allowance is 64 eps. Orders above about 90, where scipy's own root finding
fails, are not tested here.
"""

import scipy.signal._filter_design as fd

from harness import EPS, Case, Mismatch

FN = "_bessel_zeros"


def compare_zeros(got, want):
    """Each zero within 64 eps of its own magnitude."""
    g = [complex(a, b) for a, b in zip(*got)]
    w = [complex(a, b) for a, b in zip(*want)]
    if len(g) != len(w):
        raise Mismatch("got %d zeros, expected %d" % (len(g), len(w)))
    return max([abs(a - b) / (64 * EPS * abs(b)) for a, b in zip(g, w)] + [0.0])


def reference(N):
    z = fd._bessel_zeros(N)
    return [[v.real for v in z], [v.imag for v in z]]


def generate(rng, n, max_len):
    """Every order from 0 to 80 once, then up to 40 repeats of practical orders. The zeros are deterministic, so
    more calls would only repeat the same work (an order-75 design takes about a second)."""
    for order in range(min(n, 81)):
        yield Case(FN, [order], {}, compare=compare_zeros)
    for _ in range(min(max(n - 81, 0), 40)):
        yield Case(FN, [int(rng.integers(0, 31))], {}, compare=compare_zeros)


def properties(rng, n, max_len):
    """The zeros sum to -1, the check scipy itself applies."""
    for _ in range(min(n, 40)):
        order = int(rng.integers(1, 41))
        yield Case(FN, [order], {}, 1e-14, "zeros sum to -1",
                   check=lambda r: abs(complex(sum(r[0]), sum(r[1])) + 1))
