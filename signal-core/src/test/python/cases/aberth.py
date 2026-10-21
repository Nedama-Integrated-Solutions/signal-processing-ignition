"""
The internal Aberth-Ehrlich root finder (_aberth) against scipy.signal._filter_design._aberth.

Polynomials have random, well-separated real and complex-conjugate roots; both sides start from the same
points and run the same iteration, so they converge to the same roots in the same order. When scipy fails to
converge (it raises), Java must raise too.

The tests use polynomials up to degree 10 with roots within 1.5 of the origin and at least 0.1 apart, and stop
at tol=1e-10. Larger roots or degrees make the coefficients large, and rounding in evaluating the polynomial
then sets a floor on the Newton step (scipy itself stalls at 3e-10 on a degree-12 case with roots near 2.4), so
whether either side gets under tol depends on the last bit of every operation. These tests check the
iteration; Bessel filters, the real use, are tested directly against scipy's _bessel_zeros in milestone 2.
"""

import math

import numpy as np
import scipy.signal._filter_design as fd

from harness import Case

FN = "_aberth"


def reference(coeffs, x0re, x0im, tol=1e-15, maxiter=50):
    c = np.asarray(coeffs, float)
    d = np.polyder(c)
    x0 = np.asarray(x0re, float) + 1j * np.asarray(x0im, float)
    x = fd._aberth(lambda z: np.polyval(c, z), lambda z: np.polyval(d, z), x0, tol, maxiter)
    return [x.real.tolist(), x.imag.tolist()]


def random_roots(rng, degree):
    roots = []
    while len(roots) < degree:
        if degree - len(roots) >= 2 and rng.random() < 0.6:
            z = complex(rng.uniform(-1.5, 1.5), rng.uniform(0.1, 1.5))
            candidates = [z, z.conjugate()]
        else:
            candidates = [complex(rng.uniform(-1.5, 1.5), 0)]
        if all(abs(a - b) > 0.1 for a in candidates for b in roots):
            roots += candidates
    return roots


def starting_points(rng, roots):
    n = len(roots)
    radius = 1.3 * max(abs(z) for z in roots) + 0.2
    offset = float(rng.uniform(0.2, 0.6))
    return [radius * complex(math.cos(2 * math.pi * k / n + offset), math.sin(2 * math.pi * k / n + offset))
            for k in range(n)]


def generate(rng, n, max_len):
    for _ in range(n):
        degree = int(rng.integers(1, 11))
        roots = random_roots(rng, degree)
        coeffs = np.poly(roots).real.tolist()
        x0 = starting_points(rng, roots)
        scale = max(1.0, max(abs(z) for z in roots))
        kwargs = {"tol": 1e-10}
        if rng.random() < 0.2:
            kwargs["maxiter"] = int(rng.integers(1, 20))
        yield Case(FN, [coeffs, [z.real for z in x0], [z.imag for z in x0]], kwargs, 1e-9 * scale)


def properties(rng, n, max_len):
    """
    The roots found are the polynomial's roots, whenever scipy's own iteration converges on the same polynomial
    and starting points. Where it does not (a degree-12 case with roots up to 1.52 stalled in scipy at tol 1e-10),
    the value tests above already require Java to fail too.
    """
    for _ in range(n):
        degree = int(rng.integers(1, 13))
        roots = random_roots(rng, degree)
        coeffs = np.poly(roots).real.tolist()
        x0 = starting_points(rng, roots)
        try:
            reference(coeffs, [z.real for z in x0], [z.imag for z in x0], 1e-10)
        except Exception:
            continue

        def matches(r, roots=roots):
            found = [complex(a, b) for a, b in zip(r[0], r[1])]
            return max(min(abs(f - z) for f in found) for z in roots)

        yield Case(FN, [coeffs, [z.real for z in x0], [z.imag for z in x0]], {"tol": 1e-10}, 1e-8,
                   "finds the polynomial's roots", check=matches)
