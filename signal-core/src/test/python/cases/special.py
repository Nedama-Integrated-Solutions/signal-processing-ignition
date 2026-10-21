"""
The internal special functions (_i0, _ellipk, _ellipkm1, _ellipj) against scipy.special.

These are ports of the same cephes code scipy uses, so they agree to within the rounding of the elementary
functions: measured within 1 eps for i0, exactly for ellipk and ellipkm1, and within 5 eps for ellipj.
"""

import numpy as np
import scipy.special as sp

from harness import EPS, Case, pick

FN = "_special"


def ref_i0(x):
    return sp.i0(np.asarray(x, float))


def ref_ellipk(m):
    return sp.ellipk(np.asarray(m, float))


def ref_ellipkm1(p):
    return sp.ellipkm1(np.asarray(p, float))


def ref_ellipj(u, m):
    if len(u) != len(m):
        raise ValueError("u and m must have the same length")
    return [r.tolist() for r in sp.ellipj(np.asarray(u, float), np.asarray(m, float))]


REFERENCES = {"_i0": ref_i0, "_ellipk": ref_ellipk, "_ellipkm1": ref_ellipkm1, "_ellipj": ref_ellipj}
reference = ref_i0


def values(rng, ranges, k):
    out = []
    for _ in range(k):
        lo, hi, kind = pick(rng, ranges)
        if kind == "log":
            out.append(float(10.0 ** rng.uniform(lo, hi)))
        elif kind == "one_minus_log":
            out.append(float(1 - 10.0 ** -rng.uniform(lo, hi)))
        else:
            out.append(float(rng.uniform(lo, hi)))
    return out


def generate(rng, n, max_len):
    for _ in range(n):
        which = pick(rng, ["_i0", "_ellipk", "_ellipkm1", "_ellipj"])
        k = int(rng.integers(1, 200))
        if which == "_i0":
            x = values(rng, [(0, 8, "lin"), (8, 720, "lin"), (-50, 0, "lin"), (-300, 0, "log")], k)
            yield Case(which, [x], {}, 0.0, rtol=16 * EPS)
        elif which == "_ellipk":
            m = values(rng, [(-1e3, 1, "lin"), (0, 1, "lin"), (1, 16, "one_minus_log"), (-300, 300, "log"),
                             (1, 3, "lin")], k)
            yield Case(which, [m], {}, 0.0, rtol=8 * EPS)
        elif which == "_ellipkm1":
            p = values(rng, [(-300, 0, "log"), (0, 1e3, "lin"), (0, 300, "log"), (-2, 0, "lin")], k)
            yield Case(which, [p], {}, 0.0, rtol=8 * EPS)
        else:
            u = values(rng, [(-30, 30, "lin"), (-1, 1, "lin")], k)
            m = values(rng, [(0, 1, "lin"), (-12, -9, "log"), (9, 13, "one_minus_log"), (2, 9, "one_minus_log"),
                             (-0.5, 1.5, "lin")], k)
            yield Case(which, [u, m], {}, 32 * EPS, rtol=32 * EPS)


def properties(rng, n, max_len):
    """
    sn^2 + cn^2 = 1 and dn^2 + m sn^2 = 1, for any u and m in [0, 1]. scipy itself meets the first to 1 eps and
    the second only to 290 eps (cephes computes dn as a ratio of cosines), so the check allows 4 eps and 512 eps,
    reported as a fraction of those limits.
    """
    for _ in range(n):
        k = int(rng.integers(1, 100))
        u = [float(v) for v in rng.uniform(-20, 20, k)]
        m = [float(v) for v in rng.uniform(0, 1, k)]

        def identities(r, m=m):
            sn, cn, dn = r[0], r[1], r[2]
            return max(max(abs(s * s + c * c - 1) / (4 * EPS), abs(d * d + mm * s * s - 1) / (512 * EPS))
                       for s, c, d, mm in zip(sn, cn, dn, m))

        yield Case("_ellipj", [u, m], {}, 1.0, "Jacobi identities", check=identities)
