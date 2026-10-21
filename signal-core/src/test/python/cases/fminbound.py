"""
The internal bounded minimiser (_fminbound) against scipy.optimize.fminbound.

The test functions use only +, -, *, / and abs, which Python and Java evaluate identically, and the minimiser
is a step-for-step port. So the result must match exactly: the same x and f(x) to the last bit, the same status
and the same number of function calls. Any difference means the port has drifted from scipy.
"""

import math

import scipy.optimize as so

from harness import Case, pick

FN = "_fminbound"


def test_function(name, p):
    if name == "quadratic":
        return lambda x: p[1] * (x - p[0]) * (x - p[0]) + p[2]
    if name == "quartic":
        return lambda x: (x * x - p[0]) * (x * x - p[0]) + p[1] * x
    if name == "abs":
        return lambda x: abs(x - p[0])
    if name == "rational":
        return lambda x: (x - p[0]) * (x - p[0]) / (1 + x * x)
    if name == "piecewise":
        return lambda x: p[1] * (p[0] - x) if x < p[0] else x - p[0]
    if name == "cubic":
        return lambda x: ((p[0] * x + p[1]) * x + p[2]) * x
    if name == "flat":
        return lambda x: 1.0
    if name == "nan_above":
        return lambda x: math.nan if x > p[0] else (x - p[0]) * (x - p[0])
    raise ValueError("unknown test function " + name)


def reference(func, x1, x2, params=(), xtol=1e-5, maxfun=500):
    f = test_function(func, [float(v) for v in params])
    x, fval, status, nfev = so.fminbound(f, x1, x2, xtol=xtol, maxfun=maxfun, full_output=True, disp=0)
    return [float(x), float(fval), int(status), int(nfev)]


PARAMS = {"quadratic": 3, "quartic": 2, "abs": 1, "rational": 1, "piecewise": 2, "cubic": 3, "flat": 0,
          "nan_above": 1}


def generate(rng, n, max_len):
    for _ in range(n):
        name = pick(rng, sorted(PARAMS))
        params = [float(v) for v in rng.uniform(-5, 5, PARAMS[name])]
        if name == "quadratic":
            params[1] = abs(params[1]) + 0.01
        lo = float(rng.uniform(-10, 5))
        hi = lo + float(10.0 ** rng.uniform(-6, 1.3))
        kwargs = {}
        if rng.random() < 0.5:
            kwargs["xtol"] = float(10.0 ** rng.uniform(-12, -1))
        if rng.random() < 0.3:
            kwargs["maxfun"] = int(rng.integers(1, 60))
        yield Case(FN, [name, lo, hi, params], kwargs, 0.0)
    yield Case(FN, ["quadratic", 2.0, 1.0, [0.0, 1.0, 0.0]], {}, 0.0, "invalid")


def properties(rng, n, max_len):
    """
    For a parabola inside the bounds, the minimum is found to within Brent's own limit: xtol plus sqrt(eps) |x|
    from the stopping test, plus about sqrt(eps) because a parabola is too flat near its minimum for smaller
    steps in x to change f(x).
    """
    for _ in range(n):
        c = float(rng.uniform(-5, 5))
        xtol = float(10.0 ** rng.uniform(-10, -3))
        limit = 4 * xtol + 4 * math.sqrt(2.2e-16) * (abs(c) + 1)
        yield Case(FN, ["quadratic", c - 7.0, c + 3.0, [c, 2.0, 1.0]], {"xtol": xtol}, limit,
                   "parabola minimum found within Brent's tolerance", check=lambda r, c=c: abs(r[0] - c))
