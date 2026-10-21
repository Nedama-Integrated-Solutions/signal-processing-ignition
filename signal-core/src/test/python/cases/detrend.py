"""
detrend: parity with scipy.signal.detrend, plus properties that need no reference.

Where the module differs from scipy (the reference below applies these on top of scipy):
    - Signals must be 1-D and must not contain None, NaN or infinity (module-wide rule).
    - An empty breakpoint list means no breakpoints. scipy raises a TypeError for bp=[].
"""

import inspect
import math

import scipy.signal as ss

from harness import EPS, Case, check_signal, industrial, make_call, pick, random_length, scale_of

FN = "detrend"
ORDER = ["data", "axis", "type", "bp", "overwrite_data"]
DEFAULTS = {"axis": -1, "type": "linear", "bp": 0, "overwrite_data": False}
SIGNATURE = inspect.signature(ss.detrend)

# Measured agreement with scipy is under 8 eps of the signal's largest magnitude, up to 200,000 points.
RTOL = 64 * EPS


def reference(*args, **kwargs):
    bound = SIGNATURE.bind(*args, **kwargs)
    a = bound.arguments
    check_signal(a["data"])
    if a.get("type", "linear") in ("linear", "l") and isinstance(a.get("bp"), list) and not a["bp"]:
        a["bp"] = 0
    return ss.detrend(*bound.args, **bound.kwargs)


def tolerance(data):
    return RTOL * scale_of(data)


def generate(rng, n, max_len):
    """Random calls: mostly valid, some invalid values, some wrong calls."""
    for _ in range(n):
        r = rng.random()
        if r < 0.80:
            yield valid_call(rng, max_len)
        elif r < 0.92:
            yield invalid_value(rng)
        else:
            yield wrong_call(rng)


def valid_call(rng, max_len):
    n = random_length(rng, max_len)
    data = industrial(rng, n)
    if rng.random() < 0.1:
        data = [int(round(v)) for v in data]
    values = {"data": data}
    if rng.random() < 0.75:
        values["type"] = pick(rng, ["linear", "l", "constant", "c"])
    if rng.random() < 0.6:
        values["bp"] = random_breakpoints(rng, n)
    if rng.random() < 0.2:
        values["axis"] = pick(rng, [-1, 0])
    if rng.random() < 0.1:
        values["overwrite_data"] = bool(rng.random() < 0.5)
    args, kwargs = make_call(rng, ORDER, values, DEFAULTS)
    return Case(FN, args, kwargs, tolerance(data), "random")


def random_breakpoints(rng, n):
    r = rng.random()
    if r < 0.3:
        return int(rng.integers(0, n + 1))
    if r < 0.35:
        return []
    k = int(rng.integers(1, 6))
    bp = [int(v) for v in rng.integers(0, n + 1, k)]
    if rng.random() < 0.5:
        bp.sort()
    return bp


def invalid_value(rng):
    n = int(rng.integers(2, 200))
    data = industrial(rng, n)
    values = {"data": data}
    what = int(rng.integers(0, 12))
    if what == 0:
        data[int(rng.integers(0, n))] = None
    elif what == 1:
        data[int(rng.integers(0, n))] = pick(rng, [math.nan, math.inf, -math.inf])
    elif what == 2:
        values["data"] = [data[: n // 2], data[n // 2:]]
    elif what == 3:
        values["data"] = data[0]
    elif what == 4:
        values["data"] = ["a", "b", "c"]
    elif what == 5:
        values["type"] = pick(rng, ["quadratic", "LINEAR", 5, None])
    elif what == 6:
        values["bp"] = pick(rng, [n + 1, n + 100, -1])
    elif what == 7:
        values["bp"] = pick(rng, [2.5, [1.0], [1, 2.5], "a", None])
    elif what == 8:
        values["axis"] = pick(rng, [1, -2, 0.0, "0"])
    elif what == 9:
        values["data"] = []
        values["type"] = pick(rng, ["linear", "constant"])
    elif what == 10:
        values["data"] = data
        values["type"] = "constant"
        values["bp"] = "ignored for constant"
    else:
        values["data"] = [True, False, True, True]
    args, kwargs = make_call(rng, ORDER, values, DEFAULTS)
    return Case(FN, args, kwargs, tolerance(values["data"]), "invalid")


def wrong_call(rng):
    data = industrial(rng, int(rng.integers(2, 50)))
    what = int(rng.integers(0, 4))
    if what == 0:
        return Case(FN, [data], {"typ": "linear"}, 0.0, "call")
    if what == 1:
        return Case(FN, [data, -1, "linear", 0, False, 1], {}, 0.0, "call")
    if what == 2:
        return Case(FN, [data], {"data": data}, 0.0, "call")
    return Case(FN, [], {}, 0.0, "call")


def properties(rng, n, max_len):
    """Checks that hold whatever scipy does: exact fits, zero mean, orthogonal residuals."""
    for i in range(n):
        which = i % 4
        m = max(2, random_length(rng, max_len))
        if which == 0:
            a, b = rng.uniform(-1e3, 1e3), rng.uniform(-10, 10)
            data = [a + b * k for k in range(m)]
            tol = tolerance(data)
            yield Case(FN, [data], {}, tol, "line is removed exactly", check=max_abs)
        elif which == 1:
            cuts = sorted({int(v) for v in rng.integers(1, m, int(rng.integers(1, 5)))})
            data, start = [], 0
            for end in cuts + [m]:
                a, b = rng.uniform(-1e3, 1e3), rng.uniform(-10, 10)
                data += [a + b * k for k in range(end - start)]
                start = end
            yield Case(FN, [data], {"bp": cuts}, tolerance(data), "piecewise line is removed exactly", check=max_abs)
        elif which == 2:
            data = industrial(rng, m)
            yield Case(FN, [data], {"type": "constant"}, tolerance(data), "constant: zero mean", check=mean_abs)
        else:
            data = industrial(rng, m)
            yield Case(FN, [data], {}, tolerance(data), "linear: residuals orthogonal to the line",
                       check=orthogonality)


def max_abs(r):
    return max((abs(v) for v in r), default=0.0)


def mean_abs(r):
    return abs(math.fsum(r)) / len(r)


def orthogonality(r):
    m = len(r)
    u = [k - (m - 1) / 2.0 for k in range(m)]
    return max(abs(math.fsum(r)) / m, abs(math.fsum(a * b for a, b in zip(r, u))) / math.fsum(abs(v) for v in u))
