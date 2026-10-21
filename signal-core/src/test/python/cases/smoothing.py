"""
Smoothing: savgol_coeffs, savgol_filter and medfilt against scipy.signal, plus properties.

Where the module differs from scipy:
    - Savitzky-Golay least-squares problems are solved exactly and rounded once. scipy solves them with LAPACK and a
      cutoff of machine epsilon on singular values, which loses accuracy as the Vandermonde matrix grows
      ill-conditioned: against the exact coefficients, scipy is within 1e-8 relative for polyorder up to 4 and
      windows up to 201, but 2e-3 wrong at polyorder 10 / window 51 and entirely wrong at polyorder 8 / window 201,
      polyorder 10 / window 101 and polyorder 12 / window 51. This module returns the exact coefficients there too.
      So the reference below is scipy's algorithm with its least-squares steps solved exactly: exact coefficients,
      convolved by scipy's own ndimage.convolve1d (so interior outputs must match bit for bit), and exact
      polynomial fits for 'interp' edges (which must match within an ulp). scipy itself is still called first,
      so its argument errors decide which calls must fail.
    - window_length, polyorder and deriv must be integers (scipy accepts a float window_length in some modes).
    - Signals are 1-D and numbers finite (module-wide rules).
"""

import functools
import inspect
import math
import warnings
from fractions import Fraction

import numpy as np
import scipy.ndimage as nd
import scipy.signal as ss

from harness import EPS, Case, Mismatch, check_signal, industrial, make_call, pick, random_length

FN = "smoothing"
FUNCS = ["savgol_coeffs", "savgol_filter", "medfilt"]
SIGS = {"savgol_coeffs": inspect.signature(lambda window_length, polyorder, deriv=0, delta=1.0, pos=None,
                                                  use="conv": None),
        "savgol_filter": inspect.signature(ss.savgol_filter),
        "medfilt": inspect.signature(ss.medfilt)}
ORDERS = {f: list(SIGS[f].parameters) for f in FUNCS}
DEFAULTS = {f: {k: v.default for k, v in SIGS[f].parameters.items() if v.default is not inspect.Parameter.empty}
            for f in FUNCS}


# Exact least squares

def solve(m, b):
    n = len(b)
    a = [list(row) + [b[i]] for i, row in enumerate(m)]
    for k in range(n):
        p = next(i for i in range(k, n) if a[i][k] != 0)
        a[k], a[p] = a[p], a[k]
        for i in range(k + 1, n):
            if a[i][k] != 0:
                f = a[i][k] / a[k][k]
                for j in range(k, n + 1):
                    a[i][j] -= f * a[k][j]
    x = [None] * n
    for i in range(n - 1, -1, -1):
        x[i] = (a[i][n] - sum(a[i][j] * x[j] for j in range(i + 1, n))) / a[i][i]
    return x


@functools.lru_cache(maxsize=512)
def exact_coeffs(w, p, deriv, delta, pos, use):
    """The minimum-norm solution of scipy's savgol_coeffs system, solved exactly, rounded once."""
    halflen, rem = divmod(w, 2)
    if pos is None:
        pos = halflen - 0.5 if rem == 0 else halflen
    if deriv > p:
        return tuple([0.0] * w)
    yd = float(math.factorial(deriv)) / (delta ** deriv)
    x = [Fraction(float(-pos + i)) for i in range(w)]
    if use == "conv":
        x = x[::-1]
    rows = p + 1
    a = [[xj ** i for xj in x] for i in range(rows)]
    g = [[sum(a[i][j] * a[k][j] for j in range(w)) for k in range(rows)] for i in range(rows)]
    e = [Fraction(0)] * rows
    e[deriv] = Fraction(yd)
    z = solve(g, e)
    return tuple(float(sum(a[i][j] * z[i] for i in range(rows))) for j in range(w))


def exact_edge(xe, p, deriv, delta, points):
    """scipy's _fit_edge solved exactly: least-squares polynomial, derivative, values at points, / delta**deriv."""
    m = len(xe)
    order = p + 1
    sums = [sum(Fraction(t) ** k for t in range(m)) for k in range(2 * order - 1)]
    rhs = [sum(Fraction(t) ** k * Fraction(xe[t]) for t in range(m)) for k in range(order)]
    c = solve([[sums[i + k] for k in range(order)] for i in range(order)], rhs)
    for _ in range(deriv):
        c = [c[k] * k for k in range(1, len(c))] if len(c) > 1 else [Fraction(0)]
    power = delta ** deriv
    out = []
    for i in points:
        v = sum(ck * Fraction(i) ** k for k, ck in enumerate(c))
        if power == 0:
            out.append(math.nan if v == 0 else math.copysign(math.inf, v))
        else:
            out.append(float(v / Fraction(power)))
    return out


# References

def int_args(bound):
    for name in ("window_length", "polyorder", "deriv"):
        v = bound.get(name)
        if isinstance(v, float):
            raise TypeError("%s must be an integer" % name)


def finite(v):
    if isinstance(v, (list, tuple)):
        for x in v:
            finite(x)
    elif isinstance(v, float) and not math.isfinite(v):
        raise ValueError("numbers must be finite")


def ref_savgol_coeffs(*args, **kwargs):
    b = SIGS["savgol_coeffs"].bind(*args, **kwargs)
    b.apply_defaults()
    a = b.arguments
    int_args(a)
    for v in a.values():
        finite(v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss.savgol_coeffs(*b.args, **b.kwargs)
    return list(exact_coeffs(a["window_length"], a["polyorder"], a["deriv"], float(a["delta"]),
                             a["pos"], a["use"]))


def ref_savgol_filter(*args, **kwargs):
    b = SIGS["savgol_filter"].bind(*args, **kwargs)
    b.apply_defaults()
    a = b.arguments
    int_args(a)
    for v in a.values():
        finite(v)
    check_signal(a["x"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss.savgol_filter(*b.args, **b.kwargs)
        x = np.asarray(a["x"], float)
        w, p, deriv, delta, mode = a["window_length"], a["polyorder"], a["deriv"], float(a["delta"]), a["mode"]
        c = np.asarray(exact_coeffs(w, p, deriv, delta, None, "conv"))
        if x.size == 0:
            return []
        if mode != "interp":
            return nd.convolve1d(x, c, mode=mode, cval=a["cval"]).tolist()
        y = nd.convolve1d(x, c, mode="constant").tolist()
        half = w // 2
        n = len(x)
        y[:half] = exact_edge(x[:w].tolist(), p, deriv, delta, range(half))
        y[n - half:] = exact_edge(x[n - w:].tolist(), p, deriv, delta, range(w - half, w))
        return y


def ref_medfilt(*args, **kwargs):
    b = SIGS["medfilt"].bind(*args, **kwargs)
    check_signal(b.arguments["volume"])
    finite(b.arguments["volume"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ss.medfilt(*b.args, **b.kwargs)


REFERENCES = {"savgol_coeffs": ref_savgol_coeffs, "savgol_filter": ref_savgol_filter, "medfilt": ref_medfilt}
reference = ref_savgol_filter


def within_ulp(got, want):
    """Worst difference in units of 2 ulps of the expected value (the exact value, rounded once on each side)."""
    if len(got) != len(want):
        raise Mismatch("got %d values, expected %d" % (len(got), len(want)))
    worst = 0.0
    for g, w in zip(got, want):
        if w != w or g != g or abs(w) == math.inf:
            if not (g == w or (g != g and w != w)):
                raise Mismatch("got %r, expected %r" % (g, w))
            continue
        d = abs(g - w)
        if d:
            worst = max(worst, d / (2 * math.ulp(abs(w)) if w else 1e-300))
    return worst


# Generation

def generate(rng, n, max_len):
    cap = min(max_len, 5000)
    for _ in range(n):
        r = rng.random()
        fn = pick(rng, FUNCS)
        if r < 0.82:
            values = valid_values(rng, fn, cap)
            kind = "random"
        elif r < 0.96:
            fn, values = invalid_values(rng)
            kind = "invalid"
        else:
            yield wrong_call(rng)
            continue
        args, kwargs = make_call(rng, ORDERS[fn], values, DEFAULTS[fn])
        if fn == "medfilt":
            yield Case(fn, args, kwargs, 0.0, kind)
        else:
            yield Case(fn, args, kwargs, kind=kind, compare=within_ulp)


def window_and_order(rng, max_w):
    w = int(rng.integers(1, max_w + 1))
    p = int(rng.integers(-1, min(w - 1, 10) + 1)) if rng.random() < 0.1 else int(rng.integers(0, min(w - 1, 8) + 1))
    return w, p


def valid_values(rng, fn, cap):
    if fn == "savgol_coeffs":
        w, p = window_and_order(rng, 61)
        values = {"window_length": w, "polyorder": p}
        if rng.random() < 0.6:
            values["deriv"] = int(rng.integers(0, max(p, 0) + 2))
        if rng.random() < 0.4:
            values["delta"] = pick(rng, [float(rng.uniform(0.01, 10)), 0.5, 2.0])
        if rng.random() < 0.3:
            values["pos"] = pick(rng, [int(rng.integers(0, w)), float(rng.uniform(0, w - 1e-9))])
        if rng.random() < 0.4:
            values["use"] = pick(rng, ["conv", "dot"])
        return values
    if fn == "savgol_filter":
        length = max(1, random_length(rng, cap))
        x = industrial(rng, length)
        w, p = window_and_order(rng, min(51, max(1, length)) if rng.random() < 0.9 else 51)
        values = {"x": x, "window_length": w, "polyorder": p}
        if rng.random() < 0.5:
            values["deriv"] = int(rng.integers(0, 3))
        if rng.random() < 0.3:
            values["delta"] = float(rng.uniform(0.01, 10))
        if rng.random() < 0.6:
            values["mode"] = pick(rng, ["mirror", "constant", "nearest", "wrap", "interp"])
        if rng.random() < 0.2:
            values["cval"] = float(rng.uniform(-10, 10))
        if rng.random() < 0.1:
            values["axis"] = pick(rng, [-1, 0])
        return values
    length = random_length(rng, cap)
    x = industrial(rng, length)
    if rng.random() < 0.3:
        x = [int(round(v)) for v in x]
    values = {"volume": x}
    if rng.random() < 0.8:
        values["kernel_size"] = pick(rng, [1, 3, 5, 7, 9, 15, 31, 101, [5], int(2 * rng.integers(0, 60) + 1)])
    return values


def invalid_values(rng):
    x = industrial(rng, int(rng.integers(10, 200)))
    what = int(rng.integers(0, 15))
    if what == 0:
        return "savgol_coeffs", {"window_length": 5, "polyorder": 5}
    if what == 1:
        return "savgol_coeffs", {"window_length": 5, "polyorder": 2, "pos": 5}
    if what == 2:
        return "savgol_coeffs", {"window_length": 5, "polyorder": 2, "use": "conv2"}
    if what == 3:
        return "savgol_coeffs", {"window_length": 5, "polyorder": 2, "deriv": 1, "delta": 0.0}
    if what == 4:
        return "savgol_coeffs", {"window_length": 5, "polyorder": 2, "deriv": -1}
    if what == 5:
        return "savgol_filter", {"x": x, "window_length": 7, "polyorder": 2, "mode": pick(rng, ["reflect", "bad"])}
    if what == 6:
        return "savgol_filter", {"x": x[:4], "window_length": 7, "polyorder": 2}
    if what == 7:
        return "savgol_filter", {"x": x, "window_length": 7.0, "polyorder": 2}
    if what == 8:
        x[3] = pick(rng, [None, math.nan])
        return "savgol_filter", {"x": x, "window_length": 7, "polyorder": 2}
    if what == 9:
        return "medfilt", {"volume": x, "kernel_size": pick(rng, [4, 0, -1, 3.0, [3, 5]])}
    if what == 10:
        return "medfilt", {"volume": [True, False, True]}
    if what == 11:
        return "medfilt", {"volume": [[1.0, 2.0], [3.0, 4.0]]}
    if what == 12:
        return "savgol_filter", {"x": x, "window_length": 7, "polyorder": -1, "mode": "interp"}
    if what == 13:
        return "savgol_filter", {"x": x, "window_length": 9, "polyorder": 2, "deriv": 4, "delta": 0.0}
    return "medfilt", {"volume": pick(rng, [x[0], int(round(x[0] * 100)), True])}


def wrong_call(rng):
    what = int(rng.integers(0, 3))
    if what == 0:
        return Case("savgol_filter", [[1.0, 2.0, 3.0], 3], {}, kind="call")
    if what == 1:
        return Case("savgol_coeffs", [5, 2], {"derivative": 1}, kind="call")
    return Case("medfilt", [[1.0, 2.0, 3.0], 3, 1], {}, kind="call")


# Properties

def properties(rng, n, max_len):
    """
    savgol_filter reproduces any polynomial of degree up to polyorder, and its derivative; savgol_coeffs sum to 1;
    medfilt removes an isolated spike and leaves a monotonic signal's interior unchanged.
    """
    for i in range(n):
        which = i % 4
        if which == 0:
            w, p = int(2 * rng.integers(2, 16) + 1), int(rng.integers(1, 6))
            p = min(p, w - 1)
            length = int(rng.integers(w, 400))
            coef = [float(v) for v in rng.uniform(-1, 1, p + 1)]
            t = np.arange(length) / length * 4 - 2
            x = np.polyval(coef, t)
            delta = 4.0 / length
            deriv = int(rng.integers(0, 2))
            target = np.polyval(np.polyder(coef), t) if deriv else x
            scale = float(np.max(np.abs(target))) + float(np.max(np.abs(x)))
            yield Case("savgol_filter", [x.tolist(), w, p], {"deriv": deriv, "delta": delta}, 1e-8 * scale * w,
                       "reproduces a degree-%d polynomial%s" % (p, " derivative" if deriv else ""),
                       check=lambda r, target=target: float(np.max(np.abs(np.asarray(r) - target))))
        elif which == 1:
            w, p = int(2 * rng.integers(1, 30) + 1), int(rng.integers(0, 7))
            p = min(p, w - 1)
            yield Case("savgol_coeffs", [w, p], {}, 64 * EPS * w, "smoothing coefficients sum to 1",
                       check=lambda r: abs(math.fsum(r) - 1))
        elif which == 2:
            length = int(rng.integers(50, 500))
            level = float(rng.uniform(-10, 10))
            x = [level] * length
            spike = int(rng.integers(10, length - 10))
            x[spike] = level + float(rng.uniform(50, 100))
            yield Case("medfilt", [x, 5], {}, 0.0, "removes an isolated spike",
                       check=lambda r, level=level: max(abs(v - level) for v in r[2:-2]))
        else:
            length = int(rng.integers(50, 500))
            x = sorted(float(v) for v in rng.uniform(-100, 100, length))
            k = int(2 * rng.integers(1, 8) + 1)
            h = k // 2
            yield Case("medfilt", [x, k], {}, 0.0, "leaves a monotonic signal's interior unchanged",
                       check=lambda r, x=x, h=h: max(abs(a - b) for a, b in zip(r[h:-h], x[h:-h])))
