"""
Peak finding: find_peaks, peak_prominences and peak_widths against scipy.signal, plus properties.

Results are compared exactly (peak indices, every property, and int versus float): the kernels use only
comparisons, differences and one interpolation, done in scipy's order.

Where the module differs from scipy (the reference below applies these on top of scipy):
    - Ties under `distance`. scipy ranks competing peaks with numpy's default argsort, which is not stable: on the
      AVX-512 machine these tests were written on, it ordered ties differently from a stable sort in 158 of 200
      quantised arrays of 10 values and in every one from 16 values up, and that order depends on the CPU. So
      which of two equal-height peaks survives differs between computers. The module ranks ties by position, the
      later peak first, as a stable sort does; the reference is scipy's find_peaks with its distance selector
      replaced by a Python copy of the Cython loop that uses numpy's stable sort.
    - peak_widths accepts prominence_data as lists (scipy's Cython needs numpy arrays, which Ignition scripts cannot
      make); the reference converts them as scipy's own argument checks would.
    - Signals are 1-D and numbers finite (module-wide rules).
"""

import contextlib
import inspect
import math
import warnings

import numpy as np
import scipy.signal as ss
import scipy.signal._peak_finding as pf

from harness import Case, Mismatch, check_signal, industrial, make_call, pick, random_length

FN = "peaks"
FUNCS = ["find_peaks", "peak_prominences", "peak_widths"]
SIGS = {f: inspect.signature(getattr(ss, f)) for f in FUNCS}
ORDERS = {f: list(SIGS[f].parameters) for f in FUNCS}
DEFAULTS = {f: {k: v.default for k, v in SIGS[f].parameters.items() if v.default is not inspect.Parameter.empty}
            for f in FUNCS}


def stable_select_by_peak_distance(peaks, priority, distance):
    """_select_by_peak_distance from _peak_finding_utils.pyx, with numpy's stable sort for ties."""
    peaks_size = peaks.shape[0]
    c = math.ceil(distance)
    distance_ = c if -2 ** 63 <= c < 2 ** 63 else -2 ** 63
    keep = np.ones(peaks_size, dtype=bool)
    order = np.argsort(priority, kind="stable")
    for i in range(peaks_size - 1, -1, -1):
        j = order[i]
        if not keep[j]:
            continue
        k = j - 1
        while 0 <= k and peaks[j] - peaks[k] < distance_:
            keep[k] = False
            k -= 1
        k = j + 1
        while k < peaks_size and peaks[k] - peaks[j] < distance_:
            keep[k] = False
            k += 1
    return keep


@contextlib.contextmanager
def stable_distance():
    original = pf._select_by_peak_distance
    pf._select_by_peak_distance = stable_select_by_peak_distance
    try:
        yield
    finally:
        pf._select_by_peak_distance = original


def finite(v):
    if isinstance(v, (list, tuple)):
        for x in v:
            finite(x)
    elif isinstance(v, float) and not math.isfinite(v):
        raise ValueError("numbers must be finite")


def make_reference(fn):
    def reference(*args, **kwargs):
        bound = SIGS[fn].bind(*args, **kwargs)
        for k, v in bound.arguments.items():
            finite(v)
        check_signal(bound.arguments["x"])
        data = bound.arguments.get("prominence_data")
        if fn == "peak_widths" and data is not None and isinstance(data, (list, tuple)) and len(data) == 3:
            prom = np.asarray(data[0], dtype=np.float64)
            lb = pf._arg_peaks_as_expected(data[1])
            rb = pf._arg_peaks_as_expected(data[2])
            if prom.ndim != 1:
                raise ValueError("prominences must be 1-D")
            bound.arguments["prominence_data"] = (prom, lb, rb)
        with warnings.catch_warnings(), stable_distance():
            warnings.simplefilter("ignore")
            return getattr(ss, fn)(*bound.args, **bound.kwargs)
    return reference


REFERENCES = {f: make_reference(f) for f in FUNCS}
reference = REFERENCES["find_peaks"]


def plain(v):
    if isinstance(v, np.ndarray):
        return [plain(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return dict((k, plain(x)) for k, x in v.items())
    if isinstance(v, np.generic):
        return v.item()
    return v


def exact(got, want):
    """Identical structure, values and types (an int where scipy has an int)."""
    def walk(g, w, where):
        if isinstance(w, dict):
            if not isinstance(g, dict) or set(g) != set(w):
                raise Mismatch("%s: got keys %s, expected %s" % (where, sorted(g) if isinstance(g, dict) else g,
                                                                    sorted(w)))
            for k in w:
                walk(g[k], w[k], "%s[%r]" % (where, k))
        elif isinstance(w, list):
            if not isinstance(g, list) or len(g) != len(w):
                raise Mismatch("%s: got %d values, expected %d" % (where, len(g) if isinstance(g, list) else -1,
                                                                    len(w)))
            for i, (a, b) in enumerate(zip(g, w)):
                walk(a, b, "%s[%d]" % (where, i))
        else:
            if isinstance(w, bool) or isinstance(w, int):
                if not isinstance(g, int) or isinstance(g, bool) or g != w:
                    raise Mismatch("%s: got %r, expected the integer %r" % (where, g, w))
            elif not (g == w or (g != g and w != w)) or isinstance(g, int):
                raise Mismatch("%s: got %r, expected %r" % (where, g, w))
    walk(got, plain(want), "result")
    return 0.0


# Generation

def tied_signal(rng, n):
    """Small integers: plateaus, equal heights and exact ties everywhere."""
    return [float(v) for v in rng.integers(0, int(rng.integers(2, 6)), n)]


def signal(rng, cap):
    r = rng.random()
    if r < 0.35:
        return tied_signal(rng, int(rng.integers(0, 60)))
    x = industrial(rng, max(1, random_length(rng, cap)))
    if r < 0.6:
        q = float(np.ptp(x) or 1.0) / int(rng.integers(5, 60))
        x = [round(v / q) * q for v in x]
    return x


def border(rng, lo, hi):
    return pick(rng, [None, float(rng.uniform(lo, hi)), int(round(rng.uniform(lo, hi)))])


def condition(rng, lo, hi):
    r = rng.random()
    if r < 0.4:
        return float(rng.uniform(lo, hi))
    if r < 0.8:
        a, b = sorted([float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi))])
        return pick(rng, [[a, b], (a, None), [None, b], (a, b)])
    if r < 0.9:
        return [float(rng.uniform(lo, hi))]
    return [border(rng, lo, hi), border(rng, lo, hi)]


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
        yield Case(fn, args, kwargs, kind=kind, compare=exact)


def candidate_peaks(rng, x):
    found = ss.find_peaks(x)[0].tolist()
    r = rng.random()
    if r < 0.6 or not len(x):
        return found
    if r < 0.85:
        return sorted(set(found[: int(rng.integers(0, len(found) + 1))]))
    return [int(v) for v in rng.integers(0, len(x), int(rng.integers(0, 10)))]


def valid_values(rng, fn, cap):
    x = signal(rng, cap)
    lo, hi = (min(x), max(x)) if x else (0.0, 1.0)
    span = (hi - lo) or 1.0
    values = {"x": x}
    if fn == "find_peaks":
        if rng.random() < 0.4:
            values["height"] = condition(rng, lo, hi)
        if rng.random() < 0.3:
            values["threshold"] = condition(rng, 0, span / 4)
        if rng.random() < 0.4:
            values["distance"] = pick(rng, [int(rng.integers(1, 60)), float(rng.uniform(1, 60)), 1])
        if rng.random() < 0.4:
            values["prominence"] = condition(rng, 0, span)
        if rng.random() < 0.3:
            values["width"] = condition(rng, 0, 30)
        if rng.random() < 0.3:
            values["wlen"] = pick(rng, [int(rng.integers(2, 200)), float(rng.uniform(1.5, 200)), None])
        if rng.random() < 0.2:
            values["rel_height"] = pick(rng, [float(rng.uniform(0, 1.5)), 1, 0])
        if rng.random() < 0.3:
            values["plateau_size"] = condition(rng, 1, 6)
        return values
    values["peaks"] = candidate_peaks(rng, x) if x else []
    if rng.random() < 0.4:
        values["wlen"] = pick(rng, [int(rng.integers(2, 200)), float(rng.uniform(1.5, 200))])
    if fn == "peak_widths":
        if rng.random() < 0.5:
            values["rel_height"] = pick(rng, [float(rng.uniform(0, 1.5)), 1, 0])
        if rng.random() < 0.3 and x:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, lb, rb = ss.peak_prominences(x, values["peaks"])
            values["prominence_data"] = pick(rng, [[p.tolist(), lb.tolist(), rb.tolist()],
                                                   (p.tolist(), lb.tolist(), rb.tolist())])
            values.pop("wlen", None)
    return values


def invalid_values(rng):
    x = industrial(rng, int(rng.integers(20, 200)))
    what = int(rng.integers(0, 16))
    if what == 0:
        return "find_peaks", {"x": [x[:5], x[5:10]]}
    if what == 1:
        x[3] = pick(rng, [None, math.nan])
        return "find_peaks", {"x": x}
    if what == 2:
        return "find_peaks", {"x": x, "distance": pick(rng, [0.5, 0, -3])}
    if what == 3:
        return "find_peaks", {"x": x, "prominence": 0, "wlen": pick(rng, [1, 0.5, -2, True])}
    if what == 4:
        return "find_peaks", {"x": x, "width": 1, "rel_height": -0.5}
    if what == 5:
        return "find_peaks", {"x": x, "height": pick(rng, ["a", [1.0, 2.0, 3.0], [1.0] * len(x)])}
    if what == 6:
        return "peak_prominences", {"x": x, "peaks": pick(rng, [[3.0], [len(x) + 2], [-1], [[1, 2]], 5])}
    if what == 7:
        return "peak_prominences", {"x": x, "peaks": [3], "wlen": pick(rng, [1, 0])}
    if what == 8:
        return "peak_widths", {"x": x, "peaks": [3], "rel_height": -1}
    if what == 9:
        return "peak_widths", {"x": x, "peaks": [3, 9], "prominence_data": [[1.0], [2], [4]]}
    if what == 10:
        return "peak_widths", {"x": x, "peaks": [3], "prominence_data": [[1.0], [5], [4]]}
    if what == 11:
        return "peak_widths", {"x": x, "peaks": [3], "prominence_data": [[1.0], [2.0], [4]]}
    if what == 12:
        return "find_peaks", {"x": x, "plateau_size": [1, 2, 3]}
    if what == 13:
        return "find_peaks", {"x": x, "distance": [1, 2]}
    if what == 14:
        return "peak_widths", {"x": x, "peaks": [len(x) + 5]}
    return "find_peaks", {"x": x, "threshold": "big"}


def wrong_call(rng):
    what = int(rng.integers(0, 3))
    if what == 0:
        return Case("find_peaks", [], {"height": 1}, kind="call")
    if what == 1:
        return Case("peak_prominences", [[1.0, 2.0, 1.0]], {}, kind="call")
    return Case("find_peaks", [[1.0, 2.0, 1.0]], {"min_height": 1}, kind="call")


# Properties

def properties(rng, n, max_len):
    """
    Peaks found are local maxima (the middle of any plateau); with distance, kept peaks are at least ceil(distance)
    apart; prominences are between 0 and the peak's height above the signal's minimum; widths' crossings bracket
    the peak; and equal-height peaks competing under distance keep the later one.
    """
    for i in range(n):
        which = i % 5
        x = signal(rng, 3000) if which != 4 else None
        if which == 0:
            def is_maxima(r, x=x):
                peaks = r[0]
                for p in peaks:
                    lft = p
                    while lft > 0 and x[lft - 1] == x[p]:
                        lft -= 1
                    rgt = p
                    while rgt < len(x) - 1 and x[rgt + 1] == x[p]:
                        rgt += 1
                    if not (lft > 0 and rgt < len(x) - 1 and x[lft - 1] < x[p] and x[rgt + 1] < x[p]
                            and p == (lft + rgt) // 2):
                        return math.inf
                return 0.0
            yield Case("find_peaks", [x], {}, 0.0, "peaks are local maxima, at the middle of plateaus",
                       check=is_maxima)
        elif which == 1:
            d = float(rng.uniform(1, 40))
            yield Case("find_peaks", [x], {"distance": d}, 0.0, "kept peaks are at least ceil(distance) apart",
                       check=lambda r, d=d: 0.0 if all(b - a >= math.ceil(d) for a, b in zip(r[0], r[0][1:]))
                       else math.inf)
        elif which == 2:
            lo = min(x) if x else 0.0
            yield Case("find_peaks", [x], {"prominence": 0}, 0.0,
                       "prominences lie between 0 and the height above the minimum",
                       check=lambda r, x=x, lo=lo: 0.0 if all(0 <= p <= x[k] - lo for k, p in
                                                              zip(r[0], r[1]["prominences"])) else math.inf)
        elif which == 3:
            yield Case("find_peaks", [x], {"width": 0, "rel_height": float(rng.uniform(0, 1))}, 0.0,
                       "width crossings bracket the peak",
                       check=lambda r: 0.0 if all(lft <= k <= rgt for k, lft, rgt in
                                                  zip(r[0], r[1]["left_ips"], r[1]["right_ips"])) else math.inf)
        else:
            gap = int(rng.integers(2, 20))
            level = float(rng.uniform(1, 100))
            x = [0.0] * (gap + 10)
            first = 4
            x[first] = x[first + gap] = level
            yield Case("find_peaks", [x], {"distance": gap + 1}, 0.0,
                       "of two equal peaks closer than distance, the later is kept",
                       check=lambda r, keep=first + gap: 0.0 if list(r[0]) == [keep] else math.inf)
