"""
Filtering: lfilter, lfilter_zi, sosfilt, sosfilt_zi, filtfilt and sosfiltfilt against scipy.signal, plus properties.

Where the module differs from scipy (the reference below applies these on top of scipy):
    - Signals are 1-D and numbers must be finite (module-wide rules).
    - filtfilt's method='gust' is not supported (ValueError); 'pad', the default, is.

Comparison. The Java loops perform scipy's operations in scipy's order, so results are normally identical to the
last bit (measured: every IIR lfilter, sosfilt, filtfilt, sosfiltfilt, lfilter_zi and sosfilt_zi result in 1500
random cases). Two things can make them differ by roundings: lfilter's FIR path, where scipy convolves with numpy
(which sums in BLAS order, measured within 10 eps of the output's size), and a platform whose compiler fuses
multiply-adds in scipy's loops. A result that is not identical is compared by backward error: a filter computed in
floating point gives the exact output of a problem whose inputs moved by a few roundings, so scipy is re-run with
every coefficient, sample and initial condition moved by 4 roundings (random signs, three draws), and the result
may differ by twice the spread that produces, or 8 eps of the output's size, whichever is larger. For the FIR
path, where random nudges partly cancel, the classic rounding bound of a convolution applies as well (fir_bound).
"""

import inspect
import math
import warnings

import numpy as np
import scipy.signal as ss

from harness import EPS, Case, Mismatch, check_signal, industrial, make_call, pick, random_length

FN = "filtering"
FUNCS = ["lfilter", "lfilter_zi", "sosfilt", "sosfilt_zi", "filtfilt", "sosfiltfilt"]
SIGS = {f: inspect.signature(getattr(ss, f)) for f in FUNCS}
ORDERS = {f: list(SIGS[f].parameters) for f in FUNCS}
DEFAULTS = {f: {k: v.default for k, v in SIGS[f].parameters.items() if v.default is not inspect.Parameter.empty}
            for f in FUNCS}


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
            if k == "x":
                check_signal(v)
        if fn == "filtfilt" and bound.arguments.get("method", "pad") == "gust":
            raise ValueError("method='gust' is not supported")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return getattr(ss, fn)(*bound.args, **bound.kwargs)
    return reference


REFERENCES = {f: make_reference(f) for f in FUNCS}
reference = REFERENCES["lfilter"]


# Comparison

def flat(v):
    if isinstance(v, (list, tuple)):
        return [x for item in v for x in flat(item)]
    if isinstance(v, np.ndarray):
        return [float(x) for x in v.ravel()]
    return [v]


def shape_of(v):
    if isinstance(v, (list, tuple)):
        return [shape_of(x) for x in v]
    return None


def nudge(v, rng):
    """Every float in v moved by 4 roundings, with random signs."""
    if isinstance(v, (list, tuple)):
        return [nudge(x, rng) for x in v]
    if isinstance(v, float):
        return v * (1 + (4 if rng.random() < 0.5 else -4) * EPS)
    return v


def fir_bound(fn, args, kwargs):
    """
    For lfilter's and filtfilt's FIR path (a single denominator coefficient): numpy convolves in BLAS order, and
    random nudges partly cancel, so the classic bound is used instead: each pass is a sum of L products with
    |result| <= sum|b/a0| max|input|, so its rounding is at most (L + 2) eps sum|b/a0| max|input|, and filtfilt
    runs two passes.
    """
    if fn not in ("lfilter", "filtfilt"):
        return 0.0
    try:
        bound = SIGS[fn].bind(*args, **kwargs).arguments
        b = np.atleast_1d(np.asarray(bound["b"], float))
        a = np.atleast_1d(np.asarray(bound["a"], float))
        x = np.asarray(bound["x"], float)
    except Exception:
        return 0.0
    if a.size != 1 or x.size == 0 or a[0] == 0:
        return 0.0
    s_abs = float(np.sum(np.abs(b)) / abs(a[0]))
    peak = float(np.max(np.abs(x)))
    zi = bound.get("zi")
    zi_peak = float(np.max(np.abs(zi))) if zi is not None and len(zi) else 0.0
    taps = b.size + 2
    if fn == "lfilter":
        return 2 * taps * EPS * (s_abs * peak + zi_peak)
    return 4 * taps * EPS * s_abs * (s_abs * peak) * 3


def comparator(fn, args, kwargs):
    fir = fir_bound(fn, args, kwargs)

    def cmp(got, want):
        want = to_plain(want)
        if shape_of(got) != shape_of(want):
            raise Mismatch("result shape differs: got %s, expected %s" % (shape_of(got), shape_of(want)))
        g, w = flat(got), flat(want)
        if all(a == b or (a != a and b != b) for a, b in zip(g, w)):
            return 0.0
        scale = max([abs(v) for v in w if v == v and abs(v) != math.inf] + [1e-300])
        rng = np.random.default_rng(len(g))
        spread = [0.0] * len(w)
        for _ in range(3):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = flat(to_plain(getattr(ss, fn)(*nudge(list(args), rng),
                                                      **{k: nudge(v, rng) for k, v in kwargs.items()})))
            except Exception:
                continue
            if len(p) == len(w):
                spread = [max(s, abs(a - b)) for s, a, b in zip(spread, p, w)]
        worst = 0.0
        for a, b, s in zip(g, w, spread):
            if a != a or b != b or abs(a) == math.inf or abs(b) == math.inf:
                if not (a == b or (a != a and b != b)):
                    raise Mismatch("got %r, expected %r" % (a, b))
                continue
            worst = max(worst, abs(a - b) / max(8 * EPS * scale, 2 * s, fir))
        return worst
    return cmp


def to_plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [to_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


# Generation

def design(rng, output, max_order=10):
    kind = pick(rng, ["butter", "cheby1", "cheby2", "ellip", "bessel"])
    n = int(rng.integers(1, max_order + 1))
    band = pick(rng, ["low", "high", "bandpass", "bandstop"])
    if band in ("low", "high"):
        wn = float(rng.uniform(0.005, 0.95))
    else:
        lo = float(rng.uniform(0.005, 0.9))
        wn = [lo, float(rng.uniform(lo + 0.01, 0.97))]
    extra = {"butter": [], "bessel": [], "cheby1": [float(rng.uniform(0.1, 3))], "cheby2": [float(rng.uniform(20, 80))],
             "ellip": [float(rng.uniform(0.1, 3)), float(rng.uniform(30, 80))]}[kind]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(ss, kind)(n, *extra, wn, btype=band, output=output)


def random_ba(rng):
    r = rng.random()
    if r < 0.25:
        b = [float(v) for v in rng.normal(0, 1, int(rng.integers(1, 80)))]
        a = pick(rng, [[1.0], [float(rng.uniform(0.5, 3))], float(rng.uniform(0.5, 3)), 1])
        return b, a
    b, a = design(rng, "ba", 8)
    scale = 1.0 if r < 0.8 else float(rng.uniform(0.2, 5))
    return [float(v) * scale for v in b], [float(v) * scale for v in a]


def ntaps(a, b):
    la = len(a) if isinstance(a, list) else 1
    lb = len(b) if isinstance(b, list) else 1
    return max(la, lb)


def generate(rng, n, max_len):
    cap = min(max_len, 50000)
    for _ in range(n):
        r = rng.random()
        fn = pick(rng, FUNCS)
        if r < 0.82:
            values = valid_values(rng, fn, cap)
            kind = "random"
        elif r < 0.96:
            fn, values = invalid_values(rng, cap)
            kind = "invalid"
        else:
            yield wrong_call(rng)
            continue
        args, kwargs = make_call(rng, ORDERS[fn], values, DEFAULTS[fn])
        yield Case(fn, args, kwargs, kind=kind, compare=comparator(fn, args, kwargs))


def valid_values(rng, fn, cap):
    if fn in ("lfilter", "filtfilt", "lfilter_zi"):
        b, a = random_ba(rng)
        values = {"b": b, "a": a}
    else:
        values = {"sos": design(rng, "sos", 16).tolist()}
    if fn in ("lfilter_zi", "sosfilt_zi"):
        return values
    length = max(1, random_length(rng, cap))
    x = industrial(rng, length)
    values["x"] = x
    if rng.random() < 0.1:
        values["axis"] = pick(rng, [-1, 0])
    if fn == "lfilter" and rng.random() < 0.5:
        values["zi"] = [float(v) for v in rng.normal(0, 1, ntaps(values["a"], values["b"]) - 1)]
    if fn == "sosfilt" and rng.random() < 0.5:
        values["zi"] = [[float(v) for v in rng.normal(0, 1, 2)] for _ in values["sos"]]
    if fn in ("filtfilt", "sosfiltfilt"):
        if rng.random() < 0.5:
            values["padtype"] = pick(rng, ["odd", "even", "constant", None])
        if rng.random() < 0.3:
            values["padlen"] = pick(rng, [0, 1, int(rng.integers(0, max(1, length))), length - 1, length])
        if fn == "filtfilt" and rng.random() < 0.1:
            values["method"] = "pad"
    return values


def invalid_values(rng, cap):
    what = int(rng.integers(0, 16))
    x = industrial(rng, int(rng.integers(30, 300)))
    b, a = design(rng, "ba", 4)
    b, a = b.tolist(), a.tolist()
    sos = design(rng, "sos", 6).tolist()
    if what == 0:
        return "lfilter", {"b": b, "a": [], "x": x}
    if what == 1:
        return "lfilter", {"b": [], "a": a, "x": x}
    if what == 2:
        return "lfilter", {"b": b, "a": [0.0] + a[1:], "x": x}
    if what == 3:
        return "lfilter", {"b": b, "a": a, "x": x, "zi": [0.0] * (len(a) + 1)}
    if what == 4:
        return "lfilter", {"b": [1.0, 2.0], "a": [1.0], "x": []}
    if what == 5:
        return "sosfilt", {"sos": [row[:5] for row in sos], "x": x}
    if what == 6:
        bad = [list(row) for row in sos]
        bad[0][3] = 2.0
        return "sosfilt", {"sos": bad, "x": x}
    if what == 7:
        return "sosfilt", {"sos": sos, "x": x, "zi": [[0.0, 0.0]] * (len(sos) + 1)}
    if what == 8:
        return "filtfilt", {"b": b, "a": a, "x": x, "padtype": pick(rng, ["bad", "ODD", 5])}
    if what == 9:
        return "filtfilt", {"b": b, "a": a, "x": x[:5]}
    if what == 10:
        return "filtfilt", {"b": b, "a": a, "x": x, "method": pick(rng, ["gust", "nope"])}
    if what == 11:
        return "sosfiltfilt", {"sos": sos, "x": x, "padlen": len(x)}
    if what == 12:
        return "lfilter_zi", {"b": b, "a": pick(rng, [[0.0, 1.0], [1.0, -1.0]])}
    if what == 13:
        return "sosfilt_zi", {"sos": sos[0]}
    if what == 14:
        x[int(rng.integers(0, len(x)))] = pick(rng, [None, math.nan])
        return pick(rng, ["lfilter", "filtfilt"]), {"b": b, "a": a, "x": x}
    return "lfilter", {"b": b, "a": a, "x": x[0]}


def wrong_call(rng):
    what = int(rng.integers(0, 3))
    if what == 0:
        return Case("lfilter", [[1.0], [1.0]], {}, kind="call")
    if what == 1:
        return Case("sosfiltfilt", [[[1.0, 0, 0, 1.0, 0, 0]], [1.0, 2.0, 3.0]], {"pad": 1}, kind="call")
    return Case("filtfilt", [[1.0], [1.0], [1.0, 2.0], -1, "odd", None, "pad", None, 7], {}, kind="call")


# Properties

def properties(rng, n, max_len):
    """
    filtfilt and sosfiltfilt have zero lag. lfilter_zi and sosfilt_zi, fed to scipy's own filters with a constant
    input, give a constant output: the residual must be within 10 times what scipy's own zi leaves (or 1e-9), since
    for poles near the unit circle the filter's rounding sets a floor that is not the zi's fault. A wrong zi misses
    by orders of magnitude more.
    """
    for i in range(n):
        which = i % 4
        if which in (0, 1):
            m = int(rng.integers(400, 4000))
            center = m // 2
            width = float(rng.uniform(m / 40, m / 12))
            pulse = [math.exp(-0.5 * ((k - center) / width) ** 2) for k in range(m)]
            cutoff = float(rng.uniform(0.05, 0.3))
            if which == 0:
                b, a = ss.butter(int(rng.integers(2, 6)), cutoff)
                args, fn = [b.tolist(), a.tolist(), pulse], "filtfilt"
            else:
                sos = ss.butter(int(rng.integers(2, 9)), cutoff, output="sos")
                args, fn = [sos.tolist(), pulse], "sosfiltfilt"
            yield Case(fn, args, {}, 0.0, "%s: a centred pulse keeps its peak in place (no lag)" % fn,
                       check=lambda y, c=center: 0.0 if int(np.argmax(y)) == c else math.inf)
        elif which == 2:
            # Low orders only: in 'ba' form, high orders are ill-conditioned (order 14 has coefficients in the
            # thousands), and scipy's own lfilter_zi, which this matches bit for bit, misses steady state there too.
            # sosfilt_zi below covers high orders.
            b, a = design(rng, "ba", 4)
            level = float(rng.uniform(-100, 100))

            def residual(zi, b=b, a=a, level=level):
                y = ss.lfilter(b, a, [level] * 50, zi=np.asarray(zi) * level)[0]
                gain = np.sum(b) / np.sum(a)
                return float(np.max(np.abs(y - level * gain)) / max(abs(level * gain), abs(level), 1e-300))

            own = residual(ss.lfilter_zi(b, a))
            yield Case("lfilter_zi", [b.tolist(), a.tolist()], {}, max(1e-9, 10 * own),
                       "lfilter_zi reaches steady state as well as scipy's own", check=residual)
        else:
            sos = design(rng, "sos", 12)
            level = float(rng.uniform(-100, 100))

            def residual(zi, sos=sos, level=level):
                y = ss.sosfilt(sos, [level] * 50, zi=np.asarray(zi) * level)[0]
                gain = np.prod([np.sum(r[:3]) / np.sum(r[3:]) for r in sos])
                return float(np.max(np.abs(y - level * gain)) / max(abs(level * gain), abs(level), 1e-300))

            own = residual(ss.sosfilt_zi(sos))
            yield Case("sosfilt_zi", [sos.tolist()], {}, max(1e-9, 10 * own),
                       "sosfilt_zi reaches steady state as well as scipy's own", check=residual)
