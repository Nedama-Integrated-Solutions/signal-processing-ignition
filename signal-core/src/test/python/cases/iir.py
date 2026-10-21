"""
Filter design: butter, cheby1, cheby2, ellip and bessel against scipy.signal, plus properties of the designs.

Where the module differs from scipy (the reference below applies these on top of scipy):
    - Numbers must be finite (module-wide rule).
    - Orders above 500 (Bessel: 100) are refused. Bessel orders from about 90 to 100, where scipy's own root
      finding fails, work here; the tests stay within scipy's range.

Tolerances, from measurement (worst over 1500 random designs, orders up to 25):
    - zeros, poles and gain ('zpk'): within 250 eps of each value; the allowance is 1024 eps.
    - second-order sections ('sos'): within 41 eps of each row's size; the allowance is 256 eps.
    - transfer function ('ba'): expanding many poles into polynomial coefficients is ill-conditioned, for scipy
      as well (an order-46 band-stop design's coefficients are off by 2e-6 in scipy and 3e-6 here, against an
      exact 50-digit expansion). The allowance is the standard running-error bound of that expansion, the
      coefficients of the polynomial whose roots are the poles' (or zeros') absolute values, times 1024 eps,
      the same relative allowance as the gain and poles that feed it. It is tight at low orders and loose only
      where the 'ba' form has lost its meaning anyway.
    - Bessel 'mag' designs allow 1e-10 relative: scipy finds the -3 dB frequency with a secant method stopping
      at 1.48e-8, and near convergence a one-rounding difference in the function value moves the last step
      (measured up to 1700 eps at order 21, on either side). 1e-10 is still 150 times tighter than scipy's own
      accuracy there.
    - Some designs are ill-conditioned: scipy's own output moves by thousands of eps when its inputs move by a
      few roundings (an order-11 elliptic low-pass with 24 dB attenuation has stopband zeros 1.3e-4 apart, and
      scipy on Windows and on Linux differ by 1450 eps there). When a value fails the allowance above, the test
      re-runs scipy with each input moved by 4 roundings and allows twice the movement that causes.
    - Section order: zpk2sos takes poles in order of distance from the unit circle. When two poles tie to within
      64 eps (in degenerate designs every pole can sit on the circle), a one-ulp difference decides the order, so
      the same sections in another order are accepted then, and only then.
"""

import inspect
import math
import warnings

import numpy as np
import scipy.signal as ss

from harness import EPS, Case, Mismatch, make_call, pick

FN = "iir"
FUNCS = ["butter", "cheby1", "cheby2", "ellip", "bessel"]
SIGS = {f: inspect.signature(getattr(ss, f)) for f in FUNCS}
ORDERS = {f: list(SIGS[f].parameters) for f in FUNCS}
DEFAULTS = {f: {k: v.default for k, v in SIGS[f].parameters.items() if v.default is not inspect.Parameter.empty}
            for f in FUNCS}
BAND_NAMES = {"lowpass": ["low", "lowpass", "l", "lp", "LowPass"], "highpass": ["high", "highpass", "h", "hp", "HIGH"],
              "bandpass": ["band", "bandpass", "pass", "bp"], "bandstop": ["stop", "bandstop", "bands", "bs"]}


def finite(v):
    values = v if isinstance(v, (list, tuple)) else [v]
    for x in values:
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError("numbers must be finite")


def make_reference(fn):
    def reference(*args, **kwargs):
        bound = SIGS[fn].bind(*args, **kwargs)
        for v in bound.arguments.values():
            finite(v)
        n = bound.arguments.get("N")
        if isinstance(n, (int, float)) and not isinstance(n, bool) and n > (100 if fn == "bessel" else 500):
            raise ValueError("order too large")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return getattr(ss, fn)(*bound.args, **bound.kwargs)
    return reference


REFERENCES = {f: make_reference(f) for f in FUNCS}
reference = REFERENCES["butter"]


# Comparisons

def ratio_array(got, want, bound, path, slack=None):
    if len(got) != len(want):
        raise Mismatch("%s: got %d values, expected %d" % (path, len(got), len(want)))
    worst = 0.0
    for i, (g, w, b) in enumerate(zip(got, want, bound)):
        if isinstance(w, complex) != isinstance(g, complex):
            raise Mismatch("%s[%d]: got %r, expected %r (real and complex differ)" % (path, i, g, w))
        if slack is not None:
            b = max(b, slack[i])
        d = abs(g - w)
        if b == 0:
            if d != 0:
                return math.inf
            continue
        worst = max(worst, d / b)
    return worst


def abs_poly(roots):
    """Coefficients of the polynomial whose roots are -|r|: an upper bound on every intermediate of the expansion."""
    return np.abs(np.poly(-np.abs(np.asarray(roots, complex)))) if len(roots) else np.array([1.0])


def compare_ba(zpk, rel):
    z, p, k = zpk

    def cmp(got, want, slack=None):
        (b, a), (bw, aw) = got, want
        bb = rel * abs(k) * abs_poly(z) + 1e-300
        ab = rel * abs_poly(p) + 1e-300
        sb, sa = (None, None) if slack is None else slack
        return max(ratio_array(b, bw, bb, "b", sb), ratio_array(a, aw, ab, "a", sa))
    return cmp


def compare_zpk(rel):
    def cmp(got, want, slack=None):
        (z, p, k), (zw, pw, kw) = got, want
        sz, sp, sk = (None, None, 0.0) if slack is None else slack
        worst = 0.0
        for g, w, name, sl in ((z, zw, "z", sz), (p, pw, "p", sp)):
            scale = max([abs(v) for v in w] + [1e-300])
            bound = [rel * max(abs(v), 1e-12 * scale) for v in w]
            worst = max(worst, ratio_array(g, w, bound, name, sl))
        if math.isnan(kw) or math.isnan(k):
            if not (math.isnan(kw) and math.isnan(k)):
                raise Mismatch("k: got %r, expected %r" % (k, kw))
            return worst
        return max(worst, abs(k - kw) / max(rel * abs(kw) + 1e-300, sk))
    return cmp


def compare_sos(rel):
    def cmp(got, want, slack=None):
        if len(got) != len(want):
            raise Mismatch("sos: got %d sections, expected %d" % (len(got), len(want)))
        worst = 0.0
        for i, (g, w) in enumerate(zip(got, want)):
            for lo, hi, part in ((0, 3, "b"), (3, 6, "a")):
                scale = sum(abs(v) for v in w[lo:hi])
                sl = None if slack is None else slack[i][lo:hi]
                worst = max(worst, ratio_array(g[lo:hi], w[lo:hi], [rel * scale] * 3, "sos[%d] %s" % (i, part), sl))
        return worst
    return cmp


def flatten(v):
    if isinstance(v, (list, tuple)):
        return [x for item in v for x in flatten(item)]
    return [v]


def rebuild(shape, values):
    """values (a flat list) back into the nesting of shape."""
    it = iter(values)

    def build(s):
        if isinstance(s, (list, tuple)):
            return [build(x) for x in s]
        return next(it)
    return build(shape)


def perturbed(fn, args, kwargs):
    """
    scipy's design with each real input (rp, rs, each edge of Wn, fs) moved by 4 roundings either way, one at a
    time. How far the output moves is how far rounding differences between platforms (each operating system's
    maths library rounds sin, exp and the elliptic functions slightly differently) can legitimately move it.
    """
    bound = SIGS[fn].bind(*args, **kwargs)
    out = []
    for name in ("rp", "rs", "Wn", "fs"):
        value = bound.arguments.get(name)
        if value is None:
            continue
        targets = [None] if not isinstance(value, list) else list(range(len(value)))
        for t in targets:
            for sign in (1, -1):
                f = 1 + sign * 4 * EPS
                a = dict(bound.arguments)
                if t is None:
                    a[name] = value * f
                else:
                    a[name] = [v * f if i == t else v for i, v in enumerate(value)]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        out.append(getattr(ss, fn)(**a))
                except Exception:
                    pass
    return out


def tied_poles(fn, args, kwargs):
    """
    True when zpk2sos's section order is ambiguous: it takes poles in order of their distance from the unit circle
    (digital) or from the imaginary axis (analog), and two poles that are not a conjugate pair are within 64 eps
    of the same distance, so a one-rounding difference (numpy's and Java's hypot can differ by one ulp) decides it.
    """
    bound = SIGS[fn].bind(*args, **kwargs)
    a = dict(bound.arguments)
    a["output"] = "zpk"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z, p, k = getattr(ss, fn)(**a)
    except Exception:
        return False
    analog = bool(a.get("analog", False))
    d = [abs(v.real) if analog else abs(1 - abs(v)) for v in p if v.imag >= 0]
    return any(abs(d[i] - d[j]) <= 64 * EPS for i in range(len(d)) for j in range(i + 1, len(d)))


def permuted_sos(base, got, want):
    """The comparison with Java's sections matched to scipy's in whichever order fits best."""
    remaining = list(range(len(got)))
    matched = []
    for w in want:
        best = min(remaining, key=lambda i: sum(abs(x - y) for x, y in zip(got[i], w)), default=None)
        if best is None:
            return math.inf
        remaining.remove(best)
        matched.append(got[best])
    return base(matched, want)


def conditioned(fn, args, kwargs, base, sos=False):
    """
    base comparison first; if it fails, allow twice the movement scipy itself shows under 4-rounding input changes;
    and for second-order sections whose order is genuinely ambiguous (see tied_poles), accept the same sections in
    another order.
    """
    def cmp(got, want):
        ratio = base(got, want)
        if ratio <= 1.0:
            return ratio
        shape = want
        flat_want = flatten(shape)
        moves = [0.0] * len(flat_want)
        for ref in perturbed(fn, args, kwargs):
            flat_ref = flatten(plain_ref(ref))
            if len(flat_ref) != len(flat_want):
                continue
            moves = [max(m, abs(r - w)) for m, r, w in zip(moves, flat_ref, flat_want)]
        slack = rebuild(shape, [2 * m for m in moves])
        ratio = base(got, want, slack)
        if ratio > 1.0 and sos and len(got) == len(want) and tied_poles(fn, args, kwargs):
            ratio = min(ratio, permuted_sos(base, got, want))
        return ratio
    return cmp


def plain_ref(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [plain_ref(x) for x in v]
    return v


# Generation

def random_design(rng):
    fn = pick(rng, FUNCS)
    r = rng.random()
    n = 0 if r < 0.02 else int(rng.integers(1, 13)) if r < 0.87 else int(rng.integers(13, 26))
    band = pick(rng, sorted(BAND_NAMES))
    analog = rng.random() < 0.1
    fs = None
    if band in ("lowpass", "highpass"):
        wn = float(rng.uniform(0.01, 0.97)) if not analog else float(10 ** rng.uniform(-2, 3))
    else:
        lo = float(rng.uniform(0.01, 0.9))
        hi = float(rng.uniform(lo + 0.01, 0.98))
        wn = [lo, hi] if not analog else [lo * 100, hi * 100]
    if not analog and rng.random() < 0.3:
        fs = float(rng.uniform(1, 10000))
        wn = [v * fs / 2 for v in wn] if isinstance(wn, list) else wn * fs / 2
    values = {"N": n, "Wn": wn}
    if fn in ("cheby1", "ellip"):
        values["rp"] = float(rng.uniform(0.05, 5))
    if fn in ("cheby2", "ellip"):
        values["rs"] = float(rng.uniform(15, 120))
    if rng.random() < 0.8:
        values["btype"] = pick(rng, BAND_NAMES[band])
    elif band != "lowpass":
        values["btype"] = band
    if analog:
        values["analog"] = True
    elif rng.random() < 0.1:
        values["analog"] = False
    if rng.random() < 0.7:
        values["output"] = pick(rng, ["ba", "zpk", "sos", "SOS"])
    if fs is not None:
        values["fs"] = fs
    if fn == "bessel" and rng.random() < 0.7:
        values["norm"] = pick(rng, ["phase", "delay", "mag", "MAG"])
    return fn, values


def comparator(fn, args, kwargs):
    """
    Chooses the comparison from the output form, sizing the 'ba' allowance from scipy's own zeros and poles, and
    wraps it so an ill-conditioned design gets the slack its own sensitivity justifies. Bessel 'mag' designs
    allow 1e-10 relative: scipy solves their -3 dB frequency only to 1.48e-8, so differences below that are
    noise in scipy's answer, and 1e-10 still checks 150 times tighter than scipy's own accuracy.
    """
    bound = SIGS[fn].bind(*args, **kwargs)
    output = str(bound.arguments.get("output", "ba")).lower()
    mag = fn == "bessel" and str(bound.arguments.get("norm", "phase")).lower() == "mag"
    if output == "zpk":
        return conditioned(fn, args, kwargs, compare_zpk(1e-10 if mag else 1024 * EPS))
    if output == "sos":
        return conditioned(fn, args, kwargs, compare_sos(1e-10 if mag else 256 * EPS), sos=True)
    try:
        kw = dict(bound.arguments)
        kw["output"] = "zpk"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zpk = getattr(ss, fn)(**kw)
    except Exception:
        return conditioned(fn, args, kwargs, compare_sos(256 * EPS), sos=True)
    return conditioned(fn, args, kwargs, compare_ba(zpk, 1e-10 if mag else 1024 * EPS))


def degenerate():
    """
    A fixed sweep of degenerate elliptic designs, run every time: rs below or equal to rp makes the design NaN,
    which must fail exactly where scipy fails (in zpk2sos for 'sos', nowhere for 'zpk' and 'ba'), never with a
    Java exception. A run on Windows found one such leak; this keeps the corner covered.
    """
    import itertools
    for (rp, rs), n, band, analog, output in itertools.product([(1.35, 0.5), (1.0, 1.0)], [1, 3, 4, 7],
                                                              ["low", "high", "bandpass", "bandstop"],
                                                              [False, True], ["ba", "zpk", "sos"]):
        wn = (0.3 if band in ("low", "high") else [0.2, 0.6]) if not analog else (3.0 if band in ("low", "high")
                                                                                 else [2.0, 6.0])
        args = [n, rp, rs, wn]
        kwargs = {"btype": band, "analog": analog, "output": output}
        yield Case("ellip", args, kwargs, kind="degenerate", compare=comparator("ellip", args, kwargs))


def generate(rng, n, max_len):
    yield from degenerate()
    for _ in range(n):
        r = rng.random()
        if r < 0.82:
            fn, values = random_design(rng)
        elif r < 0.96:
            fn, values = invalid_design(rng)
        else:
            yield wrong_call(rng)
            continue
        args, kwargs = make_call(rng, ORDERS[fn], values, DEFAULTS[fn])
        yield Case(fn, args, kwargs, kind="random" if r < 0.82 else "invalid", compare=comparator(fn, args, kwargs))


def invalid_design(rng):
    fn, values = random_design(rng)
    what = int(rng.integers(0, 17))
    if what == 0:
        values["N"] = -1
    elif what == 1:
        values["N"] = 4.5
    elif what == 2:
        values["N"] = 4.0
    elif what == 3:
        values["Wn"] = 0.0
    elif what == 4:
        values.pop("fs", None)
        values.pop("analog", None)
        values["Wn"] = 1.2
    elif what == 5:
        values["Wn"] = [0.5, 0.2]
    elif what == 6:
        values["btype"] = "low"
        values["Wn"] = [0.1, 0.2]
    elif what == 7:
        values["btype"] = "bandpass"
        values["Wn"] = 0.3
        values.pop("fs", None)
    elif what == 8:
        values["btype"] = "side"
    elif what == 9:
        values["output"] = "tf"
    elif what == 10 and "rp" in values:
        values["rp"] = pick(rng, [-1.0, 0.0])
    elif what == 11 and "rs" in values:
        values["rs"] = pick(rng, [-2.0, 0.0, 0.5])
    elif what == 12:
        values["analog"] = True
        values["fs"] = 100.0
    elif what == 13:
        values.pop("analog", None)
        values["fs"] = pick(rng, [0.0, -10.0])
    elif what == 14 and fn == "bessel":
        values["norm"] = "nope"
    elif what == 15:
        values["btype"] = "bandpass"
        values.pop("fs", None)
        values.pop("analog", None)
        values["Wn"] = [0.1, 0.3, 0.5]
    else:
        values["N"] = "4"
    return fn, values


def wrong_call(rng):
    what = int(rng.integers(0, 3))
    if what == 0:
        return Case("butter", [4], {}, kind="call")
    if what == 1:
        return Case("cheby1", [4, 0.2], {}, kind="call")
    return Case("ellip", [4, 1.0, 40.0, 0.2], {"ripple": 1.0}, kind="call")


# Properties

def sos_response(sos, w):
    z = complex(math.cos(w), -math.sin(w))
    h = 1.0
    for b0, b1, b2, a0, a1, a2 in sos:
        h *= (b0 + b1 * z + b2 * z * z) / (a0 + a1 * z + a2 * z * z)
    return h


def properties(rng, n, max_len):
    """Each design meets its defining specification at the edge frequency, and digital poles are stable."""
    for i in range(n):
        fn = FUNCS[i % 5]
        order = int(rng.integers(1, 11))
        wn = float(rng.uniform(0.05, 0.9))
        args = [order]
        target = 1 / math.sqrt(2)
        if fn == "cheby1":
            rp = float(rng.uniform(0.1, 3))
            args.append(rp)
            target = 10 ** (-rp / 20)
        elif fn == "cheby2":
            rs = float(rng.uniform(20, 80))
            args.append(rs)
            target = 10 ** (-rs / 20)
        elif fn == "ellip":
            rp, rs = float(rng.uniform(0.1, 3)), float(rng.uniform(30, 80))
            args += [rp, rs]
            target = 10 ** (-rp / 20)
        args.append(wn)
        kwargs = {"output": "sos", "btype": pick(rng, ["low", "high"])}
        if fn == "bessel":
            kwargs["norm"] = "mag"
        label = "%s: |H| at Wn is the design value" % fn
        yield Case(fn, args, kwargs, 1e-9, label,
                   check=lambda r, wn=wn, t=target: abs(abs(sos_response(r, math.pi * wn)) - t))
        stable_kwargs = {"output": "zpk", "btype": pick(rng, ["low", "high", "bandpass", "bandstop"])}
        edge = min(wn, 0.85)
        stable_args = args[:-1] + ([wn] if stable_kwargs["btype"] in ("low", "high") else [[edge, edge + 0.1]])
        yield Case(fn, stable_args, stable_kwargs, 1.0, "%s: digital poles inside the unit circle" % fn,
                   check=lambda r: 0.0 if max([abs(p) for p in r[1]] + [0.0]) < 1 else math.inf)
