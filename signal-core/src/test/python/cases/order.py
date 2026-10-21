"""
Order selection: buttord, cheb1ord, cheb2ord and ellipord against scipy.signal, plus properties.

Where the module differs from scipy: numbers must be finite (module-wide rule).

Comparison. N must match exactly; Wn within 64 eps of its size. Measured over 4000 random specifications: every
N matched, and Wn agreed within 80 eps, except where scipy's own answer is ill-conditioned. Band-stop designs
find their passband edges with scipy's bounded minimiser (fminbound, xtol 1e-5); near the optimum a one-rounding
difference in the objective changes a parabolic step, so the edges can differ by about 1e-7 relative, and scipy's
own result moves just as much when its inputs move by 4 roundings. So, as for the filter designs, a value outside
the allowance is re-checked against scipy run with each input moved by 4 roundings either way: Java's N must equal
scipy's N for one of those inputs, and Wn may differ by up to twice the movement scipy shows.
"""

import inspect
import math
import warnings

import numpy as np
import scipy.signal as ss

from harness import EPS, Case, Mismatch, make_call, pick

FN = "order"
FUNCS = ["buttord", "cheb1ord", "cheb2ord", "ellipord"]
SIG = inspect.signature(ss.buttord)
ORDER = list(SIG.parameters)
DEFAULTS = {k: v.default for k, v in SIG.parameters.items() if v.default is not inspect.Parameter.empty}
DESIGN = {"buttord": "butter", "cheb1ord": "cheby1", "cheb2ord": "cheby2", "ellipord": "ellip"}


def finite(v):
    values = v if isinstance(v, (list, tuple)) else [v]
    for x in values:
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError("numbers must be finite")


def make_reference(fn):
    def reference(*args, **kwargs):
        bound = SIG.bind(*args, **kwargs)
        for v in bound.arguments.values():
            finite(v)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return getattr(ss, fn)(*bound.args, **bound.kwargs)
    return reference


REFERENCES = {f: make_reference(f) for f in FUNCS}
reference = REFERENCES["buttord"]


def as_list(w):
    return list(w) if isinstance(w, (list, tuple)) else [w]


def perturbed(fn, args, kwargs):
    bound = SIG.bind(*args, **kwargs)
    out = []
    for name in ("wp", "ws", "gpass", "gstop", "fs"):
        value = bound.arguments.get(name)
        if value is None or isinstance(value, bool):
            continue
        targets = list(range(len(value))) if isinstance(value, list) else [None]
        for t in targets:
            for sign in (1, -1):
                f = 1 + sign * 4 * EPS
                a = dict(bound.arguments)
                a[name] = value * f if t is None else [v * f if i == t else v for i, v in enumerate(value)]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        n, wn = getattr(ss, fn)(**a)
                    out.append((int(n), [float(v) for v in np.atleast_1d(wn)]))
                except Exception:
                    pass
    return out


def comparator(fn, args, kwargs):
    def cmp(got, want):
        n, wn = got
        nw, ww = want
        if isinstance(wn, list) != isinstance(ww, list):
            raise Mismatch("Wn: got %r, expected %r" % (wn, ww))
        g, w = as_list(wn), as_list(ww)
        if len(g) != len(w):
            raise Mismatch("Wn: got %d values, expected %d" % (len(g), len(w)))
        if n == nw:
            ratio = max([abs(a - b) / (64 * EPS * abs(b) + 1e-300) for a, b in zip(g, w)] + [0.0])
            if ratio <= 1.0:
                return ratio
        candidates = [(nw, w)] + perturbed(fn, args, kwargs)
        same = [c for c in candidates if c[0] == n and len(c[1]) == len(g)]
        if not same:
            raise Mismatch("N: got %r, expected %r (and no 4-rounding change of the inputs gives %r)" % (n, nw, n))
        base = same[0][1]
        envelope = [max(abs(c[1][i] - base[i]) for c in same) for i in range(len(base))]
        return max(abs(a - b) / max(64 * EPS * abs(b), 2 * e, 1e-300) for a, b, e in zip(g, base, envelope))
    return cmp


def random_spec(rng):
    fn = pick(rng, FUNCS)
    band = pick(rng, ["low", "high", "pass", "stop"])
    if band in ("low", "high"):
        a, b = sorted(float(v) for v in rng.uniform(0.02, 0.97, 2))
        b = max(b, a + 0.01)
        wp, ws = (a, b) if band == "low" else (b, a)
    else:
        x = sorted(float(v) for v in rng.uniform(0.02, 0.97, 4))
        for i in range(1, 4):
            x[i] = max(x[i], x[i - 1] + 0.005)
        ws, wp = ([x[0], x[3]], [x[1], x[2]]) if band == "pass" else ([x[1], x[2]], [x[0], x[3]])
    values = {"wp": wp, "ws": ws, "gpass": float(rng.uniform(0.1, 3)), "gstop": float(rng.uniform(10, 100))}
    r = rng.random()
    if r < 0.15:
        values["analog"] = True
        values["wp"] = [v * 100 for v in wp] if isinstance(wp, list) else wp * 100
        values["ws"] = [v * 100 for v in ws] if isinstance(ws, list) else ws * 100
    elif r < 0.4:
        fs = float(rng.uniform(1, 10000))
        values["fs"] = fs
        values["wp"] = [v * fs / 2 for v in wp] if isinstance(wp, list) else wp * fs / 2
        values["ws"] = [v * fs / 2 for v in ws] if isinstance(ws, list) else ws * fs / 2
    elif r < 0.45:
        values["analog"] = False
    return fn, values


def generate(rng, n, max_len):
    for _ in range(n):
        r = rng.random()
        if r < 0.82:
            fn, values = random_spec(rng)
            kind = "random"
        elif r < 0.96:
            fn, values = invalid_spec(rng)
            kind = "invalid"
        else:
            yield wrong_call(rng)
            continue
        args, kwargs = make_call(rng, ORDER, values, DEFAULTS)
        yield Case(fn, args, kwargs, kind=kind, compare=comparator(fn, args, kwargs))


def invalid_spec(rng):
    fn, values = random_spec(rng)
    what = int(rng.integers(0, 11))
    if what == 0:
        values["gpass"] = pick(rng, [0.0, -1.0])
    elif what == 1:
        values["gstop"] = pick(rng, [0.0, -5.0])
    elif what == 2:
        values["gpass"], values["gstop"] = 40.0, 3.0
    elif what == 3:
        values["analog"] = True
        values["fs"] = 100.0
    elif what == 4:
        values["wp"] = []
    elif what == 5:
        values.pop("fs", None)
        values.pop("analog", None)
        values["wp"], values["ws"] = [0.1, 0.2, 0.3], [0.15, 0.25]
    elif what == 6:
        values.pop("fs", None)
        values.pop("analog", None)
        values["wp"], values["ws"] = [0.1, 0.5], [0.2, 0.6]
    elif what == 7:
        values.pop("fs", None)
        values.pop("analog", None)
        values["wp"], values["ws"] = [0.1, 0.6], 0.3
    elif what == 8:
        values["gpass"] = values["gstop"]
    elif what == 9:
        values.pop("fs", None)
        values.pop("analog", None)
        values["wp"], values["ws"] = 0.3, 0.3
    else:
        values["gpass"] = 1e-300
    return fn, values


def wrong_call(rng):
    what = int(rng.integers(0, 3))
    if what == 0:
        return Case("buttord", [0.2, 0.3, 3], {}, kind="call")
    if what == 1:
        return Case("cheb1ord", [0.2, 0.3, 3, 40, False, None, 1], {}, kind="call")
    return Case("ellipord", [0.2, 0.3, 3, 40], {"g_stop": 40}, kind="call")


def response_db(design, n, wn, freqs, gpass, gstop, btype):
    extra = {"butter": [], "cheby1": [gpass], "cheby2": [gstop], "ellip": [gpass, gstop]}[design]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sos = getattr(ss, design)(n, *extra, wn, btype=btype, output="sos")
        _, h = ss.freqz_sos(sos, worN=np.asarray(freqs, float) * np.pi)
    return 20 * np.log10(np.maximum(np.abs(h), 1e-300))


def properties(rng, n, max_len):
    """A filter designed from the returned (N, Wn) meets the specification: at most gpass dB of loss at the
    passband edges and at least gstop dB at the stopband edges (digital, normalised frequencies)."""
    for _ in range(n):
        fn, values = random_spec(rng)
        values.pop("fs", None)
        values.pop("analog", None)
        wp, ws = values["wp"], values["ws"]
        if any(v >= 1 for v in as_list(wp) + as_list(ws)):
            continue
        band = {(False, True): "lowpass", (False, False): "highpass"}.get(
            (isinstance(wp, list), as_list(wp)[0] < as_list(ws)[0]))
        if band is None:
            band = "bandpass" if wp[0] > ws[0] else "bandstop"
        gpass, gstop = values["gpass"], values["gstop"]

        def meets(r, fn=fn, wp=wp, ws=ws, gpass=gpass, gstop=gstop, band=band):
            order, wn = r
            if order < 1 or order > 60:
                return 0.0
            p = response_db(DESIGN[fn], order, wn, as_list(wp), gpass, gstop, band)
            s = response_db(DESIGN[fn], order, wn, as_list(ws), gpass, gstop, band)
            return max(0.0, float(np.max(-gpass - p)), float(np.max(s + gstop)))

        yield Case(fn, [wp, ws, gpass, gstop], {}, 1e-6, "design from (N, Wn) meets the specification", check=meets)
