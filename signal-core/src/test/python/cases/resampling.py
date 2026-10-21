"""
Resampling: resample, resample_poly and decimate against scipy.signal, plus properties.

Comparison, by path:
    - resample_poly, and decimate with ftype 'fir': the upfirdn loop is scipy's, in scipy's order, so outputs differ
      only through firwin's coefficients (a few roundings each, from sin): within 16 eps * sum|h * up| * M, M the
      largest magnitude upfirdn sees (the extension included: 'smooth', 'line' and 'antireflect' extrapolate).
    - decimate with ftype 'iir': cheby1's sections differ from scipy's by roundings (its poles come from sinh, sin
      and cos, which each platform's maths library rounds differently), and filtering at small cutoffs magnifies
      that. As for the filtering milestone, scipy is re-run with every section coefficient and sample moved by 64
      roundings (random signs, five draws), and the result may differ by twice the spread that produces. With 32
      roundings and three draws, one Windows run reached 0.94 of the allowance where Linux reached 0.03 on the same
      cases.
    - resample: the forward FFT's bound (milestone 1) carried through the window fold, the bin selection and the
      inverse FFT, plus the inverse FFT's own rounding.

Where the module differs from scipy (the reference below applies these on top of scipy):
    - A window given as a list of numbers is the window's values (scipy needs a numpy array for that); num, q and
      n must be integers; decimate's ftype is 'iir' or 'fir' (scipy also takes a dlti system).
    - resample_poly with padtype 'reflect' or 'antireflect' on a 1-sample signal raises a ValueError: scipy's
      kernel takes a remainder modulo 2 * (len(x) - 1) there and the Python process dies ("Floating point
      exception"), so the reference raises without calling scipy.
    - Signals are 1-D and numbers finite (module-wide rules).
"""

import inspect
import math
import warnings

import numpy as np
import scipy.signal as ss

from harness import EPS, Case, Mismatch, check_signal, industrial, make_call, pick, random_length

FN = "resampling"
FUNCS = ["resample", "resample_poly", "decimate"]
SIGS = {f: inspect.signature(getattr(ss, f)) for f in FUNCS}
ORDERS = {f: list(SIGS[f].parameters) for f in FUNCS}
DEFAULTS = {f: {k: v.default for k, v in SIGS[f].parameters.items() if v.default is not inspect.Parameter.empty}
            for f in FUNCS}
MODES = ["constant", "wrap", "edge", "smooth", "symmetric", "reflect", "antisymmetric", "antireflect", "line"]
BACKGROUNDS = ["mean", "median", "minimum", "maximum"]


def finite(v):
    if isinstance(v, (list, tuple)):
        for x in v:
            finite(x)
    elif isinstance(v, float) and not math.isfinite(v):
        raise ValueError("numbers must be finite")


def window_arg(w):
    """JSON lists: one starting with a name is a spec (a tuple); one of numbers is the window's values (an array)."""
    if isinstance(w, list):
        if w and isinstance(w[0], str):
            return tuple(w)
        return np.asarray(w, dtype=float)
    return w


def make_reference(fn):
    def reference(*args, **kwargs):
        b = SIGS[fn].bind(*args, **kwargs)
        b.apply_defaults()
        a = b.arguments
        for v in a.values():
            finite(v)
        check_signal(a["x"])
        for k in ("num", "q", "n"):
            if isinstance(a.get(k), float):
                raise TypeError("%s must be an integer" % k)
        if fn == "decimate" and a["ftype"] not in ("iir", "fir"):
            raise ValueError("invalid ftype")
        if "window" in a:
            a["window"] = window_arg(a["window"])
        if fn == "resample_poly" and len(a["x"]) == 1 and a["padtype"] in ("reflect", "antireflect"):
            # Raise only where scipy would reach its crashing kernel: after its own checks pass, and unless the
            # ratio reduces to 1:1 (which returns a copy first).
            up, down = a["up"], a["down"]
            if (up == int(up) and down == int(down) and int(up) >= 1 and int(down) >= 1 and a["cval"] is None):
                g = math.gcd(int(up), int(down))
                if not (int(up) // g == 1 and int(down) // g == 1):
                    raise ValueError("padtype needs at least 2 samples (scipy crashes the interpreter on 1)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return getattr(ss, fn)(*b.args, **b.kwargs)
    return reference


REFERENCES = {f: make_reference(f) for f in FUNCS}
reference = REFERENCES["resample_poly"]


def flat(v):
    if isinstance(v, (list, tuple)):
        return [x for item in v for x in flat(item)]
    if isinstance(v, np.ndarray):
        return [complex(x) if np.iscomplexobj(v) else x.item() for x in v.ravel()]
    if isinstance(v, np.generic):
        return [v.item()]
    return [v]


def worst(got, want, tol, rel=4 * EPS):
    if len(got) != len(want):
        raise Mismatch("got %d values, expected %d" % (len(got), len(want)))
    r = 0.0
    for g, w in zip(got, want):
        if isinstance(w, int) and not isinstance(w, bool) and not isinstance(g, int):
            raise Mismatch("got %r, expected the integer %r" % (g, w))
        d = abs(g - w)
        if d:
            t = tol + rel * abs(w)
            r = max(r, d / t if t > 0 else math.inf)
    return r


def poly_bound(x, h_up, mode, cval, bg):
    """16 eps sum|h * up| * M, with M the largest magnitude upfirdn sees (extrapolating modes included)."""
    x = np.asarray(x, float)
    xs = x - bg if bg is not None else x
    m = float(np.max(np.abs(xs))) if xs.size else 0.0
    if mode in ("smooth", "line", "antireflect"):
        m += (len(h_up) + 2) * 2 * float(np.ptp(xs) if xs.size else 0.0)
    if mode == "constant" and cval:
        m = max(m, abs(cval))
    return 16 * EPS * float(np.sum(np.abs(h_up))) * m + (8 * EPS * abs(bg) if bg else 0.0)


def comparator(fn, args, kwargs):
    def cmp(got, want):
        b = SIGS[fn].bind(*args, **kwargs)
        b.apply_defaults()
        a = b.arguments
        x = np.asarray(a["x"], float)
        if fn == "resample":
            nx, num = len(x), a["num"]
            w = window_arg(a["window"])
            if w is None:
                maxw = 1.0
            elif isinstance(w, np.ndarray):
                maxw = max(1.0, float(np.max(np.abs(w))))
            else:
                maxw = max(1.0, float(np.max(np.abs(ss.get_window(w, nx)))))
            norm = float(np.linalg.norm(x))
            m2 = min(num, nx) // 2 + 1
            if a["domain"] == "time":
                e_in = 16 * EPS * math.log2(nx + 1) * norm + 8 * EPS * abs(float(np.sum(x)))
                tol = maxw * (2 * m2 * e_in / nx + 16 * EPS * math.log2(num + 1) * math.sqrt(2 * nx) * norm / nx)
            else:
                tol = 16 * EPS * math.log2(num + 1) * norm * maxw * math.sqrt(2) * num / (nx * math.sqrt(num))
            if a["t"] is not None:
                got_y, got_t = got
                want_y, want_t = want
                if flat(got_t) != flat(want_t):
                    raise Mismatch("new times differ: got %s, expected %s" % (flat(got_t)[:3], flat(want_t)[:3]))
                return worst(flat(got_y), flat(want_y), tol)
            return worst(flat(got), flat(want), tol)
        if fn == "decimate" and a["ftype"] == "iir":
            q, n = a["q"], (8 if a["n"] is None else a["n"])
            sos = ss.cheby1(n, 0.05, 0.8 / q, output="sos")
            rng = np.random.default_rng(len(x))
            spread = np.zeros(len(flat(want)))
            for _ in range(5):
                s2 = sos * (1 + 64 * EPS * rng.choice([-1.0, 1.0], sos.shape))
                s2[:, 3] = 1.0
                x2 = x * (1 + 64 * EPS * rng.choice([-1.0, 1.0], x.shape))
                y2 = (ss.sosfiltfilt(s2, x2) if a["zero_phase"] else ss.sosfilt(s2, x2))[::q]
                spread = np.maximum(spread, np.abs(y2 - np.asarray(flat(want))))
            g, wv = flat(got), flat(want)
            if len(g) != len(wv):
                raise Mismatch("got %d values, expected %d" % (len(g), len(wv)))
            scale = max([abs(v) for v in wv] + [1e-300])
            r = 0.0
            for gv, ww, s in zip(g, wv, spread):
                d = abs(gv - ww)
                if d:
                    r = max(r, d / max(2 * s, 8 * EPS * scale))
            return r
        # resample_poly, or decimate with ftype 'fir'
        if fn == "decimate":
            q = a["q"]
            n = 20 * q if a["n"] is None else a["n"]
            h_up = ss.firwin(n + 1, 1. / q, window="hamming")
            mode, cval, bg = "constant", 0, None
        else:
            up, down = int(a["up"]), int(a["down"])
            g_ = math.gcd(up, down)
            up, down = up // g_, down // g_
            w = window_arg(a["window"])
            if isinstance(w, np.ndarray):
                h_up = w * up
            elif up == down == 1:
                h_up = np.zeros(1)
            else:
                mr = max(up, down)
                h_up = ss.firwin(2 * 10 * mr + 1, 1. / mr, window=w) * up
            pt = a["padtype"]
            bg = None
            mode, cval = (pt, a["cval"] or 0) if pt in MODES else ("constant", 0)
            if pt in BACKGROUNDS and len(x):
                bg = float({"mean": np.mean, "median": np.median, "minimum": np.min, "maximum": np.max}[pt](x))
        return worst(flat(got), flat(want), poly_bound(x, h_up, mode, cval, bg))
    return cmp


# Generation

def signal(rng, cap, minimum=1):
    n = max(minimum, random_length(rng, cap))
    if rng.random() < 0.5:
        return industrial(rng, n)
    t = np.arange(n)
    f = rng.uniform(0.001, 0.2)
    return (rng.uniform(0.5, 5) * np.sin(2 * np.pi * f * t) + rng.normal(0, rng.uniform(0.01, 0.5), n)
            + rng.choice([0.0, rng.uniform(-50, 500)])).tolist()


def generate(rng, n, max_len):
    cap = min(max_len, 3000)
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
        yield Case(fn, args, kwargs, kind=kind, compare=comparator(fn, args, kwargs))


def valid_values(rng, fn, cap):
    if fn == "resample":
        x = signal(rng, cap)
        nx = len(x)
        values = {"x": x, "num": int(rng.integers(1, 2 * nx + 2))}
        if rng.random() < 0.15:
            values["num"] = nx
        if rng.random() < 0.3:
            values["window"] = pick(rng, ["hann", ["kaiser", 8.0], "hamming", 6.0,
                                          [float(v) for v in rng.uniform(0.2, 1, nx)]])
        if rng.random() < 0.15:
            values["domain"] = "freq"
        if rng.random() < 0.3:
            t0 = float(rng.uniform(-100, 100))
            values["t"] = [t0, t0 + float(rng.uniform(0.01, 10))] + ([t0 + 99.0] if rng.random() < 0.3 else [])
        return values
    if fn == "resample_poly":
        values = {"x": signal(rng, cap, 2)}
        up, down = int(rng.integers(1, 13)), int(rng.integers(1, 13))
        values["up"] = pick(rng, [up, float(up)]) if rng.random() < 0.2 else up
        values["down"] = down
        if rng.random() < 0.4:
            values["window"] = pick(rng, [("kaiser", 5.0), "hamming", "hann", 7.0, ["kaiser", 9.0],
                                          [float(v) for v in ss.firwin(int(2 * rng.integers(3, 30) + 1), 0.3)]])
        if rng.random() < 0.6:
            values["padtype"] = pick(rng, MODES + BACKGROUNDS)
        if values.get("padtype", "constant") == "constant" and rng.random() < 0.3:
            values["cval"] = float(rng.uniform(-10, 10))
        return values
    q = int(rng.integers(2, 14))
    values = {"x": signal(rng, cap, 120), "q": q}
    if rng.random() < 0.5:
        values["ftype"] = pick(rng, ["iir", "fir"])
    if rng.random() < 0.3:
        values["n"] = int(rng.integers(2, 9)) if values.get("ftype", "iir") == "iir" else int(rng.integers(4, 60))
    if rng.random() < 0.3:
        values["zero_phase"] = pick(rng, [True, False])
    return values


def invalid_values(rng):
    x = industrial(rng, int(rng.integers(40, 300)))
    what = int(rng.integers(0, 16))
    if what == 0:
        return "resample", {"x": x, "num": 50, "domain": "space"}
    if what == 1:
        return "resample", {"x": x, "num": pick(rng, [0, -5])}
    if what == 2:
        return "resample", {"x": x, "num": 50, "window": [1.0] * (len(x) + 3)}
    if what == 3:
        return "resample", {"x": x, "num": 50, "t": [1.0]}
    if what == 4:
        return "resample_poly", {"x": x, "up": pick(rng, [0, -2]), "down": 3}
    if what == 5:
        return "resample_poly", {"x": x, "up": 1.5, "down": 3}
    if what == 6:
        return "resample_poly", {"x": x, "up": 2, "down": 3, "padtype": "mean", "cval": 1.0}
    if what == 7:
        return "resample_poly", {"x": x, "up": 2, "down": 3, "padtype": pick(rng, ["zero", "Constant"])}
    if what == 8:
        return "resample_poly", {"x": [x[0]], "up": 3, "down": 2, "padtype": pick(rng, ["reflect", "antireflect"])}
    if what == 9:
        return "decimate", {"x": x, "q": 2.0}
    if what == 10:
        return "decimate", {"x": x, "q": 3, "ftype": "butter"}
    if what == 11:
        return "decimate", {"x": x, "q": pick(rng, [0, 1]), "ftype": "fir"}
    if what == 12:
        x[7] = pick(rng, [None, math.nan])
        return pick(rng, FUNCS), {"x": x, "num": 10, "up": 2, "down": 3, "q": 2}
    if what == 13:
        return "decimate", {"x": x[:20], "q": 3}
    if what == 14:
        return "resample_poly", {"x": x, "up": 2, "down": 3, "window": [[1.0, 2.0], [3.0, 4.0]]}
    return "resample", {"x": [], "num": 10}


def wrong_call(rng):
    what = int(rng.integers(0, 3))
    if what == 0:
        return Case("resample", [[1.0, 2.0, 3.0]], {}, kind="call")
    if what == 1:
        return Case("resample_poly", [[1.0, 2.0, 3.0], 2], {}, kind="call")
    return Case("decimate", [[1.0] * 100, 2], {"order": 4}, kind="call")


# Properties

def properties(rng, n, max_len):
    """
    resample_poly keeps a constant signal constant (up = 1, padtype 'edge') and returns ceil(n * up / down)
    samples; resample reconstructs a periodic tone exactly at any rate; decimate keeps a slow tone (to its ripple).
    """
    for i in range(n):
        which = i % 4
        if which == 0:
            # With up = 1 there is one filter phase, whose taps firwin scales to sum to 1, so a constant (padded
            # with its edge value) stays constant to rounding. With up > 1 each phase's sum is only close to 1.
            level = float(rng.uniform(-500, 500))
            m = int(rng.integers(5, 500))
            down = int(rng.integers(2, 9))
            yield Case("resample_poly", [[level] * m, 1, down], {"padtype": "edge"}, 1e-12 * abs(level) + 1e-12,
                       "a constant stays constant (up = 1)",
                       check=lambda r, level=level: max(abs(v - level) for v in r))
        elif which == 1:
            m = int(rng.integers(5, 2000))
            up, down = int(rng.integers(1, 13)), int(rng.integers(1, 13))
            g = math.gcd(up, down)
            expected = -(-m * (up // g) // (down // g))
            yield Case("resample_poly", [industrial(rng, m), up, down], {}, 0.0,
                       "returns ceil(n * up / down) samples",
                       check=lambda r, e=expected: 0.0 if len(r) == e else math.inf)
        elif which == 2:
            m = int(rng.integers(16, 1024))
            k = int(rng.integers(1, m // 2 - 1))
            num = int(rng.integers(2 * k + 2, 3 * m))
            x = [math.sin(2 * math.pi * k * j / m) for j in range(m)]
            target = [math.sin(2 * math.pi * k * j / num) for j in range(num)]
            yield Case("resample", [x, num], {}, 1e-10, "a periodic tone is reconstructed at any rate",
                       check=lambda r, t=target: max(abs(a - b) for a, b in zip(r, t)))
        else:
            q = int(rng.integers(2, 8))
            m = int(rng.integers(max(600, 150 * q), 3000))
            f = 0.02 / q
            x = [math.sin(2 * math.pi * f * j) for j in range(m)]
            # cheby1's 0.05 dB passband ripple, applied twice by the zero-phase filter: up to about 1.2%
            yield Case("decimate", [x, q], {}, 0.02, "a slow tone survives decimation",
                       check=lambda r, x=x, q=q: max(abs(r[j] - x[j * q]) for j in range(50, len(r) - 50)))
