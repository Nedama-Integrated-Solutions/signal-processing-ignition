"""
Spectral analysis: welch, periodogram, spectrogram and hilbert against scipy.signal, plus properties.

Segmenting, detrending, window scaling, doubling and averaging follow scipy operation by operation, so frequencies
and times must match exactly. The transforms are this module's FFT, which agrees with scipy.fft to within a few
roundings (milestone 1), so spectra are compared against a bound derived from that:

    E = 16 eps log2(nfft + 1) ||segment * window|| + 8 eps |sum(segment * window)| + 8 eps max|segment| sum|window|

per segment (the FFT's error, numpy's Bluestein smear of a large mean, and detrend's rounding, which is relative
to the raw samples: removing the mean of a constant 191.189 leaves rounding of order eps * 191, not of the
residual's size). A spectrum value
d * s * |X|^2 (s the scaling, d = 2 for doubled one-sided bins) may then differ by d * s * (2 |X| E + E^2), with |X|
taken from scipy's own result; averaging keeps the bound (for the median, |X| is bounded through the mean). For
mode 'complex' and 'magnitude' the bound is sqrt(s) * E; for 'angle' it is sqrt(s) * E / |X| radians, compared
around the circle, and 'phase' (unwrapped) is compared modulo 2 pi, since unwrapping adds whole turns wherever a
bin's angle is decided by rounding (a bin near zero, such as DC after detrending, has an arbitrary angle); the
phase must also be a valid unwrapping, with no step over pi between neighbouring frequencies.

Where the module differs from scipy (the reference below applies these on top of scipy):
    - nperseg, noverlap, nfft and N must be integers; detrend must be 'constant', 'linear' or off (scipy also
      takes a function).
    - welch and spectrogram accept plain lists. In scipy 1.18.1 both fail on a list with AttributeError:
      spectrogram reads x.shape before converting x, and welch (through csd) converts x but keeps the original
      list as y, then reads y.shape. The reference converts to a numpy array first.
    - Signals are 1-D and numbers finite (module-wide rules).
"""

import inspect
import math
import warnings

import numpy as np
import scipy.signal as ss

from harness import EPS, Case, Mismatch, check_signal, industrial, make_call, pick, random_length

FN = "spectral"
FUNCS = ["welch", "periodogram", "spectrogram", "hilbert"]
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


def as_spec(window):
    """A list starting with a name is a window spec (a tuple in scipy); JSON cannot tell lists from tuples."""
    if isinstance(window, list) and window and isinstance(window[0], str):
        return tuple(window)
    return window


def make_reference(fn):
    def reference(*args, **kwargs):
        b = SIGS[fn].bind(*args, **kwargs)
        for k, v in b.arguments.items():
            finite(v)
        check_signal(b.arguments["x"])
        for k in ("nperseg", "noverlap", "nfft", "N"):
            if isinstance(b.arguments.get(k), float):
                raise TypeError("%s must be an integer" % k)
        if "window" in b.arguments:
            b.arguments["window"] = as_spec(b.arguments["window"])
        if fn in ("spectrogram", "welch"):
            # scipy 1.18.1 needs a numpy array here: spectrogram reads x.shape before converting, and welch
            # (through csd) converts x but keeps the original list as y, then reads y.shape.
            b.arguments["x"] = np.asarray(b.arguments["x"], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return getattr(ss, fn)(*b.args, **b.kwargs)
    return reference


REFERENCES = {f: make_reference(f) for f in FUNCS}
reference = REFERENCES["welch"]


# Bounds

def segments_and_window(fn, b):
    """scipy's segments (detrended) and window for this call, and its parameters."""
    a = dict(DEFAULTS[fn])
    a.update(b.arguments)
    x = np.asarray(a["x"], float)
    window = as_spec(a.get("window"))
    n = len(x)
    if fn == "periodogram":
        if window is None:
            window = "boxcar"
        nfft = a["nfft"]
        if nfft is None:
            nperseg = n
        elif nfft == n:
            nperseg = nfft
        elif nfft > n:
            nperseg = n
        else:
            x = x[:nfft]
            nperseg, nfft = nfft, None
        noverlap = 0
    else:
        nperseg, noverlap, nfft = a.get("nperseg"), a.get("noverlap"), a.get("nfft")
    if isinstance(window, (str, tuple)):
        nperseg = min(256 if nperseg is None else nperseg, n)
        win = ss.get_window(window, nperseg)
    else:
        win = np.asarray(window, float)
        nperseg = len(win)
    if noverlap is None:
        noverlap = nperseg // 8 if fn == "spectrogram" else nperseg // 2
    nfft = nperseg if nfft is None else nfft
    step = nperseg - noverlap
    count = (len(x) - noverlap) // step if fn != "spectrogram" else (len(x) - nperseg) // step + 1
    detrend = a.get("detrend")
    segs = []
    for p in range(count):
        raw = x[p * step:p * step + nperseg]
        s = ss.detrend(raw, type=detrend) if isinstance(detrend, str) and detrend else raw
        segs.append((raw, s))
    return segs, win, nfft


def fft_error(segs, win, nfft):
    """The bound E above, for the worst segment. detrend's rounding is relative to the raw samples."""
    e = 0.0
    for raw, s in segs:
        sw = s * win
        e = max(e, 16 * EPS * math.log2(nfft + 1) * float(np.linalg.norm(sw)) + 8 * EPS * abs(float(np.sum(sw)))
                + 8 * EPS * float(np.max(np.abs(raw))) * float(np.sum(np.abs(win))))
    return e


def flat(v):
    if isinstance(v, (list, tuple)):
        return [x for item in v for x in flat(item)]
    if isinstance(v, np.ndarray):
        return [complex(x) if np.iscomplexobj(v) else float(x) for x in v.ravel()]
    return [v]


def same_values(got, want, what):
    if len(got) != len(want):
        raise Mismatch("%s: got %d values, expected %d" % (what, len(got), len(want)))
    for g, w in zip(got, want):
        if abs(g - w) > 2 * math.ulp(abs(w)):
            raise Mismatch("%s: got %r, expected %r" % (what, g, w))


def worst(got, want, tol):
    r = 0.0
    for g, w, t in zip(got, want, tol):
        d = abs(g - w)
        if d:
            r = max(r, d / t if t > 0 else math.inf)
    return r


def with_argument(fn, args, kwargs, name, value):
    """The reference's result for this call with one argument changed, however the call passed it."""
    b = SIGS[fn].bind(*args, **kwargs)
    b.arguments[name] = value
    return REFERENCES[fn](*b.args, **b.kwargs)


def comparator(fn, args, kwargs):
    def cmp(got, want):
        b = SIGS[fn].bind(*args, **kwargs)
        if fn == "hilbert":
            x = np.asarray(b.arguments["x"], float)
            n = b.arguments.get("N") or len(x)
            xs = x[:n]
            tol = (32 * EPS * math.log2(n + 1) * float(np.linalg.norm(xs)) / math.sqrt(n)
                   + 16 * EPS * abs(float(np.sum(xs))) / math.sqrt(n) + 1e-300)
            g, w = flat(got), flat(want)
            if len(g) != len(w):
                raise Mismatch("got %d values, expected %d" % (len(g), len(w)))
            return worst(g, w, [tol + 4 * EPS * abs(v) for v in w])
        want = list(want)
        if len(got) != len(want):
            raise Mismatch("got %d results, expected %d" % (len(got), len(want)))
        same_values(flat(got[0]), flat(want[0]), "frequencies")
        if fn == "spectrogram":
            same_values(flat(got[1]), flat(want[1]), "times")
        g, w = flat(got[-1]), flat(want[-1])
        if len(g) != len(w):
            raise Mismatch("got %d spectrum values, expected %d" % (len(g), len(w)))
        if not w:
            return 0.0
        segs, win, nfft = segments_and_window(fn, b)
        a = dict(DEFAULTS[fn])
        a.update(b.arguments)
        onesided = bool(a.get("return_onesided", True))
        nf = len(flat(want[0]))
        doubled = [2 if onesided and 0 < k < (nf - 1 if nfft % 2 == 0 else nf) else 1 for k in range(nf)]
        if fn == "spectrogram":
            fs = float(a["fs"])
            s = (1.0 / (fs * float(np.sum(win * win))) if a["scaling"] == "density"
                 else 1.0 / float(np.sum(win)) ** 2)
            e = fft_error(segs, win, nfft)
            mode = a["mode"]
            cols = len(flat(want[1]))
            tol = []
            for k in range(nf):
                for p in range(cols):
                    v = w[k * cols + p]
                    if mode == "psd":
                        x_abs = math.sqrt(max(v, 0) / (abs(s) * doubled[k]))
                        tol.append(abs(s) * doubled[k] * (2 * x_abs * e + e * e) + 4 * EPS * abs(v) + 1e-300)
                    else:
                        tol.append(math.sqrt(abs(s)) * e + 4 * EPS * abs(v) + 1e-300)
            if mode in ("psd", "complex", "magnitude"):
                return worst(g, w, tol)
            if mode == "phase":
                # Equal modulo 2 pi does not test the unwrapping itself (its corrections are whole turns too), so
                # the result must also be a valid unwrapping: no step over pi between neighbouring frequencies.
                for k in range(1, nf):
                    for p in range(cols):
                        step = abs(g[k * cols + p] - g[(k - 1) * cols + p])
                        if step > math.pi + 64 * EPS * max(abs(g[k * cols + p]), 1.0):
                            raise Mismatch("phase not unwrapped: a step of %.4g rad between frequencies %d and %d"
                                           % (step, k - 1, k))
            cplx = flat(with_argument(fn, args, kwargs, "mode", "complex")[2])
            r = 0.0
            for gv, wv, t, c in zip(g, w, tol, cplx):
                d = abs(gv - wv)
                d = abs(math.remainder(d, 2 * math.pi)) if mode == "phase" or d > math.pi else d
                if mode == "angle":
                    d = min(d, abs(2 * math.pi - d))
                allowed = min(math.pi, t / abs(c)) if abs(c) > 0 else math.pi
                allowed += 8 * EPS * max(abs(wv), 1.0)
                if d:
                    r = max(r, d / allowed)
            return r
        # welch and periodogram: the window is scaled before the FFT
        fs = float(a["fs"])
        if a["scaling"] == "density":
            fac = 1 / math.sqrt(sum(float(v) * float(v) for v in win) / (1 / fs))
        else:
            fac = 1 / abs(sum(float(v) for v in win))
        e = fft_error(segs, win * fac, nfft)
        kseg = len(segs)
        median = a.get("average", "mean") == "median" and kseg > 1
        if median:
            mean = flat(with_argument(fn, args, kwargs, "average", "mean")[1])
            bias = 1 + sum(1. / (2 * i + 1) - 1. / (2 * i) for i in range(1, (kseg - 1) // 2 + 1))
        tol = []
        for k in range(nf):
            if median:
                x_abs = math.sqrt(max(mean[k], 0) * kseg / doubled[k])
                t = doubled[k] * (2 * x_abs * e + e * e) / bias
            else:
                x_abs = math.sqrt(max(w[k], 0) / doubled[k])
                t = doubled[k] * (2 * x_abs * e + e * e)
            tol.append(t + 4 * EPS * abs(w[k]) + 1e-300)
        return worst(g, w, tol)
    return cmp


# Generation

WINDOWS = ["hann_periodic", "hann", "hamming", "boxcar", "blackman", "flattop", "bartlett", "nuttall",
           ["tukey", 0.3], ["kaiser", 8.0], ["tukey_periodic", 0.25], ["gaussian", 7.0]]


def signal(rng, cap):
    n = max(1, random_length(rng, cap))
    if rng.random() < 0.5:
        return industrial(rng, n)
    t = np.arange(n)
    f1, f2 = rng.uniform(0.01, 0.45, 2)
    x = (rng.uniform(0.5, 5) * np.sin(2 * np.pi * f1 * t) + rng.uniform(0, 2) * np.cos(2 * np.pi * f2 * t)
         + rng.normal(0, rng.uniform(0.01, 1), n) + rng.choice([0.0, rng.uniform(-50, 1500)]))
    return x.tolist()


def window_for(rng, n):
    if rng.random() < 0.15:
        m = int(rng.integers(1, max(2, min(n, 300)) + 1))
        return [float(v) for v in rng.uniform(0.1, 1, m)], m
    return pick(rng, WINDOWS), None


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
        yield Case(fn, args, kwargs, kind=kind, compare=comparator(fn, args, kwargs))


def valid_values(rng, fn, cap):
    x = signal(rng, cap)
    n = len(x)
    values = {"x": x}
    if fn == "hilbert":
        if rng.random() < 0.3:
            values["N"] = int(rng.integers(1, 2 * n + 2))
        return values
    if rng.random() < 0.4:
        values["fs"] = pick(rng, [float(rng.uniform(0.1, 1000)), 50, 1.0])
    if rng.random() < 0.4:
        values["detrend"] = pick(rng, ["constant", "linear", False, None, "c", "l"]
                                 + (["", 0] if fn == "spectrogram" else []))
    if rng.random() < 0.3:
        values["return_onesided"] = pick(rng, [True, False])
    if rng.random() < 0.3:
        values["scaling"] = pick(rng, ["density", "spectrum"])
    if fn == "periodogram":
        if rng.random() < 0.5:
            values["window"] = pick(rng, WINDOWS + [None])
        if rng.random() < 0.4:
            values["nfft"] = pick(rng, [n, int(rng.integers(1, n + 1)), n + int(rng.integers(1, 300))])
        return values
    window, m = window_for(rng, n)
    if rng.random() < 0.6 or m is not None:
        values["window"] = window
    nperseg = m
    if m is None and rng.random() < 0.7:
        nperseg = int(rng.integers(1, min(n, 1024) + 1)) if rng.random() < 0.9 else n + 10
        values["nperseg"] = nperseg
    eff = min(nperseg if nperseg is not None else 256, n)
    if rng.random() < 0.4 and eff > 1:
        values["noverlap"] = int(rng.integers(-(eff // 2), eff))
    if rng.random() < 0.3:
        values["nfft"] = eff + int(rng.integers(0, 300))
    if fn == "welch" and rng.random() < 0.3:
        values["average"] = pick(rng, ["mean", "median"])
    if fn == "spectrogram" and rng.random() < 0.6:
        values["mode"] = pick(rng, ["psd", "complex", "magnitude", "angle", "phase"])
    return values


def invalid_values(rng):
    x = industrial(rng, int(rng.integers(40, 300)))
    what = int(rng.integers(0, 16))
    if what == 0:
        return "welch", {"x": x, "fs": pick(rng, [0, -2.0])}
    if what == 1:
        return pick(rng, ["welch", "spectrogram"]), {"x": x, "nperseg": pick(rng, [0, -4])}
    if what == 2:
        return "welch", {"x": x, "nperseg": 32, "noverlap": pick(rng, [32, 40])}
    if what == 3:
        return pick(rng, ["welch", "spectrogram"]), {"x": x, "nperseg": 32, "nfft": 16}
    if what == 4:
        return pick(rng, ["welch", "periodogram", "spectrogram"]), {"x": x, "detrend": "quadratic"}
    if what == 5:
        return pick(rng, ["welch", "periodogram", "spectrogram"]), {"x": x, "scaling": "power"}
    if what == 6:
        return "welch", {"x": x, "nperseg": 16, "average": "mode"}
    if what == 7:
        return pick(rng, ["welch", "spectrogram"]), {"x": x[:20], "window": [1.0] * 30}
    if what == 8:
        return "spectrogram", {"x": x, "mode": "power"}
    if what == 9:
        return "spectrogram", {"x": x, "fs": 0}
    if what == 10:
        return "hilbert", {"x": x, "N": pick(rng, [0, -3])}
    if what == 11:
        x[5] = pick(rng, [None, math.nan])
        return pick(rng, FUNCS), {"x": x}
    if what == 12:
        return pick(rng, ["welch", "spectrogram"]), {"x": x, "nperseg": 32.0}
    if what == 13:
        return "welch", {"x": x, "window": pick(rng, [None, 5.0])}
    if what == 14:
        return "hilbert", {"x": []}
    return "spectrogram", {"x": [], "nperseg": 4}


def wrong_call(rng):
    what = int(rng.integers(0, 3))
    if what == 0:
        return Case("welch", [], {"fs": 2.0}, kind="call")
    if what == 1:
        return Case("hilbert", [[1.0, 2.0, 3.0]], {"n": 8}, kind="call")
    return Case("spectrogram", [[1.0] * 20], {"window_length": 8}, kind="call")


# Properties

def properties(rng, n, max_len):
    """
    Parseval for the periodogram (sum of the density times the bin width is the variance); a tone's frequency is
    welch's peak; hilbert's real part is the signal; spectrogram densities are not negative.
    """
    for i in range(n):
        which = i % 4
        if which == 0:
            m = int(rng.integers(8, 3000))
            x = rng.normal(0, rng.uniform(0.1, 10), m) + rng.uniform(-100, 100)
            fs = float(rng.uniform(0.5, 100))
            var = float(np.mean((x - x.mean()) ** 2))
            yield Case("periodogram", [x.tolist()], {"fs": fs}, 1e-10,
                       "Parseval: sum(Pxx) * fs / n equals the variance",
                       check=lambda r, fs=fs, m=m, var=var: abs(math.fsum(r[1]) * fs / m - var) / var)
        elif which == 1:
            m = int(rng.integers(2048, 8000))
            f0 = float(rng.uniform(0.05, 0.45))
            x = np.sin(2 * np.pi * f0 * np.arange(m)) + rng.normal(0, 0.1, m)
            nperseg = 512
            yield Case("welch", [x.tolist()], {"nperseg": nperseg}, 1.0 / nperseg + 1e-12,
                       "a tone's frequency is the welch peak",
                       check=lambda r, f0=f0: abs(r[0][int(np.argmax(r[1]))] - f0))
        elif which == 2:
            x = signal(rng, 3000)
            scale = max(abs(v) for v in x) or 1.0
            yield Case("hilbert", [x], {}, 64 * EPS * math.log2(len(x) + 1) * scale * 4,
                       "hilbert's real part is the signal",
                       check=lambda r, x=x: max(abs(z.real - v) for z, v in zip(r, x)))
        else:
            x = signal(rng, 3000)
            yield Case("spectrogram", [x], {"nperseg": min(64, len(x))}, 1e-300,
                       "spectrogram densities are not negative",
                       check=lambda r: max([0.0] + [-v for row in r[2] for v in row]))
