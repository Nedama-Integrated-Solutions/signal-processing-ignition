"""
freqz, freqz_sos and sosfreqz against scipy.signal, plus properties.

Where the module differs from scipy: plot must be None (it would call back into Python); numbers must be finite
and real.

Tolerance: w within 8 eps of its size. Each h value within 64 eps (1 + log2 n) (sum|b| + |h| sum|a|) / |A|, the
rounding bound of evaluating B/A (A and B being the polynomials at that frequency; for second-order sections the
bound of each section, scaled by the other sections' gains, summed).
"""

import cmath
import inspect
import math
import warnings

import scipy.signal as ss

from harness import EPS, Case, Mismatch, make_call, pick

FN = "freqz"
SIG = inspect.signature(ss.freqz)
SOS_SIG = inspect.signature(ss.freqz_sos)
ORDER = list(SIG.parameters)
SOS_ORDER = list(SOS_SIG.parameters)
DEFAULTS = {k: v.default for k, v in SIG.parameters.items() if v.default is not inspect.Parameter.empty}
SOS_DEFAULTS = {k: v.default for k, v in SOS_SIG.parameters.items() if v.default is not inspect.Parameter.empty}


def finite(v):
    values = v if isinstance(v, (list, tuple)) else [v]
    for x in values:
        if isinstance(x, (list, tuple)):
            finite(x)
        elif isinstance(x, float) and not math.isfinite(x):
            raise ValueError("numbers must be finite")


def ref_freqz(*args, **kwargs):
    bound = SIG.bind(*args, **kwargs)
    for v in bound.arguments.values():
        finite(v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ss.freqz(*bound.args, **bound.kwargs)


def make_sos_reference(fn):
    def reference(*args, **kwargs):
        bound = SOS_SIG.bind(*args, **kwargs)
        for v in bound.arguments.values():
            finite(v)
        return getattr(ss, fn)(*bound.args, **bound.kwargs)
    return reference


REFERENCES = {"freqz": ref_freqz, "freqz_sos": make_sos_reference("freqz_sos"),
              "sosfreqz": make_sos_reference("sosfreqz")}
reference = ref_freqz


def poly_at(c, zm1):
    return sum(v * zm1 ** k for k, v in enumerate(c))


def compare_response(sections, fs):
    """sections: list of (b, a), one for freqz; fs: the sampling frequency the w values are in."""
    def cmp(got, want):
        (w, h), (ww, hw) = got, want
        if len(w) != len(ww) or len(h) != len(hw):
            raise Mismatch("got %d frequencies, expected %d" % (len(w), len(ww)))
        worst = 0.0
        for g, x in zip(w, ww):
            b = 8 * EPS * abs(x)
            d = abs(g - x)
            worst = max(worst, (0.0 if d == 0 else math.inf) if b == 0 else d / b)
        n = max([len(b) for b, a in sections] + [2])
        factor = 64 * EPS * (1 + math.log2(max(len(h), n)))
        for k, (g, x) in enumerate(zip(h, hw)):
            if not isinstance(g, complex):
                raise Mismatch("h[%d]: got %r, expected a complex number" % (k, g))
            if cmath.isnan(x) or cmath.isnan(g):
                if not (cmath.isnan(x) and cmath.isnan(g)):
                    raise Mismatch("h[%d]: got %r, expected %r" % (k, g, x))
                continue
            zm1 = cmath.exp(-1j * 2 * math.pi * ww[k] / fs)
            bound = 0.0
            parts = []
            for b, a in sections:
                B = poly_at(b, zm1)
                A = poly_at(a, zm1)
                hs = B / A if A != 0 else complex(math.inf)
                eb = (sum(abs(v) for v in b) + abs(hs) * sum(abs(v) for v in a)) / max(abs(A), 1e-300)
                parts.append((hs, eb))
            for i, (hs, eb) in enumerate(parts):
                others = 1.0
                for j, (hj, _) in enumerate(parts):
                    if j != i:
                        others *= abs(hj)
                bound += eb * others
            bound *= factor
            d = abs(g - x)
            worst = max(worst, (0.0 if d == 0 else math.inf) if bound == 0 else d / bound)
        return worst
    return cmp


def response_case(fn, args, kwargs, sections, fs):
    return Case(fn, args, kwargs, kind="random", compare=compare_response(sections, fs))


def random_filter(rng):
    r = rng.random()
    if r < 0.4:
        n = int(rng.integers(1, 200))
        b = [float(v) for v in rng.normal(0, 1, n)]
        a = pick(rng, [1, 1.0, [1.0], float(rng.uniform(0.5, 2))])
        return b, a
    order = int(rng.integers(1, 9))
    kind = pick(rng, ["butter", "cheby1", "ellip"])
    extra = {"butter": [], "cheby1": [1.0], "ellip": [1.0, 40.0]}[kind]
    b, a = getattr(ss, kind)(order, *extra, float(rng.uniform(0.05, 0.9)), btype=pick(rng, ["low", "high"]))
    return b.tolist(), a.tolist()


def random_sos(rng):
    order = int(rng.integers(1, 13))
    kind = pick(rng, ["butter", "cheby1", "cheby2", "ellip"])
    extra = {"butter": [], "cheby1": [1.0], "cheby2": [40.0], "ellip": [1.0, 40.0]}[kind]
    band = pick(rng, ["low", "high", "bandpass", "bandstop"])
    wn = float(rng.uniform(0.05, 0.9)) if band in ("low", "high") else sorted(rng.uniform(0.05, 0.9, 2).tolist())
    if isinstance(wn, list) and wn[1] - wn[0] < 0.02:
        wn[1] = wn[0] + 0.02
    return getattr(ss, kind)(order, *extra, wn, btype=band, output="sos").tolist()


def random_worn(rng, fs):
    r = rng.random()
    if r < 0.6:
        return int(rng.integers(0, 3000))
    if r < 0.7:
        return None
    if r < 0.8:
        return float(rng.uniform(0, fs / 2))
    return [float(v) for v in rng.uniform(0, fs / 2, int(rng.integers(1, 300)))]


def generate(rng, n, max_len):
    for _ in range(n):
        r = rng.random()
        fs = 2 * math.pi if rng.random() < 0.6 else float(rng.uniform(1, 10000))
        if r < 0.5:
            b, a = random_filter(rng)
            values = {"b": b, "a": a}
            worn = random_worn(rng, fs)
            if worn is not None or rng.random() < 0.3:
                values["worN"] = worn
            if rng.random() < 0.3:
                values["whole"] = bool(rng.random() < 0.5)
            if rng.random() < 0.3:
                values["include_nyquist"] = bool(rng.random() < 0.5)
            if fs != 2 * math.pi:
                values["fs"] = fs
            args, kwargs = make_call(rng, ORDER, values, DEFAULTS)
            sections = [(b, [a] if not isinstance(a, list) else a)]
            yield response_case("freqz", args, kwargs, sections, fs)
        elif r < 0.9:
            fn = pick(rng, ["freqz_sos", "sosfreqz"])
            sos = random_sos(rng)
            values = {"sos": sos}
            worn = random_worn(rng, fs)
            if worn is not None or rng.random() < 0.3:
                values["worN"] = worn
            if rng.random() < 0.3:
                values["whole"] = bool(rng.random() < 0.5)
            if fs != 2 * math.pi:
                values["fs"] = fs
            args, kwargs = make_call(rng, SOS_ORDER, values, SOS_DEFAULTS)
            yield response_case(fn, args, kwargs, [(row[:3], row[3:]) for row in sos], fs)
        else:
            yield invalid(rng)


def invalid(rng):
    what = int(rng.integers(0, 7))
    if what == 0:
        return Case("freqz", [[1.0, 2.0]], {"worN": -1}, kind="invalid")
    if what == 1:
        return Case("freqz", [[1.0, 2.0]], {"fs": None}, kind="invalid")
    if what == 2:
        return Case("freqz_sos", [[[1.0, 0.0, 0.0, 2.0, 0.0, 0.0]]], {}, kind="invalid")
    if what == 3:
        return Case("sosfreqz", [[[1.0, 0.0, 0.0, 1.0, 0.0]]], {}, kind="invalid")
    if what == 4:
        return Case("freqz_sos", [[]], {}, kind="invalid")
    if what == 5:
        return Case("freqz", [], {}, kind="call")
    return Case("freqz", [[1.0]], {"wor_n": 8}, kind="call")


def properties(rng, n, max_len):
    """
    b = a = [1] gives 1 everywhere; a low-pass design has unit gain at zero frequency. freqz(1, 1) takes the FFT
    path, and for FFT lengths with a large prime factor the transform removes the mean first (see Fft.java),
    which rounds an impulse's transform by about 1e-16, so the check allows 1e-13 rather than exactly 1.
    """
    for i in range(n):
        if i % 2 == 0:
            yield Case("freqz", [[1.0], [1.0]], {"worN": int(rng.integers(1, 2000))}, 1e-13,
                       "freqz(1, 1) is 1", check=lambda r: max(abs(h - 1) for h in r[1]))
        else:
            sos = ss.butter(int(rng.integers(1, 11)), float(rng.uniform(0.05, 0.9)), output="sos").tolist()
            yield Case("freqz_sos", [sos], {"worN": 64}, 1e-12, "low-pass gain at 0 Hz is 1",
                       check=lambda r: abs(r[1][0] - 1))
