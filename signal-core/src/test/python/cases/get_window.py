"""
get_window: parity with scipy.signal.get_window, plus properties that need no reference.

Where the module differs from scipy (the reference below applies these on top of scipy):
    - A list works like a tuple: ["kaiser", 8.6] is the same as ("kaiser", 8.6). Jython and JSON cannot always
      tell a list from a tuple, and scipy raises a ValueError for a list.
    - dpss windows are not supported (ValueError).
    - Numeric parameters must be finite (module-wide rule).
"""

import inspect
import math
import warnings

import scipy.signal as ss

from harness import EPS, Case, make_call, pick

FN = "get_window"
ORDER = ["window", "Nx", "fftbins"]
DEFAULTS = {"fftbins": True}
SIGNATURE = inspect.signature(ss.get_window)

# Aliases from scipy's _WIN_FUNC_DATA, with how many parameters each window takes.
FAMILIES = {
    "barthann": (["barthann", "brthan", "bth"], "none"),
    "bartlett": (["bartlett", "bart", "brt"], "none"),
    "blackman": (["blackman", "black", "blk"], "none"),
    "blackmanharris": (["blackmanharris", "blackharr", "bkh"], "none"),
    "bohman": (["bohman", "bman", "bmn"], "none"),
    "boxcar": (["boxcar", "box", "ones", "rect", "rectangular"], "none"),
    "chebwin": (["chebwin", "cheb"], "required"),
    "cosine": (["cosine", "halfcosine"], "none"),
    "exponential": (["exponential", "poisson"], "optional"),
    "flattop": (["flattop", "flat", "flt"], "none"),
    "gaussian": (["gaussian", "gauss", "gss"], "required"),
    "general_cosine": (["general cosine", "general_cosine"], "required"),
    "general_gaussian": (["general gaussian", "general_gaussian", "general gauss", "general_gauss", "ggs"],
                         "required"),
    "general_hamming": (["general hamming", "general_hamming"], "required"),
    "hamming": (["hamming", "hamm", "ham"], "none"),
    "hann": (["hann", "han"], "none"),
    "kaiser": (["kaiser", "ksr"], "required"),
    "kbd": (["kaiser bessel derived", "kaiser_bessel_derived", "kbd"], "required"),
    "lanczos": (["lanczos", "sinc"], "none"),
    "nuttall": (["nuttall", "nutl", "nut"], "none"),
    "parzen": (["parzen", "parz", "par"], "none"),
    "taylor": (["taylor", "taylorwin"], "optional"),
    "triang": (["triangle", "triang", "tri"], "none"),
    "tukey": (["tukey", "tuk"], "optional"),
}
ALIASES = {alias: family for family, (names, _) in FAMILIES.items() for alias in names}


def reference(*args, **kwargs):
    bound = SIGNATURE.bind(*args, **kwargs)
    a = bound.arguments
    window = a["window"]
    if isinstance(window, list):
        window = tuple(window)
        a["window"] = window
    if isinstance(window, tuple):
        if window and isinstance(window[0], str):
            name = window[0]
            for suffix in ("_symmetric", "_periodic"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
            if name == "dpss":
                raise ValueError("dpss is not supported")
        for p in window[1:]:
            check_finite(p)
    elif isinstance(window, float):
        check_finite(window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ss.get_window(*bound.args, **bound.kwargs)


def check_finite(p):
    values = p if isinstance(p, (list, tuple)) else [p]
    for v in values:
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("parameters must be finite")


def tolerance(family, nx, params):
    """
    Measured agreement is within 4 eps for every window except chebwin. chebwin evaluates
    cosh(order * acosh(x)), which amplifies rounding by the window's order: against a 60-digit reference, scipy
    and this module lose accuracy at the same rate (both about 3600 eps at M=400), so the tolerance grows
    with M squared.
    """
    if family == "chebwin":
        return max(64.0, 0.5 * (nx + 1) ** 2) * EPS
    scale = 1.0
    if family == "general_cosine":
        scale = 1.0 + sum(abs(v) for v in params[0])
    elif family == "general_hamming":
        scale = 1.0 + abs(params[0]) + abs(1 - params[0])
    elif family == "taylor":
        scale = 4.0
    return 32 * EPS * scale


def random_params(rng, family, nx):
    """Parameters for a window family, or None to pass the bare name."""
    r = rng.random()
    if family == "chebwin":
        return [float(rng.uniform(20, 150))]
    if family == "exponential":
        if r < 0.3:
            return None
        center = None if rng.random() < 0.6 else float(rng.uniform(0, nx))
        return [center, float(rng.uniform(0.5, 50))]
    if family == "gaussian":
        return [float(rng.uniform(0.3, max(1.0, nx / 2)))]
    if family == "general_cosine":
        return [[float(v) for v in rng.uniform(-1, 1, int(rng.integers(1, 7)))]]
    if family == "general_gaussian":
        return [float(rng.uniform(0.5, 3)), float(rng.uniform(0.5, max(1.0, nx / 2)))]
    if family == "general_hamming":
        return [float(rng.uniform(0, 1))]
    if family == "kaiser":
        return [float(rng.uniform(0, 50)) if r < 0.9 else float(rng.uniform(50, 800))]
    if family == "kbd":
        return [float(rng.uniform(0, 20))]
    if family == "taylor":
        if r < 0.3:
            return None
        params = [int(rng.integers(1, 11)), float(rng.uniform(15, 100))]
        if rng.random() < 0.5:
            params.append(bool(rng.random() < 0.5))
        return params
    if family == "tukey":
        if r < 0.3:
            return None
        return [float(rng.uniform(-0.2, 1.2))]
    return None


def random_length(rng, max_len, family):
    cap = min(max_len, 5000 if family == "chebwin" else 20000)
    r = rng.random()
    if r < 0.15:
        return int(rng.integers(1, 6))
    if r < 0.6:
        return int(rng.integers(6, 129))
    return int(round(math.exp(rng.uniform(math.log(129), math.log(max(cap, 130))))))


def generate(rng, n, max_len):
    for _ in range(n):
        r = rng.random()
        if r < 0.80:
            yield valid_call(rng, max_len)
        elif r < 0.93:
            yield invalid_value(rng)
        else:
            yield wrong_call(rng)


def valid_call(rng, max_len):
    family = pick(rng, sorted(FAMILIES))
    nx = random_length(rng, max_len, family)
    fftbins = bool(rng.random() < 0.6)
    name = pick(rng, FAMILIES[family][0])
    suffix = ""
    if rng.random() < 0.15:
        suffix = pick(rng, ["_symmetric", "_periodic"])
    if family == "kbd":
        nx += nx % 2
        if not suffix:
            suffix = "_symmetric" if rng.random() < 0.8 else ""
    params = random_params(rng, family, nx)
    if params is None:
        window = name + suffix
    else:
        window = [name + suffix] + params
    if family == "kaiser" and params is not None and rng.random() < 0.3 and not suffix:
        window = params[0] if rng.random() < 0.7 else int(round(params[0]))
    values = {"window": window, "Nx": nx}
    if rng.random() < 0.7:
        values["fftbins"] = fftbins
    args, kwargs = make_call(rng, ORDER, values, DEFAULTS)
    return Case(FN, args, kwargs, tolerance(family, nx, params or [None]), "random")


def invalid_value(rng):
    nx = int(rng.integers(2, 200))
    values = {"window": "hann", "Nx": nx}
    what = int(rng.integers(0, 24))
    choices = {
        0: ("Nx", 0), 1: ("Nx", -3), 2: ("Nx", float(nx)), 3: ("Nx", True), 4: ("Nx", "8"), 5: ("Nx", None),
        6: ("fftbins", 1), 7: ("fftbins", "yes"), 8: ("fftbins", None),
        9: ("window", None), 10: ("window", "nonsense"), 11: ("window", "5"), 12: ("window", ["hann", 1.0]),
        13: ("window", ["kaiser"]), 14: ("window", ["kaiser", 1.0, 2.0]), 15: ("window", [5.0]),
        16: ("window", []), 17: ("window", ["dpss", 2.0]), 18: ("window", ["kbd", 3.0]),
        19: ("window", ["taylor", 0]), 20: ("window", ["taylor", 4.5]), 21: ("window", ["general_cosine", 0.5]),
        22: ("window", ["kaiser", "a"]), 23: ("window", ["tukey", None]),
    }
    key, value = choices[what]
    values[key] = value
    if what == 18 and rng.random() < 0.5:
        values["window"] = ["kbd_symmetric", 3.0]
        values["Nx"] = nx | 1
    if rng.random() < 0.2:
        values["window"] = ["exponential", 2.0]
        values["fftbins"] = False
    args, kwargs = make_call(rng, ORDER, values, DEFAULTS)
    return Case(FN, args, kwargs, 32 * EPS, "invalid")


def wrong_call(rng):
    what = int(rng.integers(0, 4))
    if what == 0:
        return Case(FN, ["hann"], {}, 0.0, "call")
    if what == 1:
        return Case(FN, ["hann", 8, True, 1], {}, 0.0, "call")
    if what == 2:
        return Case(FN, ["hann", 8], {"fft_bins": False}, 0.0, "call")
    return Case(FN, ["hann", 8], {"Nx": 8}, 0.0, "call")


def properties(rng, n, max_len):
    """
    Symmetric windows are palindromes; periodic Hann and Hamming overlap-add to a constant at 50%. The palindrome
    check allows 64 eps: scipy's formulas compute the two halves of some windows (Tukey's tapers, for example)
    with different expressions, so they mirror each other only to within rounding, in scipy as well.
    """
    for i in range(n):
        if i % 3 == 0:
            family = pick(rng, ["hann", "hamming", "blackman", "bartlett", "triang", "parzen", "bohman", "cosine",
                                "lanczos", "flattop", "nuttall", "barthann"])
            nx = int(rng.integers(2, 4000))
            yield Case(FN, [family, nx], {"fftbins": False}, 64 * EPS, "symmetric window is a palindrome",
                       check=palindrome)
        elif i % 3 == 1:
            family = pick(rng, [("kaiser", float(rng.uniform(0, 30))), ("tukey", float(rng.uniform(0, 1))),
                                ("gaussian", float(rng.uniform(1, 50))), ("taylor",)])
            nx = int(rng.integers(2, 4000))
            # Tukey's tapers take cos of terms of size 2/alpha, so their rounding grows as alpha shrinks.
            limit = 64 * EPS * (1 + 4 / family[1]) if family[0] == "tukey" and family[1] > 0 else 64 * EPS
            yield Case(FN, [list(family), nx], {"fftbins": False}, limit, "symmetric window is a palindrome",
                       check=palindrome)
        else:
            family, total = pick(rng, [("hann", 1.0), ("hamming", 1.08)])
            nx = 2 * int(rng.integers(1, 2000))
            yield Case(FN, [family, nx], {}, 16 * EPS, "periodic %s overlap-adds to %g" % (family, total),
                       check=lambda w, total=total: overlap_add(w, total))


def palindrome(w):
    return max((abs(a - b) for a, b in zip(w, reversed(w))), default=0.0)


def overlap_add(w, total):
    h = len(w) // 2
    return max(abs(w[k] + w[k + h] - total) for k in range(h))
