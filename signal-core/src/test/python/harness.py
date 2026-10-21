"""
Shared pieces of the Signal Processing test harness.

    JavaCli          runs com.nedama.signal.SignalCli over a batch of calls
    Case             one call, with its tolerance and an optional property check
    reference_call   runs a reference function and turns any exception into ERR
    compare          compares a Java result with a reference result, returning the worst error / tolerance
    make_call        splits chosen arguments into positional and keyword parts at random
    industrial       signals shaped like plant data, not just white noise
"""

import json
import math
import warnings
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

CORE = Path(__file__).resolve().parents[3]
REPO = CORE.parent
EPS = np.finfo(float).eps
ERR = "ERR"


# Cases

@dataclass
class Case:
    fn: str
    args: list
    kwargs: dict = field(default_factory=dict)
    tol: float = 0.0
    kind: str = "random"
    rtol: float = 0.0                  # added to tol as rtol * |expected value|, element by element
    check: Optional[Callable] = None   # property check: check(result) -> error, compared with tol
    compare: Optional[Callable] = None  # custom comparison: compare(got, want) -> worst error as a fraction of tolerance
    id: int = 0

    def request(self):
        return {"id": self.id, "fn": self.fn, "args": self.args, "kwargs": self.kwargs}

    def export(self):
        return {"id": self.id, "fn": self.fn, "args": self.args, "kwargs": self.kwargs, "tol": self.tol,
                "rtol": self.rtol, "kind": self.kind}


def reference_call(fn, args, kwargs):
    """Runs a reference. Any exception means the call is invalid, which Java must report as an error."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return plain(fn(*args, **kwargs))
    except Exception:
        return ERR


# Java driver

class JavaCli:

    def __init__(self, java="java"):
        classes = CORE / "target" / "classes"
        test_classes = CORE / "target" / "test-classes"
        if not test_classes.exists():
            sys.exit("Compiled classes not found. Run 'mvn -q test-compile' in %s first." % CORE)
        self.cmd = [java, "-cp", os.pathsep.join([str(classes), str(test_classes)]), "com.nedama.signal.SignalCli"]

    def run(self, requests):
        text = "\n".join(json.dumps(r, allow_nan=True) for r in requests) + "\n"
        proc = subprocess.run(self.cmd, input=text, capture_output=True, text=True, encoding="utf-8")
        if proc.returncode != 0:
            sys.exit("Java driver failed:\n" + proc.stderr)
        replies = [decode(line) for line in proc.stdout.splitlines()]
        if len(replies) != len(requests):
            sys.exit("Java driver returned %d replies for %d calls" % (len(replies), len(requests)))
        return replies

    def signatures(self):
        return self.run([{"id": 0, "fn": "__signatures__"}])[0]["ok"]


# Comparison

class Mismatch(Exception):
    pass


def _complex_hook(obj):
    if len(obj) == 1 and "__complex__" in obj:
        re, im = obj["__complex__"]
        return complex(re, im)
    return obj


def decode(text):
    """json.loads, turning {"__complex__": [re, im]} (how Java and Ignition write complex numbers) into complex."""
    return json.loads(text, object_hook=_complex_hook)


def plain(x):
    """numpy arrays and scalars to plain Python lists, floats and ints; tuples to lists."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    return x


def compare(got, want, tol, path="result", rtol=0.0):
    """
    Worst |got - want| / (tol + rtol * |want|) over every number. Integers and structure must match exactly, and
    with no tolerance at all, floats must match bit for bit.
    """
    if isinstance(want, complex):
        if not isinstance(got, complex):
            raise Mismatch("%s: got %r, expected a complex number" % (path, got))
        bound = tol + rtol * abs(want)
        worst = 0.0
        for g, w, part in ((got.real, want.real, "real"), (got.imag, want.imag, "imag")):
            if math.isnan(w) or math.isnan(g) or math.isinf(w) or math.isinf(g):
                if not (g == w or (math.isnan(g) and math.isnan(w))):
                    raise Mismatch("%s: got %r, expected %r" % (path, got, want))
                continue
            diff = abs(g - w)
            if bound == 0.0:
                worst = max(worst, 0.0 if diff == 0.0 else math.inf)
            else:
                worst = max(worst, diff / bound)
        return worst
    if isinstance(want, bool) or want is None or isinstance(want, str):
        if got != want:
            raise Mismatch("%s: got %r, expected %r" % (path, got, want))
        return 0.0
    if isinstance(want, int):
        if isinstance(got, bool) or not isinstance(got, int) or got != want:
            raise Mismatch("%s: got %r, expected the integer %r" % (path, got, want))
        return 0.0
    if isinstance(want, float):
        if not isinstance(got, float):
            raise Mismatch("%s: got %r, expected a float" % (path, got))
        if math.isnan(want) or math.isnan(got):
            if math.isnan(want) and math.isnan(got):
                return 0.0
            raise Mismatch("%s: got %r, expected %r" % (path, got, want))
        if math.isinf(want) or math.isinf(got):
            if got == want:
                return 0.0
            raise Mismatch("%s: got %r, expected %r" % (path, got, want))
        diff = abs(got - want)
        bound = tol + rtol * abs(want)
        if bound == 0.0:
            return 0.0 if diff == 0.0 else math.inf
        return diff / bound
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            raise Mismatch("%s: got %s, expected a list of %d" % (path, describe(got), len(want)))
        worst = 0.0
        for i, (g, w) in enumerate(zip(got, want)):
            worst = max(worst, compare(g, w, tol, "%s[%d]" % (path, i), rtol))
        return worst
    if isinstance(want, dict):
        if not isinstance(got, dict) or set(got) != set(want):
            raise Mismatch("%s: got keys %s, expected %s" % (path, sorted(got) if isinstance(got, dict) else got,
                                                             sorted(want)))
        worst = 0.0
        for k in want:
            worst = max(worst, compare(got[k], want[k], tol, "%s[%r]" % (path, k), rtol))
        return worst
    raise Mismatch("%s: cannot compare reference value of type %s" % (path, type(want).__name__))


def describe(v, limit=6):
    """Short text for a value, with long lists cut down."""
    if isinstance(v, list):
        if len(v) > limit:
            return "[%s, ... %d items]" % (", ".join(describe(x) for x in v[:limit]), len(v))
        return "[%s]" % ", ".join(describe(x) for x in v)
    if isinstance(v, dict):
        return "{%s}" % ", ".join("%r: %s" % (k, describe(x)) for k, x in v.items())
    if isinstance(v, float):
        return "%.6g" % v
    return repr(v)


def describe_call(fn, args, kwargs):
    parts = [describe(a) for a in args] + ["%s=%s" % (k, describe(v)) for k, v in kwargs.items()]
    return "%s(%s)" % (fn, ", ".join(parts))


def scale_of(values):
    """Largest finite magnitude in a list of numbers, or 1.0 if there is none."""
    m = 0.0
    if isinstance(values, (list, tuple)):
        for v in values:
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                m = max(m, abs(float(v)))
    return m if m > 0.0 else 1.0


# Call building

def make_call(rng, order, values, defaults):
    """
    Splits chosen arguments into args and kwargs the way a person might call the function: a random number of
    leading parameters passed by position, the rest by keyword. Parameters skipped in the positional part are
    filled with their defaults, since Python cannot skip a positional slot.
    """
    chosen = [p for p in order if p in values]
    last = max((order.index(p) for p in chosen), default=-1)
    k = int(rng.integers(0, last + 2))
    args = [values.get(p, defaults.get(p)) for p in order[:k]]
    kwargs = {p: values[p] for p in order[k:] if p in values}
    return args, kwargs


def pick(rng, options):
    """One of the options, returned as the Python object itself (numpy's choice converts types)."""
    return options[int(rng.integers(0, len(options)))]


# Signals shaped like plant data

KINDS = ("noisy_step", "ramp", "vibration", "quantised", "batch", "random_walk", "constant", "large_offset",
         "spiky", "white")


def random_length(rng, max_len):
    r = rng.random()
    if r < 0.05:
        n = int(rng.integers(0, 4))
    elif r < 0.40:
        n = int(rng.integers(4, 65))
    elif r < 0.80:
        n = int(rng.integers(65, 2049))
    else:
        n = int(round(math.exp(rng.uniform(math.log(2049), math.log(max(max_len, 2050))))))
    return min(n, max_len)


def industrial(rng, n, kind=None):
    """A list of floats shaped like one kind of plant signal, at a random amplitude between 1e-6 and 1e6."""
    kind = kind or KINDS[int(rng.integers(0, len(KINDS)))]
    t = np.arange(n, dtype=float)
    amp = 10.0 ** rng.uniform(-6, 6)
    noise = rng.normal(0.0, 1.0, n)
    if kind == "noisy_step":
        k = int(rng.integers(0, n + 1))
        x = rng.uniform(-1, 1) + rng.uniform(-1, 1) * (t >= k) + 0.05 * noise
    elif kind == "ramp":
        x = rng.uniform(-1, 1) + rng.uniform(-1, 1) * t / max(n, 1) + 0.02 * noise
    elif kind == "vibration":
        f0 = rng.uniform(0.001, 0.2)
        x = sum(rng.uniform(0, 1) / h * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.3)) for h in range(1, 6))
        x = x + 0.1 * noise
    elif kind == "quantised":
        smooth = np.cumsum(noise) / math.sqrt(max(n, 1))
        q = pick(rng, [0.01, 0.1, 0.5])
        x = np.round(smooth / q) * q
    elif kind == "batch":
        x = 0.1 * t / max(n, 1)
        for _ in range(int(rng.integers(1, 5))):
            c, w = rng.uniform(0, n), rng.uniform(1, max(2.0, n / 8))
            x = x + rng.uniform(0.2, 1) * np.exp(-0.5 * ((t - c) / w) ** 2)
        x = x + 0.01 * noise
    elif kind == "random_walk":
        x = np.cumsum(noise)
    elif kind == "constant":
        x = np.full(n, rng.uniform(-1, 1))
    elif kind == "large_offset":
        x = 1e6 * pick(rng, [-1, 1]) + 1e-3 * noise
        amp = 10.0 ** rng.uniform(-3, 3)
    elif kind == "spiky":
        x = 0.1 * noise
        spikes = rng.random(n) < 0.01
        x[spikes] += rng.choice([-1, 1], spikes.sum()) * rng.uniform(5, 20, spikes.sum())
    else:
        x = noise
    return [float(v) for v in amp * x]


# Rules the module applies to every function, which the references add on top of scipy

def check_signal(x):
    """Every signal is a 1-D sequence of real numbers with no None, NaN or infinite values."""
    if isinstance(x, (list, tuple)):
        for v in x:
            if isinstance(v, (list, tuple)):
                raise TypeError("signals must be 1-D")
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                raise ValueError("signals must not contain None, NaN or infinite values")
