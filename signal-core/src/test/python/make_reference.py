"""
Writes docs/functions.md, the function reference, from the module itself: parameter names from the Java core,
defaults from SciPy's own signatures (which the module mirrors), and descriptions from the Ignition hints in
SignalScriptModule.properties. Run from the signal-core folder after mvn -q test-compile:

    python src/test/python/make_reference.py
"""

import inspect
import json
import pathlib
import sys

import scipy.signal as ss

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from harness import JavaCli  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[4]
HINTS = ROOT / "common-81" / "src" / "main" / "resources" / "com" / "nedama" / "signal" / "common81" / \
    "SignalScriptModule.properties"
GROUPS = [
    ("Detrending and windows", ["detrend", "get_window"]),
    ("Filter design", ["butter", "cheby1", "cheby2", "ellip", "bessel"]),
    ("Filter order selection", ["buttord", "cheb1ord", "cheb2ord", "ellipord"]),
    ("Frequency response", ["freqz", "freqz_sos", "sosfreqz"]),
    ("Filtering", ["lfilter", "lfilter_zi", "sosfilt", "sosfilt_zi", "filtfilt", "sosfiltfilt"]),
    ("Smoothing", ["savgol_coeffs", "savgol_filter", "medfilt"]),
    ("Peak finding", ["find_peaks", "peak_prominences", "peak_widths"]),
    ("Spectral analysis", ["welch", "periodogram", "spectrogram", "hilbert"]),
    ("Resampling", ["resample", "resample_poly", "decimate"]),
]


def hints():
    out = {}
    for line in HINTS.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def main():
    cli = JavaCli(sys.argv[1] if len(sys.argv) > 1 else "java")
    sigs = cli.run([{"id": 0, "fn": "__signatures__", "args": [], "kwargs": {}}])[0]["ok"]
    h = hints()
    names = [fn for _, fns in GROUPS for fn in fns]
    missing = sorted(set(sigs) - set(names))
    if missing:
        raise SystemExit("not in any group: %s" % missing)
    lines = ["# Function reference", "",
             "All functions are under `system.signal` and work in every scope. Names, parameters, defaults and "
             "results follow SciPy 1.18.1; positional and keyword calls both work. Signals are lists (or tuples, "
             "or Java arrays) of numbers, evenly sampled. See [Differences from SciPy](differences.md) for the "
             "few deliberate departures.", ""]
    for title, fns in GROUPS:
        lines += ["## " + title, ""]
        for fn in fns:
            params = sigs[fn]["params"]
            ref = inspect.signature(getattr(ss, fn)).parameters
            shown = []
            for p in params:
                d = ref[p].default if p in ref else inspect.Parameter.empty
                shown.append(p if d is inspect.Parameter.empty else "%s=%r" % (p, d))
            lines += ["### %s" % fn, "", "```python", "system.signal.%s(%s)" % (fn, ", ".join(shown)), "```", "",
                      h.get(fn + ".desc", ""), "", "| Parameter | Description |", "| --- | --- |"]
            for p in params:
                lines.append("| `%s` | %s |" % (p, h.get("%s.param.%s" % (fn, p), "").replace("|", "\\|")))
            lines += ["", "**Returns:** " + h.get(fn + ".returns", ""), ""]
    out = ROOT / "docs" / "functions.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote %s (%d functions)" % (out, len(names)))


if __name__ == "__main__":
    main()
