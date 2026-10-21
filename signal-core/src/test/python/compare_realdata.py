#!/usr/bin/env python3
"""
Signal Processing end-to-end test with real plant data, part 2 of 2: checks every result e2e_realdata.py recorded
in Ignition against scipy, with the same references and comparisons as the random tests.

Usage, from the signal-core folder:
    python src/test/python/compare_realdata.py --results <path printed by e2e_realdata.py>

Exit code 0 when every result passes, 1 otherwise.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Case, decode  # noqa: E402
from random_check import check_all  # noqa: E402
from cases import detrend, filtering, peaks, resampling, smoothing, spectral  # noqa: E402


def case_for(fn, args, kwargs):
    """The case the random tests would build for this call, so the same comparison applies."""
    if fn == "detrend":
        return Case(fn, args, kwargs, detrend.tolerance(args[0]), "realdata")
    if fn == "medfilt":
        return Case(fn, args, kwargs, 0.0, "realdata")
    if fn in ("savgol_filter", "savgol_coeffs"):
        return Case(fn, args, kwargs, kind="realdata", compare=smoothing.within_ulp)
    if fn in peaks.FUNCS:
        return Case(fn, args, kwargs, kind="realdata", compare=peaks.exact)
    for module in (filtering, spectral, resampling):
        if fn in module.FUNCS:
            return Case(fn, args, kwargs, kind="realdata", compare=module.comparator(fn, args, kwargs))
    raise SystemExit("no comparison for %s" % fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="the results file e2e_realdata.py wrote")
    ap.add_argument("--show", type=int, default=5, help="failures to print per group (default 5)")
    args = ap.parse_args()
    with open(args.results, encoding="utf-8") as f:
        header = json.loads(f.readline())
        if header.get("kind") != "realdata":
            sys.exit("%s is not an e2e_realdata.py results file" % args.results)
        signals, cases, replies = {}, [], {}
        for line in f:
            if not line.strip():
                continue
            rec = decode(line)
            if "signal" in rec and "x" in rec:
                signals[rec["signal"]] = rec["x"]
                continue
            req, reply = rec["request"], rec["reply"]
            x = signals[req["signal"]]
            call_args = [x if a == "$x" else a for a in req["args"]]
            case = case_for(req["fn"], call_args, req["kwargs"])
            case.kind = "realdata"
            cases.append(case)
            replies[id(case)] = reply
    print("Ignition %s, %d tags, %d hours at %s s, %d calls"
          % (header["ignition"], len(signals), header["hours"], header["interval_seconds"], len(cases)))
    ok = check_all(cases, lambda batch: [replies[id(c)] for c in batch], args.show, None)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
