#!/usr/bin/env python3
"""
Randomised cross-check of the Signal Processing Java core against scipy.signal.

Each function is called with signals shaped like plant data, at random lengths and amplitudes, with arguments
passed by position and by keyword, and every result is compared with scipy plus the module's documented rules
(see the modules in cases/). Whenever scipy raises for a call, Java must raise too. Property checks that need
no reference run alongside, and a source check keeps the Ignition layer in step with the core.

Usage, from the signal-core folder:
    mvn -q test-compile
    python src/test/python/random_check.py                     every function, random seed
    python src/test/python/random_check.py --seed 1 --n 5000
    python src/test/python/random_check.py --only detrend
    python src/test/python/random_check.py --export            cases for the Ignition accuracy test

Exit code 0 when every check passes, 1 otherwise. Failing cases go to target/failures.jsonl.
"""

import argparse
import json
import math
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_sources  # noqa: E402
from cases import BY_NAME, MODULES, REFERENCES  # noqa: E402
from harness import (CORE, ERR, REPO, JavaCli, Mismatch, compare, describe,  # noqa: E402
                     describe_call, reference_call)

EXPORT_PATH = CORE / "target" / "ignition-cases.jsonl"
FAILURES_PATH = CORE / "target" / "failures.jsonl"


# Building and checking cases

CHUNK = 250


def build_cases(names, n, seed, max_len):
    """
    Yields the cases one at a time, numbered from 1. The same names, n, seed and max_len always give the same
    cases, here and in compare_ignition.py. Nothing is kept, so memory stays flat however many cases there are.
    """
    next_id = 1
    for name in names:
        module = BY_NAME[name]
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        for source in (module.generate(rng, n, max_len), module.properties(rng, max(1, n // 5), max_len)):
            for case in source:
                case.id = next_id
                next_id += 1
                yield case


def evaluate(case, reply):
    """(group, passed, worst error as a fraction of tolerance, reason) for one case and its reply."""
    if case.check is not None:
        group = case.fn + " properties"
        if "ok" not in reply:
            return group, False, math.inf, outcome(reply)
        try:
            err = case.check(reply["ok"])
        except Exception as e:
            # A check that cannot even use Java's result (for example a design rejecting its Wn) is a failure.
            return group, False, math.inf, "%s: the check raised %s: %s" % (case.kind, type(e).__name__, e)
        ratio = err / case.tol if case.tol > 0 else (0.0 if err == 0 else math.inf)
        return group, ratio <= 1.0, ratio, "%s: error %.3g, tolerance %.3g" % (case.kind, err, case.tol)
    want = reference_call(REFERENCES[case.fn], case.args, case.kwargs)
    if want == ERR:
        group = case.fn + " errors"
        if "err" in reply:
            return group, True, 0.0, ""
        return group, False, math.inf, "scipy raises, Java gave " + outcome(reply)
    group = case.fn + " values"
    if "ok" not in reply:
        return group, False, math.inf, "scipy returns a result, Java gave " + outcome(reply)
    try:
        if case.compare is not None:
            ratio = case.compare(reply["ok"], want)
        else:
            ratio = compare(reply["ok"], want, case.tol, rtol=case.rtol)
    except Mismatch as e:
        return group, False, math.inf, str(e)
    except Exception as e:
        return group, False, math.inf, "the comparison raised %s: %s" % (type(e).__name__, e)
    if case.compare is not None:
        return group, ratio <= 1.0, ratio, "worst error %.3g x tolerance (custom comparison)" % ratio
    return group, ratio <= 1.0, ratio, "worst error %.3g x tolerance (tol %.3g, rtol %.3g)" % (
        ratio, case.tol, case.rtol)


def encode_complex(v):
    if isinstance(v, complex):
        return {"__complex__": [v.real, v.imag]}
    raise TypeError("cannot store %r" % type(v).__name__)


def outcome(reply):
    if "ok" in reply:
        return "a result: " + describe(reply["ok"])
    if "err" in reply:
        return "an error: " + reply["err"]
    return "a CRASH: " + reply.get("crash", "no reply").strip().splitlines()[0]


def check_all(cases, get_replies, show, reproduce):
    """
    Runs the cases through get_replies (a function from a list of cases to their replies) in chunks, prints the
    summary table and failures, and returns True when everything passed.
    """
    groups = {}
    failures = []
    batch = []

    def run(batch):
        for case, reply in zip(batch, get_replies(batch)):
            group, passed, ratio, reason = evaluate(case, reply)
            g = groups.setdefault(group, {"n": 0, "fail": 0, "worst": 0.0, "ms": 0.0})
            g["n"] += 1
            if "ns" in reply:
                g["ms"] = max(g["ms"], reply["ns"] / 1e6)
            elif "ms" in reply:
                g["ms"] = max(g["ms"], reply["ms"])
            if math.isfinite(ratio):
                g["worst"] = max(g["worst"], ratio)
            if not passed:
                g["fail"] += 1
                failures.append((group, case, reply, reason))

    for case in cases:
        batch.append(case)
        if len(batch) == CHUNK:
            run(batch)
            batch = []
    if batch:
        run(batch)

    print()
    print("%-28s %7s %7s %20s %12s" % ("Check", "Cases", "Failed", "Worst (x tolerance)", "Slowest ms"))
    for name in sorted(groups):
        g = groups[name]
        worst = "%.3f" % g["worst"] if name.endswith(("values", "properties")) else "-"
        print("%-28s %7d %7d %20s %12.1f" % (name, g["n"], g["fail"], worst, g["ms"]))

    shown = {}
    for group, case, reply, reason in failures:
        if shown.get(group, 0) < show:
            shown[group] = shown.get(group, 0) + 1
            print("\nFAIL %s, case %d (%s): %s" % (group, case.id, case.kind, describe_call(case.fn, case.args,
                                                                                         case.kwargs)))
            print("     " + reason)
    if failures:
        FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FAILURES_PATH, "w", encoding="utf-8") as f:
            for group, case, reply, reason in failures:
                f.write(json.dumps({"group": group, "reason": reason, "case": case.export(), "reply": reply},
                                   default=encode_complex) + "\n")
        print("\nAll %d failures written to %s" % (len(failures), FAILURES_PATH))
        print("Reproduce with: " + reproduce)
    return not failures


def sources_ok(signatures):
    ok = True
    print()
    for name, problems in check_sources.check(REPO, signatures):
        print("%-28s %s" % (name, "PASS" if not problems else "FAIL"))
        for p in problems:
            print("     " + p)
        ok = ok and not problems
    return ok


# Command line

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=None, help="random calls per function (default 2000, export 200)")
    ap.add_argument("--seed", type=int, default=None, help="random seed (default: random, printed)")
    ap.add_argument("--only", default=None, help="comma-separated function names (default: all)")
    ap.add_argument("--max-len", type=int, default=None, help="longest signal (default 100000, export 5000)")
    ap.add_argument("--java", default="java", help="java executable (default: java on PATH)")
    ap.add_argument("--show", type=int, default=10, help="failures to print per group (default 10)")
    ap.add_argument("--export", action="store_true", help="write the Ignition accuracy test cases and stop")
    args = ap.parse_args()

    if args.only:
        names = [s.strip() for s in args.only.split(",")]
    else:
        # Internal building blocks (names starting with _) cannot be called from Ignition, so export skips them.
        names = [m.FN for m in MODULES if not (args.export and m.FN.startswith("_"))]
    unknown = [s for s in names if s not in BY_NAME]
    if unknown:
        sys.exit("Unknown function(s): %s. Known: %s" % (", ".join(unknown), ", ".join(BY_NAME)))
    if args.export and any(s.startswith("_") for s in names):
        sys.exit("Internal building blocks (names starting with _) cannot be exported to Ignition.")
    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    n = args.n if args.n is not None else (200 if args.export else 2000)
    max_len = args.max_len if args.max_len is not None else (5000 if args.export else 100000)
    cases = build_cases(names, n, seed, max_len)

    if args.export:
        EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        header = {"seed": seed, "n": n, "max_len": max_len, "functions": names, "scipy": scipy.__version__}
        count = 0
        with open(EXPORT_PATH, "w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
            for case in cases:
                f.write(json.dumps(case.request(), allow_nan=True) + "\n")
                count += 1
        print("Wrote %d cases to %s" % (count, EXPORT_PATH))
        source = (CORE / "src" / "test" / "ignition" / "accuracy_test.py").read_text(encoding="utf-8")
        filled = source.replace("CASES = None", "CASES = r%s" % json.dumps(str(EXPORT_PATH.resolve())), 1)
        script = EXPORT_PATH.parent / "accuracy_test.py"
        script.write_text(filled, encoding="utf-8")
        print("Wrote %s with that path filled in: paste it into the Designer Script Console and run it." % script)
        return 0

    print("scipy %s, numpy %s, seed %d, %d calls per function, signals up to %d points"
          % (scipy.__version__, np.__version__, seed, n, max_len))
    cli = JavaCli(args.java)
    reproduce = "python src/test/python/random_check.py --seed %d --n %d --max-len %d --only %s" % (
        seed, n, max_len, ",".join(names))
    values_ok = check_all(cases, lambda batch: cli.run([c.request() for c in batch]), args.show, reproduce)
    src_ok = sources_ok(cli.signatures())
    print("\n" + ("PASS" if values_ok and src_ok else "FAIL"))
    return 0 if values_ok and src_ok else 1


if __name__ == "__main__":
    sys.exit(main())
