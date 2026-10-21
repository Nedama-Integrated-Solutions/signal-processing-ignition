#!/usr/bin/env python3
"""
Signal Processing accuracy test, part 2 of 2: checks the results that accuracy_test.py wrote in Ignition.

Rebuilds the exported cases from the seed recorded with them, so exactly the same scipy references and
property checks as random_check.py apply, then reports the checks that only a gateway can do.

Usage, from the signal-core folder:
    python src/test/python/compare_ignition.py --results <path printed by accuracy_test.py>

Exit code 0 when every check passes, 1 otherwise.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import decode  # noqa: E402
from random_check import EXPORT_PATH, build_cases, check_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="the results file accuracy_test.py wrote")
    ap.add_argument("--cases", default=str(EXPORT_PATH), help="the exported cases (default: %(default)s)")
    ap.add_argument("--show", type=int, default=10, help="failures to print per group (default 10)")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        lines = [decode(line) for line in f if line.strip()]
    header, replies, checks = lines[0], {}, []
    for line in lines[1:]:
        if line.get("id") == "ignition":
            checks = line["checks"]
        else:
            replies[line["id"]] = line

    with open(args.cases, encoding="utf-8") as f:
        exported = [line.strip() for line in f if line.strip()][1:]
    # Compared as text: NaN never equals NaN, so comparing parsed values would always fail on NaN cases.
    rebuilt = [json.dumps(c.request(), allow_nan=True)
               for c in build_cases(header["functions"], header["n"], header["seed"], header["max_len"])]
    if rebuilt != exported:
        sys.exit("The cases rebuilt from seed %d do not match %s. Check that the Python packages match "
                 "requirements.txt, or export again and rerun accuracy_test.py." % (header["seed"], args.cases))
    del rebuilt, exported

    print("Ignition %s, scipy %s, seed %d" % (header.get("ignition", "?"), header.get("scipy", "?"), header["seed"]))
    cases = build_cases(header["functions"], header["n"], header["seed"], header["max_len"])
    missing = []

    def replies_for(batch):
        out = []
        for c in batch:
            if c.id not in replies:
                missing.append(c.id)
            out.append(replies.get(c.id, {"id": c.id, "crash": "no result in the file"}))
        return out

    reproduce = "python src/test/python/random_check.py --export --seed %d --n %d --max-len %d --only %s" % (
        header["seed"], header["n"], header["max_len"], ",".join(header["functions"]))
    values_ok = check_all(cases, replies_for, args.show, reproduce)
    if missing:
        print("\nNo result for %d cases, for example case %d" % (len(missing), missing[0]))

    print()
    checks_ok = bool(checks)
    if not checks:
        print("No Ignition-only checks found in the results file.")
    for c in checks:
        print("%-40s %s %s" % (c["check"], "PASS" if c["pass"] else "FAIL", c.get("detail", "")))
        checks_ok = checks_ok and c["pass"]
    ok = values_ok and checks_ok and not missing
    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
