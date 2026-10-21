# Signal Processing end-to-end test with real plant data, part 1 of 2.
#
# Pulls historian data for the tags below, fills gaps, runs a suite of calls through system.signal on each tag,
# and writes every call and result to a file. Part 2 checks every result against scipy with the same rules as the
# random tests:
#     python src/test/python/compare_realdata.py --results <path printed at the end>
#
# Set TAGS (and HOURS, INTERVAL_SECONDS) below, paste into the Designer Script Console and run it. It also runs in
# gateway scope from a project library script (call run() from a message handler).

import json
import math
import time

from java.io import BufferedWriter, FileWriter
from java.lang import Throwable

TAGS = ["[default]Line1/Flow", "[default]Line1/Temperature"]
HOURS = 6
INTERVAL_SECONDS = 1


def history(path, hours, interval_seconds):
    end = system.date.now()
    start = system.date.addHours(end, -hours)
    ds = system.tag.queryTagHistory(paths=[path], startDate=start, endDate=end, intervalSeconds=interval_seconds,
                                    aggregationMode="Average", returnFormat="Wide")
    values = [ds.getValueAt(r, 1) for r in range(ds.rowCount)]
    gaps = sum(1 for v in values if v is None)
    last = next((v for v in values if v is not None), None)
    if last is None:
        return [], gaps
    filled = []
    for v in values:
        last = v if v is not None else last
        filled.append(float(last))
    return filled, gaps


def dumps(v):
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, float):
        if v != v:
            return "NaN"
        if v in (float("inf"), float("-inf")):
            return "Infinity" if v > 0 else "-Infinity"
        return repr(v)
    if isinstance(v, (int, long)):
        return str(v)
    if isinstance(v, complex):
        return '{"__complex__":[%s,%s]}' % (dumps(v.real), dumps(v.imag))
    if isinstance(v, (list, tuple)):
        return "[" + ",".join([dumps(x) for x in v]) + "]"
    if isinstance(v, dict):
        return "{" + ",".join([json.dumps(str(k)) + ":" + dumps(x) for k, x in v.items()]) + "}"
    return json.dumps(v)


def suite(x):
    """The calls made on each tag, as (function, args, kwargs). "$x" stands for the tag's signal."""
    n = len(x)
    mean = sum(x) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in x) / n) or 1.0
    sos = system.signal.butter(4, 0.05, output="sos")
    b, a = system.signal.butter(3, 0.05)
    calls = [
        ("detrend", ["$x"], {}),
        ("detrend", ["$x"], {"type": "constant"}),
        ("sosfiltfilt", [sos, "$x"], {}),
        ("sosfilt", [sos, "$x"], {}),
        ("filtfilt", [b, a, "$x"], {}),
        ("lfilter", [b, a, "$x"], {}),
        ("medfilt", ["$x", 5], {}),
        ("savgol_filter", ["$x", 31, 3], {}),
        ("savgol_filter", ["$x", 31, 3], {"deriv": 1, "delta": float(INTERVAL_SECONDS)}),
        ("find_peaks", ["$x"], {"prominence": std, "width": 2}),
        ("welch", ["$x"], {"fs": 1.0 / INTERVAL_SECONDS, "nperseg": min(2048, n)}),
        ("periodogram", ["$x"], {"fs": 1.0 / INTERVAL_SECONDS}),
        ("spectrogram", ["$x"], {"fs": 1.0 / INTERVAL_SECONDS, "nperseg": min(256, n)}),
        ("hilbert", [[v - mean for v in x]], {}),
        ("decimate", ["$x", 10], {}),
        ("decimate", ["$x", 10], {"ftype": "fir"}),
        ("resample_poly", ["$x", 2, 3], {}),
        ("resample", ["$x", max(1, n // 2)], {}),
    ]
    return calls


def run():
    out = system.file.getTempFile("jsonl")
    writer = BufferedWriter(FileWriter(out))
    writer.write(json.dumps({"kind": "realdata", "ignition": str(system.util.getVersion()), "tags": TAGS,
                             "hours": HOURS, "interval_seconds": INTERVAL_SECONDS}))
    writer.newLine()
    lines = []
    case_id = 0
    for tag in TAGS:
        x, gaps = history(tag, HOURS, INTERVAL_SECONDS)
        if len(x) < 100:
            lines.append("%s: only %d samples, skipped" % (tag, len(x)))
            continue
        writer.write(dumps({"signal": tag, "x": x}))
        writer.newLine()
        total_ms = 0.0
        failures = 0
        for fn, args, kwargs in suite(x):
            real_args = [x if a == "$x" else a for a in args]
            request = {"id": case_id, "fn": fn, "args": args, "kwargs": kwargs, "signal": tag}
            t0 = time.time()
            try:
                reply = {"id": case_id, "ok": getattr(system.signal, fn)(*real_args, **kwargs)}
            except (ValueError, TypeError) as e:
                reply = {"id": case_id, "err": "%s: %s" % (type(e).__name__, e)}
                failures += 1
            except Throwable as e:
                reply = {"id": case_id, "crash": "Java exception leaked: %s" % e}
                failures += 1
            reply["ms"] = (time.time() - t0) * 1000.0
            total_ms += reply["ms"]
            writer.write(dumps({"request": request, "reply": reply}))
            writer.newLine()
            case_id += 1
        writer.flush()
        lines.append("%s: %d samples (%d gaps filled), %d calls in %.0f ms, %d errors"
                     % (tag, len(x), gaps, len(suite(x)), total_ms, failures))
    writer.close()
    lines.append("Results written to " + out)
    lines.append("Next: python src/test/python/compare_realdata.py --results " + out)
    return lines


for line in run():
    print line
