# Signal Processing end-to-end concurrency test.
#
# Gateways run many scripts at once. This computes reference results one call at a time, then runs the same calls
# from a pool of threads, many times over, and requires every concurrent result to be identical to its reference.
# Paste into the Designer Script Console and run it; for gateway scope, put it in a project library script and
# call run() from a message handler.

import math
import time

from java.util.concurrent import Callable, Executors, TimeUnit

THREADS = 8
ROUNDS = 25


class Call(Callable):
    def __init__(self, fn, args, kwargs):
        self.fn, self.args, self.kwargs = fn, args, kwargs

    def call(self):
        return getattr(system.signal, self.fn)(*self.args, **self.kwargs)


def workload():
    x = [math.sin(k * 0.01) + 0.3 * math.sin(k * 0.37) + 0.001 * k for k in range(20000)]
    sos = system.signal.butter(6, 0.05, output="sos")
    return [
        ("sosfiltfilt", [sos, x], {}),
        ("savgol_filter", [x, 51, 3], {}),
        ("find_peaks", [x], {"prominence": 0.2, "width": 3}),
        ("welch", [x], {"fs": 10.0, "nperseg": 1024}),
        ("decimate", [x, 8], {}),
        ("resample_poly", [x, 3, 7], {}),
        ("medfilt", [x, 7], {}),
        ("ellip", [8, 0.5, 60, [0.1, 0.3], "bandpass"], {"output": "sos"}),
        ("hilbert", [x], {}),
        ("spectrogram", [x], {"nperseg": 256}),
    ]


def run():
    calls = workload()
    reference = [getattr(system.signal, fn)(*a, **k) for fn, a, k in calls]
    pool = Executors.newFixedThreadPool(THREADS)
    t0 = time.time()
    futures = []
    for r in range(ROUNDS):
        for i, (fn, a, k) in enumerate(calls):
            futures.append((i, pool.submit(Call(fn, a, k))))
    mismatches = 0
    errors = 0
    for i, fut in futures:
        try:
            if fut.get() != reference[i]:
                mismatches += 1
        except Exception as e:
            errors += 1
    pool.shutdown()
    pool.awaitTermination(60, TimeUnit.SECONDS)
    seconds = time.time() - t0
    lines = ["Ignition %s, %d threads, %d calls" % (system.util.getVersion(), THREADS, len(futures)),
             "%.1f s, %.0f calls per second" % (seconds, len(futures) / seconds),
             "mismatches %d, errors %d" % (mismatches, errors),
             "PASS" if mismatches == 0 and errors == 0 else "FAIL"]
    return lines


for line in run():
    print line
