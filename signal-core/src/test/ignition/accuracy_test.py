# Signal Processing accuracy test, part 1 of 2.
#
# 1. Export the cases, from the signal-core folder:
#        python src/test/python/random_check.py --export
#    This also writes target/accuracy_test.py, this file with the cases' path filled in.
# 2. Paste target/accuracy_test.py into the Ignition Designer Script Console and run it.
# 3. Check the results in Python against scipy (part 2):
#        python src/test/python/compare_ignition.py --results <path printed at the end>
#
# Besides running every exported case through system.signal, this checks what only a gateway can check:
# other input types (tuples, Java arrays, an ArrayList), keyword calls, that errors arrive as Python
# ValueError and TypeError, zero lag, and the time for 1,000,000-point signals. It streams: cases are read and
# results written one at a time, and progress is logged every 100 cases (logger SignalAccuracyTest).
#
# Works on Ignition 8.1 and 8.3. The imports are needed: Jython does not preload them.

import json
import math
import time

import jarray
from java.io import BufferedReader, BufferedWriter, FileReader, FileWriter
from java.lang import Throwable
from java.util import ArrayList

# random_check.py --export writes target/accuracy_test.py, a copy of this file with CASES set to the cases it
# exported. Paste that copy, not this file.
CASES = None
if CASES is None:
    raise ValueError("Run random_check.py --export and paste target/accuracy_test.py, which has CASES filled in")
OUT = system.file.getTempFile("jsonl")
sig = system.signal
log = system.util.getLogger("SignalAccuracyTest")


def dumps(v):
    # Faster than Jython's pure-Python json for long lists of floats. Complex numbers become
    # {"__complex__": [re, im]}, which compare_ignition.py reads back as complex.
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, float):
        if v != v:
            return "NaN"
        if v == float("inf"):
            return "Infinity"
        if v == float("-inf"):
            return "-Infinity"
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


def run(case):
    kwargs = dict((str(k), v) for k, v in case["kwargs"].items())
    fn = getattr(sig, case["fn"])
    t0 = time.time()
    try:
        result = fn(*case["args"], **kwargs)
        return {"id": case["id"], "ok": result, "ms": (time.time() - t0) * 1000.0}
    except (ValueError, TypeError) as e:
        return {"id": case["id"], "err": "%s: %s" % (type(e).__name__, e), "ms": (time.time() - t0) * 1000.0}
    except Throwable as e:
        return {"id": case["id"], "crash": "Java exception leaked: %s" % e}
    except Exception as e:
        return {"id": case["id"], "crash": "%s: %s" % (type(e).__name__, e)}


# Exported cases, read and written one line at a time so memory stays flat. Progress goes to the Designer log
# (logger SignalAccuracyTest) every 100 cases; if a run ever stalls, the last result in the output file and the
# last log line show where.
reader = BufferedReader(FileReader(CASES))
writer = BufferedWriter(FileWriter(OUT))
header = json.loads(reader.readLine())
header["ignition"] = str(system.util.getVersion())
writer.write(json.dumps(header))
writer.newLine()
counts = {"ok": 0, "err": 0, "crash": 0}
slow = []
done = 0
t_start = time.time()
line = reader.readLine()
while line is not None:
    if line.strip():
        case = json.loads(line)
        reply = run(case)
        for key in counts:
            if key in reply:
                counts[key] += 1
        if reply.get("ms", 0) > 1000:
            slow.append("case %s (%s): %.0f ms" % (case["id"], case["fn"], reply["ms"]))
        writer.write(dumps(reply))
        writer.newLine()
        done += 1
        if done % 100 == 0:
            writer.flush()
            log.info("case %d done, %.1f s" % (done, time.time() - t_start))
    line = reader.readLine()
reader.close()
writer.flush()
t_cases = time.time() - t_start
log.info("all %d cases done in %.1f s; running the gateway checks" % (done, t_cases))

# Checks only a gateway can do
checks = []


def check(label, passed, detail=""):
    checks.append({"check": label, "pass": bool(passed), "detail": detail})


def same(label, got, expected):
    check(label, list(got) == list(expected))


def raises(label, error, call):
    try:
        call()
        check(label, False, "no error")
    except error:
        check(label, True)
    except Throwable as e:
        check(label, False, "Java exception leaked: %s" % e)
    except Exception as e:
        check(label, False, "%s: %s" % (type(e).__name__, e))


x = [float((i * 37) % 101) + 0.25 * i for i in range(500)]
base = sig.detrend(x)
check("result is a list of floats", isinstance(base, list) and all(isinstance(v, float) for v in base))
same("tuple input", sig.detrend(tuple(x)), base)
same("Java double[] input", sig.detrend(jarray.array(x, "d")), base)
same("Java ArrayList input", sig.detrend(ArrayList(x)), base)
same("data passed by keyword", sig.detrend(data=x), base)
same("all arguments positional", sig.detrend(x, -1, "linear", 0, False), base)
ints = [int(v) for v in x]
same("integer input", sig.detrend(ints), sig.detrend([float(v) for v in ints]))
raises("None in data raises ValueError", ValueError, lambda: sig.detrend([1.0, None, 3.0]))
raises("NaN in data raises ValueError", ValueError, lambda: sig.detrend([1.0, float("nan"), 3.0]))
raises("unknown keyword raises TypeError", TypeError, lambda: sig.detrend(x, typ="linear"))
raises("bad trend type raises ValueError", ValueError, lambda: sig.detrend(x, type="quadratic"))
raises("float breakpoint raises TypeError", TypeError, lambda: sig.detrend(x, bp=2.5))

win = sig.get_window(("kaiser", 8.6), 64)
check("get_window: list of floats", isinstance(win, list) and len(win) == 64 and all(isinstance(v, float) for v in win))
same("get_window: tuple and list specs agree", sig.get_window(["kaiser", 8.6], 64), win)
same("get_window: a number is a Kaiser beta", sig.get_window(8.6, 64), win)
same("get_window: keywords", sig.get_window(window="hann", Nx=16, fftbins=False), sig.get_window("hann_symmetric", 16))
raises("get_window: dpss raises ValueError", ValueError, lambda: sig.get_window(("dpss", 2.0), 64))
raises("get_window: Nx=0 raises ValueError", ValueError, lambda: sig.get_window("hann", 0))

b, a = sig.butter(4, 0.2)
check("butter: (b, a) unpacks into two lists of floats",
      len(b) == 5 and len(a) == 5 and all(isinstance(v, float) for v in b + a))
z, p, k = sig.butter(4, 0.2, output="zpk")
check("butter: zpk gives complex poles and a float gain", all(isinstance(v, complex) for v in p) and isinstance(k, float))
sos = sig.butter(6, [0.1, 0.3], "bandpass", output="sos")
check("butter: sos is a list of 6-number rows", len(sos) == 6 and all(len(row) == 6 for row in sos))
w, h = sig.freqz(b, a, worN=64)
check("freqz: complex response, abs() works", len(h) == 64 and isinstance(h[0], complex) and abs(abs(h[0]) - 1) < 1e-12)
w2, h2 = sig.sosfreqz(sig.butter(4, 0.2, output="sos"), worN=64)
check("sosfreqz matches freqz of the same filter", max(abs(x - y) for x, y in zip(h, h2)) < 1e-10)
same("freqz_sos and sosfreqz agree", sig.freqz_sos(sos)[1], sig.sosfreqz(sos)[1])
raises("butter: Wn >= 1 raises ValueError", ValueError, lambda: sig.butter(4, 1.5))
raises("freqz: plot raises TypeError", TypeError, lambda: sig.freqz(b, a, plot=lambda w, h: None))
n, wn = sig.buttord(0.2, 0.3, 3, 40)
check("buttord: N is an int and Wn a float", isinstance(n, int) and isinstance(wn, float), "N=%d" % n)
n2, wn2 = sig.ellipord([0.2, 0.5], [0.1, 0.6], 1, 60)
check("ellipord: band Wn is a list of two", isinstance(wn2, list) and len(wn2) == 2)
check("buttord feeds butter", len(sig.butter(n, wn, output="sos")) == (n + 1) // 2)
raises("buttord: gpass > gstop raises ValueError", ValueError, lambda: sig.buttord(0.2, 0.3, 40, 3))
sosf = sig.butter(4, 0.1, output="sos")
pulse = [math.exp(-0.5 * ((k - 500) / 40.0) ** 2) for k in range(1001)]
yz = sig.sosfiltfilt(sosf, pulse)
check("sosfiltfilt: zero lag (peak stays at sample 500)", yz.index(max(yz)) == 500)
yc = sig.sosfilt(sosf, pulse)
check("sosfilt: causal filter lags (peak after 500)", yc.index(max(yc)) > 500, "peak at %d" % yc.index(max(yc)))
ys, zf = sig.sosfilt(sosf, pulse, zi=sig.sosfilt_zi(sosf))
check("sosfilt with zi returns (y, zf)", len(ys) == 1001 and len(zf) == 2 and len(zf[0]) == 2)
bb, aa = sig.butter(3, 0.2)
same("filtfilt and sosfiltfilt agree for a low order", [round(v, 9) for v in sig.filtfilt(bb, aa, pulse)],
     [round(v, 9) for v in sig.sosfiltfilt(sig.butter(3, 0.2, output="sos"), pulse)])
raises("filtfilt: method='gust' raises ValueError", ValueError, lambda: sig.filtfilt(bb, aa, pulse, method="gust"))
raises("filtfilt: signal shorter than padlen raises ValueError", ValueError, lambda: sig.filtfilt(bb, aa, [1.0, 2.0]))
mi = sig.medfilt([1, 5, 2, 9, 3])
check("medfilt: integers in, integers out", mi == [1, 2, 5, 3, 3] and all(isinstance(v, int) for v in mi))
check("medfilt: removes a spike", sig.medfilt([2.0, 2.0, 2.0, 50.0, 2.0, 2.0, 2.0])[3] == 2.0)
line = [0.5 * k + 3.0 for k in range(40)]
sl = sig.savgol_filter(line, 11, 2)
check("savgol_filter: reproduces a straight line", max(abs(a - b) for a, b in zip(sl, line)) < 1e-12)
slope = sig.savgol_filter(line, 11, 2, deriv=1, delta=0.1)
check("savgol_filter: derivative of the line is its slope", max(abs(v - 5.0) for v in slope) < 1e-9)
check("savgol_coeffs: 5 points, order 2 are -3/35, 12/35, 17/35, ...",
      sig.savgol_coeffs(5, 2)[0] == -3.0 / 35 and sig.savgol_coeffs(5, 2)[2] == 17.0 / 35)
pk, props = sig.find_peaks([0, 2, 1, 3, 1, 3, 1, 2, 0], prominence=1.5)
check("find_peaks: (list of ints, dict) unpacks", pk == [3, 5] and all(isinstance(v, int) for v in pk)
      and props["prominences"] == [3.0, 3.0] and props["left_bases"] == [0, 0], "peaks %s" % pk)
pk, props = sig.find_peaks([0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 0.0], plateau_size=(2, None))
check("find_peaks: plateau middles and sizes", pk == [2, 5] and props["plateau_sizes"] == [3, 2])
check("find_peaks: equal peaks under distance keep the later",
      sig.find_peaks([0.0, 5.0, 0.0, 5.0, 0.0], distance=3)[0] == [3])
pr = sig.peak_prominences([0, 2, 1, 3, 1, 3, 1, 2, 0], [3, 5])
w = sig.peak_widths([0, 2, 1, 3, 1, 3, 1, 2, 0], [3, 5], prominence_data=pr)
check("peak_widths accepts prominence_data from peak_prominences", list(w[0]) == [1.5, 1.5])
raises("find_peaks: distance < 1 raises ValueError", ValueError, lambda: sig.find_peaks([0, 1, 0], distance=0.5))
raises("peak_prominences: float peaks raise TypeError", TypeError, lambda: sig.peak_prominences([0, 1, 0], [1.0]))
tone = [math.sin(2 * math.pi * 12.5 * k / 100.0) for k in range(4096)]
f, pxx = sig.welch(tone, fs=100.0, nperseg=400)
check("welch: a 12.5 Hz tone peaks at 12.5 Hz", f[pxx.index(max(pxx))] == 12.5, "peak at %s Hz" % f[pxx.index(max(pxx))])
f, pxx = sig.periodogram(tone, fs=100.0)
check("periodogram: Parseval (power equals variance)",
      abs(sum(pxx) * 100.0 / len(tone) - sum((v - sum(tone) / len(tone)) ** 2 for v in tone) / len(tone)) < 1e-9)
f, t, sxx = sig.spectrogram(tone, fs=100.0, nperseg=256)
check("spectrogram: (f, t, rows of floats)", len(sxx) == len(f) and len(sxx[0]) == len(t) and isinstance(sxx[0][0], float))
f, t, sc = sig.spectrogram(tone, fs=100.0, nperseg=256, mode="complex")
check("spectrogram: mode='complex' gives complex numbers", isinstance(sc[3][0], complex))
env = [abs(z) for z in sig.hilbert(tone)]
check("hilbert: envelope of a unit tone is 1", max(abs(e - 1.0) for e in env[200:-200]) < 1e-3)
raises("welch: fs <= 0 raises ValueError", ValueError, lambda: sig.welch(tone, fs=0.0))
slow_tone = [math.sin(2 * math.pi * k / 200.0) for k in range(1000)]
up = sig.resample_poly(slow_tone, 3, 2)
check("resample_poly: 1000 samples * 3 / 2 = 1500", len(up) == 1500)
dec = sig.decimate(slow_tone, 4)
check("decimate: keeps a slow tone (to the filter's ripple)", len(dec) == 250
      and max(abs(dec[j] - slow_tone[4 * j]) for j in range(20, 230)) < 0.02)
per = [math.sin(2 * math.pi * 3 * k / 64.0) for k in range(64)]
rs, tn = sig.resample(per, 160, t=[0.0, 1.0])
check("resample: a periodic tone is rebuilt exactly, with new times",
      max(abs(rs[j] - math.sin(2 * math.pi * 3 * j / 160.0)) for j in range(160)) < 1e-12 and tn[1] == 0.4)
raises("resample_poly: a 1-sample 'reflect' raises ValueError (scipy crashes)", ValueError,
       lambda: sig.resample_poly([2.0], 3, 2, padtype="reflect"))
raises("decimate: q=1 with 'fir' raises ValueError", ValueError, lambda: sig.decimate(slow_tone, 1, ftype="fir"))
t0 = time.time()
sig.bessel(8, 0.2, norm="mag", output="sos")
check("bessel(8) design", True, "%.1f ms" % ((time.time() - t0) * 1000.0))

# Long signals last, released straight after, so the run's memory stays small until here.
big = [((i * 7919) % 1000) * 0.001 + i * 1e-6 for i in xrange(1000000)]
t0 = time.time()
r = sig.detrend(big)
check("1,000,000-point linear detrend", len(r) == len(big), "%.0f ms" % ((time.time() - t0) * 1000.0))
t0 = time.time()
r = sig.sosfiltfilt(sig.butter(8, 0.05, output="sos"), big)
check("1,000,000-point sosfiltfilt, order 8", len(r) == len(big), "%.0f ms" % ((time.time() - t0) * 1000.0))
t0 = time.time()
r = sig.find_peaks(big, prominence=0.1, width=2)[0]
check("1,000,000-point find_peaks with prominence and width", len(r) > 0, "%.0f ms, %d peaks"
      % ((time.time() - t0) * 1000.0, len(r)))
t0 = time.time()
r = sig.welch(big, fs=1000.0, nperseg=1024)
check("1,000,000-point welch, 1024-sample segments", len(r[0]) == 513, "%.0f ms" % ((time.time() - t0) * 1000.0))
t0 = time.time()
r = sig.decimate(big, 10)
check("1,000,000-point decimate by 10", len(r) == 100000, "%.0f ms" % ((time.time() - t0) * 1000.0))
big = r = None

writer.write(dumps({"id": "ignition", "checks": checks}))
writer.newLine()
writer.close()
log.info("finished; results in " + OUT)

print("Ignition %s" % header["ignition"])
print("%d cases in %.1f s: %d results, %d errors, %d crashes"
      % (done, t_cases, counts["ok"], counts["err"], counts["crash"]))
for entry in slow:
    print("slow: " + entry)
for c in checks:
    print("%-40s %s %s" % (c["check"], "PASS" if c["pass"] else "FAIL", c["detail"]))
print("Results written to " + OUT)
print("Next: python src/test/python/compare_ignition.py --results " + OUT)
