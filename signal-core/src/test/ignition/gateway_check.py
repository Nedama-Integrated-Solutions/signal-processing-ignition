# Signal Processing gateway-scope spot check.
#
# The accuracy test runs in the Designer. This checks that system.signal is registered in gateway scope too and
# that every function works there, with known answers where there is one.
#
# 1. In a project, create a Project Library script named signalcheck and paste this file into it.
# 2. Under Gateway Events, Message, add a handler named signalCheck whose body is:
#        return signalcheck.run()
#    Save the project.
# 3. In the Designer Script Console, run:
#        for row in system.util.sendRequest(system.util.getProjectName(), "signalCheck"):
#            print row
#    Every row should end in PASS.

import math

from java.lang import Throwable


def run():
    sig = system.signal
    results = []

    def check(label, call, expect=None):
        try:
            value = call()
            ok = True if expect is None else expect(value)
            results.append("%-48s %s" % (label, "PASS" if ok else "FAIL"))
        except Exception as e:
            results.append("%-48s FAIL %s: %s" % (label, type(e).__name__, e))
        except Throwable as e:
            results.append("%-48s FAIL Java: %s" % (label, e))

    tone = [math.sin(2 * math.pi * 12.5 * k / 100.0) for k in range(2048)]
    pulse = [math.exp(-0.5 * ((k - 500) / 40.0) ** 2) for k in range(1001)]
    ramp = [0.5 * k + 3.0 for k in range(40)]
    sos = sig.butter(4, 0.1, output="sos")
    b, a = sig.butter(3, 0.2)

    results.append("Ignition %s, gateway scope" % system.util.getVersion())
    check("detrend", lambda: sig.detrend(ramp), lambda r: max(abs(v) for v in r) < 1e-9)
    check("get_window", lambda: sig.get_window("hann", 8), lambda r: len(r) == 8)
    check("butter", lambda: sig.butter(4, 0.2), lambda r: len(r[0]) == 5)
    check("cheby1", lambda: sig.cheby1(4, 1, 0.2, output="sos"), lambda r: len(r) == 2)
    check("cheby2", lambda: sig.cheby2(4, 40, 0.2, output="sos"), lambda r: len(r) == 2)
    check("ellip", lambda: sig.ellip(4, 1, 40, 0.2, output="sos"), lambda r: len(r) == 2)
    check("bessel", lambda: sig.bessel(4, 0.2, output="sos"), lambda r: len(r) == 2)
    check("buttord", lambda: sig.buttord(0.2, 0.3, 1, 40), lambda r: isinstance(r[0], int))
    check("cheb1ord", lambda: sig.cheb1ord(0.2, 0.3, 1, 40), lambda r: isinstance(r[0], int))
    check("cheb2ord", lambda: sig.cheb2ord(0.2, 0.3, 1, 40), lambda r: isinstance(r[0], int))
    check("ellipord", lambda: sig.ellipord(0.2, 0.3, 1, 40), lambda r: isinstance(r[0], int))
    check("freqz", lambda: sig.freqz(b, a), lambda r: abs(abs(r[1][0]) - 1.0) < 1e-9)
    check("freqz_sos", lambda: sig.freqz_sos(sos), lambda r: abs(abs(r[1][0]) - 1.0) < 1e-9)
    check("sosfreqz", lambda: sig.sosfreqz(sos), lambda r: abs(abs(r[1][0]) - 1.0) < 1e-9)
    check("lfilter", lambda: sig.lfilter(b, a, pulse), lambda r: len(r) == 1001)
    check("lfilter_zi", lambda: sig.lfilter_zi(b, a), lambda r: len(r) == 3)
    check("sosfilt", lambda: sig.sosfilt(sos, pulse), lambda r: r.index(max(r)) > 500)
    check("sosfilt_zi", lambda: sig.sosfilt_zi(sos), lambda r: len(r) == 2)
    check("filtfilt", lambda: sig.filtfilt(b, a, pulse), lambda r: r.index(max(r)) == 500)
    check("sosfiltfilt (zero lag)", lambda: sig.sosfiltfilt(sos, pulse), lambda r: r.index(max(r)) == 500)
    check("savgol_coeffs (-3/35)", lambda: sig.savgol_coeffs(5, 2), lambda r: r[0] == -3.0 / 35)
    check("savgol_filter", lambda: sig.savgol_filter(ramp, 11, 2), lambda r: max(abs(x - y) for x, y in zip(r, ramp)) < 1e-12)
    check("medfilt", lambda: sig.medfilt([1, 5, 2, 9, 3]), lambda r: r == [1, 2, 5, 3, 3])
    check("find_peaks", lambda: sig.find_peaks([0, 2, 1, 3, 1, 3, 1, 2, 0], prominence=1.5), lambda r: r[0] == [3, 5])
    check("peak_prominences", lambda: sig.peak_prominences([0, 2, 1, 3, 1, 3, 1, 2, 0], [3, 5]),
          lambda r: list(r[0]) == [3.0, 3.0])
    check("peak_widths", lambda: sig.peak_widths([0, 2, 1, 3, 1, 3, 1, 2, 0], [3, 5]), lambda r: len(r) == 4)
    check("welch (12.5 Hz tone)", lambda: sig.welch(tone, fs=100.0, nperseg=400),
          lambda r: r[0][r[1].index(max(r[1]))] == 12.5)
    check("periodogram", lambda: sig.periodogram(tone, fs=100.0), lambda r: len(r[0]) == 1025)
    check("spectrogram", lambda: sig.spectrogram(tone, fs=100.0, nperseg=256), lambda r: len(r[2]) == len(r[0]))
    check("hilbert (unit envelope)", lambda: sig.hilbert(tone),
          lambda r: max(abs(abs(z) - 1.0) for z in r[200:-200]) < 1e-3)
    check("resample", lambda: sig.resample(ramp, 80), lambda r: len(r) == 80)
    check("resample_poly", lambda: sig.resample_poly(tone, 3, 2), lambda r: len(r) == 3072)
    check("decimate", lambda: sig.decimate(tone, 4), lambda r: len(r) == 512)
    try:
        sig.detrend([1.0, None, 2.0])
        results.append("%-48s FAIL no error" % "None in data raises ValueError")
    except ValueError:
        results.append("%-48s PASS" % "None in data raises ValueError")
    return results
