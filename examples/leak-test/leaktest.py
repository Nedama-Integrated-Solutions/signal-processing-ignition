# Pressure rate-of-rise leak test, with a simulator for testing.
#
# A chamber is pumped down and isolated; a tight chamber's pressure barely rises, a leaking one rises at a steady
# rate. Leak rate = chamber volume x pressure rise rate (mbar.L/s). system.signal.detrend does the analysis:
# a linear detrend gives the fitted line (the rise rate) and the noise, a constant detrend the total variation
# (for R squared), and a two-piece detrend with a breakpoint compares the rise rate in the two halves of the test,
# which tells a leak (steady rate) from outgassing (a rate that is still falling).
#
# Install as a Project Library script named leaktest. See README.md for the tags, events and view.

import math
import random

from java.lang import Thread

ROOT = "[default]LeakTest"
PRESSURE = ROOT + "/Sim/Pressure"         # point at a real chamber pressure tag to test a real chamber
CYCLE_S = 840                             # simulator: one test every 14 minutes
ISOLATE_AT_S = 60                         # pump-down, then the chamber is isolated
TEST_S = 600                              # test length (the PLC's test-active signal is high this long)
log = system.util.getLogger("LeakTest")


# Analysis

def segment_sigma(noise, m, dt):
    """Standard uncertainty of a least-squares slope over m samples dt apart, for white noise."""
    return noise / (dt * math.sqrt(m * (m * m - 1) / 12.0))


def analyse(x, dt, volume_l, limit, min_drop=0.2):
    """
    Leak test result for pressures x (mbar) sampled every dt seconds after settling. Returns a dict:
    leak (mbar.L/s), rise (mbar/min), noise (mbar), r2, ratio (last/first quarter's rise rate), detection
    (mbar.L/s), sigma (standard uncertainty of leak, mbar.L/s), verdict and message.

    Outgassing shows as a rise rate that is still falling. The first and last quarters get their own lines
    (detrend with breakpoints at the quarter points); the test is inconclusive only when the first quarter's rate
    exceeds the last's by more than 3 standard uncertainties of the difference and by at least min_drop of it,
    so noise on a small leak cannot trigger it.
    """
    n = len(x)
    if n < 30:
        return {"verdict": "ERROR", "message": "only %d samples after settling" % n}
    residual = system.signal.detrend(x)                              # x minus its best straight line
    line = [a - r for a, r in zip(x, residual)]
    duration = (n - 1) * dt
    slope = (line[-1] - line[0]) / duration                          # mbar/s
    leak = slope * volume_l
    sse = sum(r * r for r in residual)
    noise = math.sqrt(sse / (n - 2))
    centred = system.signal.detrend(x, type="constant")              # x minus its mean
    sst = sum(c * c for c in centred)
    r2 = 1 - sse / sst if sst > 0 else 0.0
    sigma = segment_sigma(noise, n, dt) * volume_l
    detection = 3 * sigma
    q = n // 4
    pieces = system.signal.detrend(x, bp=[q, n - q])                 # a line per first quarter, middle, last quarter
    fit3 = [a - r for a, r in zip(x, pieces)]
    early = (fit3[q - 1] - fit3[0]) / ((q - 1) * dt)
    late = (fit3[n - 1] - fit3[n - q]) / ((q - 1) * dt)
    ratio = late / early if early > 0 else None
    drop_sigma = math.sqrt(2) * segment_sigma(noise, q, dt)
    outgassing = early - late > 3 * drop_sigma and early - late > min_drop * max(early, 0.0)
    if leak <= detection:
        verdict, message = "PASS", "no measurable rise (below %.3g mbar.L/s)" % detection
    elif outgassing:
        verdict, message = ("INCONCLUSIVE", "rise rate still falling (last/first quarter %.2f): outgassing, "
                            "lengthen the settle time" % ratio)
    elif leak <= limit:
        verdict, message = "PASS", "leak %.3g <= limit %.3g mbar.L/s" % (leak, limit)
    else:
        verdict, message = "FAIL", "leak %.3g > limit %.3g mbar.L/s" % (leak, limit)
    return {"leak": leak, "rise": slope * 60.0, "noise": noise, "r2": r2, "ratio": ratio, "detection": detection,
            "sigma": sigma, "verdict": verdict, "message": message, "points": n, "line": line}


# Simulator: a 500 L chamber with a known leak, outgassing that decays, and sensor noise

def sim_pressure(tt, true_leak, volume_l, rng, noise=0.002, outgas=0.02, tau=40.0):
    """Chamber pressure (mbar) tt seconds after isolation."""
    p = 0.05 + true_leak / volume_l * tt + outgas * (1 - math.exp(-tt / tau))
    return p + rng.gauss(0, noise)


def simulate_step():
    """Gateway timer, every 1000 ms: one simulator step."""
    paths = [ROOT + "/Sim/Enabled", ROOT + "/State/SimStart", ROOT + "/Sim/TrueLeak", ROOT + "/Config/Volume_L",
             ROOT + "/Sim/TestActive"]
    enabled, sim_start, true_leak, volume, active = [q.value for q in system.tag.readBlocking(paths)]
    if not enabled:
        return
    now = system.date.now()
    if sim_start is None:
        system.tag.writeBlocking([ROOT + "/State/SimStart"], [now])
        return
    phase = system.date.secondsBetween(sim_start, now) % CYCLE_S
    testing = ISOLATE_AT_S <= phase < ISOLATE_AT_S + TEST_S
    rng = random.Random()
    if testing and not active:
        true_leak = random.choice([0.01, 0.03, 0.05, 0.08, 0.15, 0.3])   # a new truth for each test
        system.tag.writeBlocking([ROOT + "/Sim/TrueLeak"], [true_leak])
    p = sim_pressure(phase - ISOLATE_AT_S, true_leak, volume, rng) if testing else 0.05 + rng.gauss(0, 0.002)
    system.tag.writeBlocking([PRESSURE, ROOT + "/Sim/TestActive"], [p, testing])


# Running a test from history

def history(start, end, dt):
    ds = system.tag.queryTagHistory(paths=[PRESSURE], startDate=start, endDate=end, intervalSeconds=dt,
                                    aggregationMode="Average", returnFormat="Wide")
    times = [ds.getValueAt(r, 0) for r in range(ds.rowCount)]
    values = [ds.getValueAt(r, 1) for r in range(ds.rowCount)]
    last = next((v for v in values if v is not None), None)
    filled = []
    for v in values:
        last = v if v is not None else last
        filled.append(float(last) if last is not None else None)
    return times, filled


def evaluate(start, end):
    """Analyses the test between start and end (the test-active signal's edges) and writes the results."""
    cfg = [q.value for q in system.tag.readBlocking([ROOT + "/Config/Volume_L", ROOT + "/Config/Limit",
                                                     ROOT + "/Config/SettleSeconds", ROOT + "/Config/IntervalSeconds",
                                                     ROOT + "/Sim/TrueLeak"])]
    volume, limit, settle, dt, true_leak = cfg
    times, x = history(system.date.addSeconds(start, settle), end, dt)
    pairs = [(t, v) for t, v in zip(times, x) if v is not None]
    r = analyse([v for _, v in pairs], float(dt), volume, limit)
    if r["verdict"] == "ERROR":
        system.tag.writeBlocking([ROOT + "/Results/Verdict", ROOT + "/Results/Message"], ["ERROR", r["message"]])
        log.warn("Leak test %s to %s: %s" % (start, end, r["message"]))
        return r
    trace = system.dataset.toDataSet(["t_stamp", "pressure", "fit"],
                                     [[t, v, f] for (t, v), f in zip(pairs, r["line"])])
    names = ["LeakRate", "RiseRate", "Noise", "R2", "RateRatio", "DetectionLimit", "Sigma", "Verdict", "Message",
             "TestStart", "TestEnd", "Points", "LastTrace"]
    values = [r["leak"], r["rise"], r["noise"], r["r2"], r["ratio"], r["detection"], r["sigma"], r["verdict"],
              r["message"], start, end, r["points"], trace]
    system.tag.writeBlocking([ROOT + "/Results/" + n for n in names], values)
    error_sigmas = (r["leak"] - true_leak) / r["sigma"] if r["sigma"] > 0 else 0.0
    expected = "PASS" if true_leak <= limit else "FAIL"              # meaningful while the simulator runs
    row = [end, true_leak, r["leak"], r["sigma"], error_sigmas, r["verdict"], expected]
    old = system.tag.readBlocking([ROOT + "/Results/Log"])[0].value
    rows = [[old.getValueAt(i, c) for c in range(old.columnCount)] for i in range(old.rowCount)] if old else []
    rows = (rows + [row])[-100:]
    log_ds = system.dataset.toDataSet(["test_end", "true_leak", "measured", "sigma", "error_sigmas", "verdict",
                                       "expected"], rows)
    system.tag.writeBlocking([ROOT + "/Results/Log"], [log_ds])
    log.info("Leak test %s: %s" % (end, r["message"]))
    return r


def evaluate_later(start, end, delay_s=15):
    """Waits for the historian to store the last samples (store and forward), then evaluates."""
    def task():
        Thread.sleep(delay_s * 1000)
        try:
            evaluate(start, end)
        except Exception as e:
            log.error("Leak test evaluation failed: %s" % e)
    system.util.invokeAsynchronous(task)


# Setup

def setup(history_provider):
    """Creates the LeakTest tags. history_provider: your tag history provider's name."""
    def atomic(name, data_type, value=None, **extra):
        tag = {"name": name, "tagType": "AtomicTag", "valueSource": "memory", "dataType": data_type}
        if value is not None:
            tag["value"] = value
        tag.update(extra)
        return tag

    folders = {
        "Config": [atomic("Volume_L", "Float8", 500.0), atomic("Limit", "Float8", 0.1),
                   atomic("SettleSeconds", "Int4", 180), atomic("IntervalSeconds", "Int4", 1)],
        "Sim": [atomic("Enabled", "Boolean", True), atomic("TrueLeak", "Float8", 0.05),
                atomic("TestActive", "Boolean", False),
                atomic("Pressure", "Float8", 0.05, historyEnabled=True, historyProvider=history_provider,
                       historicalDeadbandStyle="Discrete", historicalDeadband=0.0)],
        "State": [atomic("SimStart", "DateTime"), atomic("TestStart", "DateTime")],
        "Results": [atomic(n, "Float8") for n in ("LeakRate", "RiseRate", "Noise", "R2", "RateRatio",
                                                  "DetectionLimit", "Sigma")]
                   + [atomic("Verdict", "String", ""), atomic("Message", "String", ""), atomic("TestStart", "DateTime"),
                      atomic("TestEnd", "DateTime"), atomic("Points", "Int4", 0), atomic("LastTrace", "DataSet"),
                      atomic("Log", "DataSet")],
    }
    root = {"name": "LeakTest", "tagType": "Folder",
            "tags": [{"name": k, "tagType": "Folder", "tags": v} for k, v in folders.items()]}
    return system.tag.configure("[default]", [root], "m")
