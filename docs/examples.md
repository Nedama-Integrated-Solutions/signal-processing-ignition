# Examples

Worked recipes for plant data. Each runs in any scope: a Script Console, a gateway timer, a tag change script or
a Perspective binding transform.

## Getting evenly sampled data

Every function assumes evenly spaced samples. Ask the historian for a fixed interval, and fill the gaps it
returns as None (the module refuses None, so a gap never turns silently into NaN):

```python
def history(path, minutes, interval_seconds=1):
    end = system.date.now()
    start = system.date.addMinutes(end, -minutes)
    ds = system.tag.queryTagHistory(paths=[path], startDate=start, endDate=end,
                                    intervalSeconds=interval_seconds, aggregationMode="Average",
                                    returnFormat="Wide")
    times = [ds.getValueAt(r, 0) for r in range(ds.rowCount)]
    values = [ds.getValueAt(r, 1) for r in range(ds.rowCount)]
    last = next((v for v in values if v is not None), 0.0)
    filled = []
    for v in values:                      # carry the last good value across gaps
        last = v if v is not None else last
        filled.append(float(last))
    return times, filled

times, x = history("[default]Line1/Flow", 120)
fs = 1.0                                  # samples per second, from interval_seconds
```

## Clean a noisy signal without lag

A causal filter delays every feature. Filtering forwards and backwards cancels the delay, so steps and peaks stay
where they happened:

```python
sos = system.signal.butter(4, 0.05, fs=fs, output="sos")    # 4th order, 0.05 Hz cut-off
clean = system.signal.sosfiltfilt(sos, x)

rows = [[t, raw, c] for t, raw, c in zip(times, x, clean)]
chart = system.dataset.toDataSet(["t_stamp", "raw", "clean"], rows)
```

Use second-order sections (`output="sos"` with `sosfiltfilt`) rather than `b, a` with `filtfilt`: they stay
accurate at high orders and low cut-offs. Zero-phase filtering needs the samples after each point, so it suits
history windows; for a live value, `sosfilt` with `zi` carries the filter state from one call to the next.

## Choose a filter from a specification

Let the order selection functions find the lowest order that meets the specification, then check the result:

```python
# pass 0.2 Hz within 1 dB, attenuate 1 Hz by 40 dB, at 10 samples per second
N, Wn = system.signal.buttord(0.2, 1.0, 1, 40, fs=10.0)
sos = system.signal.butter(N, Wn, fs=10.0, output="sos")
f, h = system.signal.freqz_sos(sos, worN=[0.2, 1.0], fs=10.0)
gains_db = [20 * math.log10(abs(v)) for v in h]             # import math first
```

## Remove spikes and get a smooth rate of change

A median filter removes single-sample spikes and keeps real steps sharp. A Savitzky-Golay filter then smooths
while keeping peak shapes, and gives the derivative directly:

```python
despiked = system.signal.medfilt(x, 5)
rate = system.signal.savgol_filter(despiked, 31, 3, deriv=1, delta=1.0 / fs)   # units per second
rate_per_minute = [60.0 * r for r in rate]
```

## Find batch peaks and when they happened

Prominence keeps the peaks that stand out from their surroundings; width and distance are in samples:

```python
peaks, props = system.signal.find_peaks(clean, prominence=2.0, distance=int(300 * fs))
for i, p in enumerate(peaks):
    print times[p], clean[p], props["prominences"][i]

widths, heights, left, right = system.signal.peak_widths(clean, peaks, rel_height=0.5)
durations_s = [w / fs for w in widths]                      # duration at half prominence
```

## Find the period of an oscillating loop

A control loop that cycles shows up as a peak in its power spectral density:

```python
f, pxx = system.signal.welch(x, fs=fs, nperseg=2048, nfft=16384)
k = pxx.index(max(pxx[1:]))                                 # skip the zero-frequency bin
period_s = 1.0 / f[k]
```

Each segment (`nperseg` samples) should span several periods of the slowest oscillation you look for, and
`nfft` (zero-padding) places the peak between the coarse bins. With two hours of one-second data, these settings
find periods of 60 s, 300 s and 900 s to within about 1%; with `nperseg=1024` and no `nfft`, a 900 s period
reads as 1024 s, the nearest bin. For how the spectrum changes over time, use `spectrogram`.

## Vibration envelope

The magnitude of the analytic signal is the amplitude envelope, which shows bearing impacts and beating:

```python
envelope = [abs(z) for z in system.signal.hilbert(vibration)]
```

## Align and thin tags

Bring a tag logged every 5 s onto a 1 s time base, or thin a long 1 s pull to one sample per minute without
aliasing:

```python
aligned = system.signal.resample_poly(tag_5s, 5, 1)
per_minute = system.signal.decimate(system.signal.decimate(x_1s, 6), 10)
```

For large factors, decimate in stages, as above (6 then 10 rather than 60): SciPy's documentation recommends
staying at 13 or below per call with the default IIR filter.
