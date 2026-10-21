# Differences from SciPy

Signal Processing follows SciPy 1.18.1: the same function names, parameters, defaults and results. This page
lists every place where it deliberately does something else, and why.

## Everywhere

- **Data must be clean numbers.** None, NaN and infinity in a signal raise ValueError, where SciPy would let NaN
  spread silently through every result. Historian gaps arrive as None, so fill or drop them first (see the
  [examples](examples.md)).
- **Signals are 1-D.** `axis` accepts -1 and 0, which mean the same for a list.
- **Lists and tuples are interchangeable.** Jython and JSON cannot always tell them apart, so a window given as
  `['kaiser', 8.6]` works like `('kaiser', 8.6)`, and a list of numbers is a list of values.
- **Integer arguments must be integers.** Orders, lengths and factors (`N`, `nperseg`, `noverlap`, `nfft`,
  `window_length`, `polyorder`, `num`, `q`, ...) refuse floats with TypeError. SciPy truncates floats in some
  places and fails on them in others.
- **Errors are Python ValueError and TypeError**, raised where SciPy raises them. Messages may be worded
  differently.
- **Script functions only.** There are no expression functions in this version.

## Function by function

| Function | Difference |
| --- | --- |
| `get_window` | dpss windows are not available. |
| `detrend` | An empty breakpoint list means no breakpoints (SciPy raises TypeError). |
| `butter`, `cheby1`, `cheby2`, `ellip`, `bessel` | Orders above 500 are refused (Bessel: above 100). |
| `freqz` | `plot` must be None: there is nothing to plot to in a gateway. |
| `filtfilt` | `method='gust'` is not supported; the default `'pad'` is. |
| `savgol_coeffs`, `savgol_filter` | The least-squares fits are solved exactly. See below. |
| `find_peaks` | Ties under `distance` are ranked by position. See below. Per-sample borders (numpy arrays of `len(x)`) are not available; a list is compared with the peaks as in SciPy. |
| `peak_widths` | `prominence_data` may be lists (SciPy needs numpy arrays). |
| `welch`, `periodogram`, `spectrogram` | `detrend` is `'constant'`, `'linear'` or off; functions are not accepted. |
| `resample`, `resample_poly` | A window given as a list of numbers is used as the window's values (SciPy needs a numpy array). |
| `decimate` | `ftype` is `'iir'` or `'fir'`; dlti systems are not accepted. |

## Where this module does better than SciPy 1.18.1

These came out of testing against SciPy and are kept on purpose.

- **Savitzky-Golay coefficients stay correct at high orders.** SciPy solves the fit with a floating-point least
  squares solver that discards information when the problem is badly conditioned: its coefficients are accurate
  to about 1e-8 for polyorder up to 4, but entirely wrong at polyorder 8 over 201 points. This module solves the
  fit exactly and rounds once. Where SciPy is accurate, results agree.
- **`find_peaks` with `distance` is deterministic.** When equal-height peaks compete, SciPy ranks them with an
  unstable sort whose order for ties depends on the CPU's vector instructions; on quantised plant data that
  changed SciPy's answer in about 3% of tested calls. This module ranks ties by position, keeping the later peak,
  as a stable sort does.
- **Prominences and widths are fast on trending signals.** SciPy's search walks back to the nearest higher
  sample for every peak, which on a rising tag costs time proportional to the square of the length (1.4 s for
  100,000 points). This module finds the same answers in 57 ms.
- **Plain lists work for `welch` and `spectrogram`.** SciPy 1.18.1 raises AttributeError for a list in both.
- **`resample_poly` cannot crash the interpreter.** With `padtype='reflect'` or `'antireflect'` on a 1-sample
  signal, SciPy 1.18.1 kills the Python process ("Floating point exception"); this module raises ValueError.
- **Bessel filters work up to order 100.** SciPy's root finding fails from about order 90.

## How closely results agree

Every function is compared with SciPy on random signals shaped like plant data: each test run checks about 65,000
calls, some 54 million samples of signal, with single signals of up to 100,000 samples. The same comparison runs on
the module inside Ignition 8.1 and 8.3, along with 1,000,000-point signals, a gateway-scope check of every function,
and a concurrency test in which 8 threads must reproduce single-threaded results exactly.

- **Identical to the last bit:** `lfilter` (IIR), `sosfilt`, `sosfiltfilt`, `filtfilt`, `lfilter_zi`,
  `sosfilt_zi`, `medfilt`, `find_peaks`, `peak_prominences`, `peak_widths`, and the interior of `savgol_filter`.
- **Within a few roundings of the result's size:** `detrend`, `get_window`, `freqz`, the FFT-based functions
  (`welch`, `periodogram`, `spectrogram`, `hilbert`, `resample`) and `resample_poly`.
- **Within the problem's own sensitivity:** filter designs and order selection, and `decimate`'s IIR filter. Where
  a design is ill-conditioned, a tiny change in the input moves SciPy's own answer by more than any difference
  between the two.
