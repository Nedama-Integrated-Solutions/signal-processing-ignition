# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-19

First release, for Ignition 8.1 (8.1.33 or newer) and 8.3 (8.3.3 or newer): 33 `scipy.signal` functions under
`system.signal`, with SciPy 1.18.1's names, parameters, defaults and results, in every scope.

### Added
- Detrending and windows: `detrend`, `get_window` (every window except dpss, with the `_symmetric` and
  `_periodic` suffixes).
- Filter design: `butter`, `cheby1`, `cheby2`, `ellip`, `bessel` (every band type, 'ba', 'zpk' and 'sos'
  output, analog designs, `fs`); order selection `buttord`, `cheb1ord`, `cheb2ord`, `ellipord`; responses
  `freqz`, `freqz_sos` and its older name `sosfreqz`. Zeros, poles and responses are Python complex numbers.
- Filtering: `lfilter`, `lfilter_zi`, `sosfilt`, `sosfilt_zi`, `filtfilt`, `sosfiltfilt`.
- Smoothing: `savgol_coeffs`, `savgol_filter` (all five edge modes, derivatives), `medfilt`.
- Peaks: `find_peaks` (height, threshold, distance, prominence, width, plateau size), `peak_prominences`,
  `peak_widths`.
- Spectra: `welch`, `periodogram`, `spectrogram` (modes psd, complex, magnitude, angle, phase), `hilbert`.
- Resampling: `resample`, `resample_poly` (every padtype), `decimate`.
- Tests: random comparison with SciPy, property checks, invalid-call checks, an accuracy test run inside
  Ignition and a gateway-scope spot check.

### Differences from SciPy
- Lists and tuples are interchangeable; signals are 1-D; None, NaN and infinity in the data raise ValueError.
- Where SciPy needs a numpy array, lists work: windows given as values, `prominence_data`, and `welch` and
  `spectrogram` input (SciPy 1.18.1 fails on plain lists there).
- Savitzky-Golay least squares are solved exactly, so coefficients stay correct at high orders where SciPy's lose
  accuracy.
- Equal-height peaks competing under `distance` are ranked by position, the later first (SciPy's choice depends
  on the CPU).
- `resample_poly` with padtype 'reflect' or 'antireflect' on one sample raises ValueError (SciPy crashes the
  Python process).
- Not supported: dpss windows, `freqz`'s `plot`, `filtfilt`'s method='gust', detrend functions passed as
  callables, dlti systems in `decimate`. Integer arguments (orders, lengths, factors) must be integers.

[0.1.0]: https://github.com/Nedama-Integrated-Solutions/signal-processing-ignition/releases/tag/v0.1.0
