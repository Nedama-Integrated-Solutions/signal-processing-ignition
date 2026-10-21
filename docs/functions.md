# Function reference

All functions are under `system.signal` and work in every scope. Names, parameters, defaults and results follow SciPy 1.18.1; positional and keyword calls both work. Signals are lists (or tuples, or Java arrays) of numbers, evenly sampled. See [Differences from SciPy](differences.md) for the few deliberate departures.

## Detrending and windows

### detrend

```python
system.signal.detrend(data, axis=-1, type='linear', bp=0, overwrite_data=False)
```

Remove a linear or constant trend from a signal. Same as scipy.signal.detrend for 1-D data.

| Parameter | Description |
| --- | --- |
| `data` | The signal, a list of numbers. None, NaN and infinity are not allowed. |
| `axis` | -1 (default) or 0. Signals are 1-D, so both mean the same. |
| `type` | 'linear' (default) subtracts a least-squares line; 'constant' subtracts the mean. |
| `bp` | Breakpoints for 'linear': sample indices where a new line starts. An integer or a list. |
| `overwrite_data` | Accepted for compatibility with scipy. The input is never modified. |

**Returns:** The detrended signal, a list of floats.

### get_window

```python
system.signal.get_window(window, Nx, fftbins=True)
```

Return a window of Nx points. Same as scipy.signal.get_window. Windows: hann, hamming, blackman, blackmanharris, nuttall, flattop, boxcar, triang, bartlett, parzen, bohman, barthann, cosine, lanczos, tukey, kaiser, kaiser_bessel_derived, gaussian, general_gaussian, general_hamming, general_cosine, exponential, taylor, chebwin.

| Parameter | Description |
| --- | --- |
| `window` | A name such as 'hann', a tuple with parameters such as ('kaiser', 8.6) or ('tukey', 0.25), or a number (a Kaiser beta). A name ending in _symmetric or _periodic overrides fftbins. |
| `Nx` | Number of points, a positive integer. |
| `fftbins` | True (default): periodic window, for spectral analysis. False: symmetric window, for filter design. |

**Returns:** The window, a list of floats.

## Filter design

### butter

```python
system.signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
```

Design a Butterworth filter (maximally flat passband). Same as scipy.signal.butter.

| Parameter | Description |
| --- | --- |
| `N` | Filter order, a whole number. |
| `Wn` | Critical frequency: a number for low-pass and high-pass, [low, high] for band-pass and band-stop. Normalised to 0..1 (1 = Nyquist) unless fs is given, then in the same units as fs. |
| `btype` | 'lowpass' (default), 'highpass', 'bandpass' or 'bandstop'. |
| `analog` | True for an analog filter (Wn in rad/s). Default False. |
| `output` | 'ba' (default), 'zpk' or 'sos'. Use 'sos' for filtering: it stays accurate at high orders. |
| `fs` | Sampling frequency of the system, in the units of Wn. Default None (Wn normalised). |

**Returns:** (b, a), (z, p, k) or a list of sos rows, depending on output. z and p hold complex numbers.

### cheby1

```python
system.signal.cheby1(N, rp, Wn, btype='low', analog=False, output='ba', fs=None)
```

Design a Chebyshev type I filter (ripple in the passband). Same as scipy.signal.cheby1.

| Parameter | Description |
| --- | --- |
| `N` | Filter order, a whole number. |
| `rp` | Passband ripple in dB (positive). |
| `Wn` | Critical frequency: a number for low-pass and high-pass, [low, high] for band-pass and band-stop. Normalised to 0..1 (1 = Nyquist) unless fs is given, then in the same units as fs. |
| `btype` | 'lowpass' (default), 'highpass', 'bandpass' or 'bandstop'. |
| `analog` | True for an analog filter (Wn in rad/s). Default False. |
| `output` | 'ba' (default), 'zpk' or 'sos'. Use 'sos' for filtering: it stays accurate at high orders. |
| `fs` | Sampling frequency of the system, in the units of Wn. Default None (Wn normalised). |

**Returns:** (b, a), (z, p, k) or a list of sos rows, depending on output. z and p hold complex numbers.

### cheby2

```python
system.signal.cheby2(N, rs, Wn, btype='low', analog=False, output='ba', fs=None)
```

Design a Chebyshev type II filter (ripple in the stopband). Same as scipy.signal.cheby2.

| Parameter | Description |
| --- | --- |
| `N` | Filter order, a whole number. |
| `rs` | Minimum stopband attenuation in dB (positive). |
| `Wn` | Frequency where the attenuation first reaches rs: a number, or [low, high] for band filters. Normalised to 0..1 unless fs is given. |
| `btype` | 'lowpass' (default), 'highpass', 'bandpass' or 'bandstop'. |
| `analog` | True for an analog filter (Wn in rad/s). Default False. |
| `output` | 'ba' (default), 'zpk' or 'sos'. Use 'sos' for filtering: it stays accurate at high orders. |
| `fs` | Sampling frequency of the system, in the units of Wn. Default None (Wn normalised). |

**Returns:** (b, a), (z, p, k) or a list of sos rows, depending on output. z and p hold complex numbers.

### ellip

```python
system.signal.ellip(N, rp, rs, Wn, btype='low', analog=False, output='ba', fs=None)
```

Design an elliptic (Cauer) filter, the sharpest transition for its order. Same as scipy.signal.ellip.

| Parameter | Description |
| --- | --- |
| `N` | Filter order, a whole number. |
| `rp` | Passband ripple in dB (positive). |
| `rs` | Minimum stopband attenuation in dB (positive). |
| `Wn` | Critical frequency: a number for low-pass and high-pass, [low, high] for band-pass and band-stop. Normalised to 0..1 (1 = Nyquist) unless fs is given, then in the same units as fs. |
| `btype` | 'lowpass' (default), 'highpass', 'bandpass' or 'bandstop'. |
| `analog` | True for an analog filter (Wn in rad/s). Default False. |
| `output` | 'ba' (default), 'zpk' or 'sos'. Use 'sos' for filtering: it stays accurate at high orders. |
| `fs` | Sampling frequency of the system, in the units of Wn. Default None (Wn normalised). |

**Returns:** (b, a), (z, p, k) or a list of sos rows, depending on output. z and p hold complex numbers.

### bessel

```python
system.signal.bessel(N, Wn, btype='low', analog=False, output='ba', norm='phase', fs=None)
```

Design a Bessel filter (flattest group delay, so the least overshoot). Same as scipy.signal.bessel.

| Parameter | Description |
| --- | --- |
| `N` | Filter order, a whole number. |
| `Wn` | Critical frequency: a number for low-pass and high-pass, [low, high] for band-pass and band-stop. Normalised to 0..1 (1 = Nyquist) unless fs is given, then in the same units as fs. |
| `btype` | 'lowpass' (default), 'highpass', 'bandpass' or 'bandstop'. |
| `analog` | True for an analog filter (Wn in rad/s). Default False. |
| `output` | 'ba' (default), 'zpk' or 'sos'. Use 'sos' for filtering: it stays accurate at high orders. |
| `norm` | 'phase' (default), 'delay' or 'mag': what Wn means. |
| `fs` | Sampling frequency of the system, in the units of Wn. Default None (Wn normalised). |

**Returns:** (b, a), (z, p, k) or a list of sos rows, depending on output. z and p hold complex numbers.

## Filter order selection

### buttord

```python
system.signal.buttord(wp, ws, gpass, gstop, analog=False, fs=None)
```

Lowest Butterworth order that meets a specification, and its natural frequency. Same as scipy.signal.buttord.

| Parameter | Description |
| --- | --- |
| `wp` | Passband edge: a number for low-pass and high-pass, [low, high] for band filters. Normalised to 0..1 (1 = Nyquist) unless fs is given, then in the units of fs. |
| `ws` | Stopband edge, in the same form and units as wp. |
| `gpass` | Maximum loss in the passband, in dB. |
| `gstop` | Minimum attenuation in the stopband, in dB. |
| `analog` | True for an analog filter (frequencies in rad/s). Default False. |
| `fs` | Sampling frequency, in the units of wp and ws. Default None (normalised). |

**Returns:** (N, Wn): the order (an int) and the natural frequency (a number, or [low, high]), ready for butter(N, ..., Wn).

### cheb1ord

```python
system.signal.cheb1ord(wp, ws, gpass, gstop, analog=False, fs=None)
```

Lowest Chebyshev type I order that meets a specification, and its natural frequency. Same as scipy.signal.cheb1ord.

| Parameter | Description |
| --- | --- |
| `wp` | Passband edge: a number for low-pass and high-pass, [low, high] for band filters. Normalised to 0..1 (1 = Nyquist) unless fs is given, then in the units of fs. |
| `ws` | Stopband edge, in the same form and units as wp. |
| `gpass` | Maximum loss in the passband, in dB. |
| `gstop` | Minimum attenuation in the stopband, in dB. |
| `analog` | True for an analog filter (frequencies in rad/s). Default False. |
| `fs` | Sampling frequency, in the units of wp and ws. Default None (normalised). |

**Returns:** (N, Wn): the order (an int) and the natural frequency (a number, or [low, high]), ready for cheby1(N, ..., Wn).

### cheb2ord

```python
system.signal.cheb2ord(wp, ws, gpass, gstop, analog=False, fs=None)
```

Lowest Chebyshev type II order that meets a specification, and its natural frequency. Same as scipy.signal.cheb2ord.

| Parameter | Description |
| --- | --- |
| `wp` | Passband edge: a number for low-pass and high-pass, [low, high] for band filters. Normalised to 0..1 (1 = Nyquist) unless fs is given, then in the units of fs. |
| `ws` | Stopband edge, in the same form and units as wp. |
| `gpass` | Maximum loss in the passband, in dB. |
| `gstop` | Minimum attenuation in the stopband, in dB. |
| `analog` | True for an analog filter (frequencies in rad/s). Default False. |
| `fs` | Sampling frequency, in the units of wp and ws. Default None (normalised). |

**Returns:** (N, Wn): the order (an int) and the natural frequency (a number, or [low, high]), ready for cheby2(N, ..., Wn).

### ellipord

```python
system.signal.ellipord(wp, ws, gpass, gstop, analog=False, fs=None)
```

Lowest elliptic order that meets a specification, and its natural frequency. Same as scipy.signal.ellipord.

| Parameter | Description |
| --- | --- |
| `wp` | Passband edge: a number for low-pass and high-pass, [low, high] for band filters. Normalised to 0..1 (1 = Nyquist) unless fs is given, then in the units of fs. |
| `ws` | Stopband edge, in the same form and units as wp. |
| `gpass` | Maximum loss in the passband, in dB. |
| `gstop` | Minimum attenuation in the stopband, in dB. |
| `analog` | True for an analog filter (frequencies in rad/s). Default False. |
| `fs` | Sampling frequency, in the units of wp and ws. Default None (normalised). |

**Returns:** (N, Wn): the order (an int) and the natural frequency (a number, or [low, high]), ready for ellip(N, ..., Wn).

## Frequency response

### freqz

```python
system.signal.freqz(b, a=1, worN=512, whole=False, plot=None, fs=6.283185307179586, include_nyquist=False)
```

Frequency response of a digital filter. Same as scipy.signal.freqz.

| Parameter | Description |
| --- | --- |
| `b` | Numerator coefficients. |
| `a` | Denominator coefficients. Default 1. |
| `worN` | Number of frequencies (default 512), or the frequencies to evaluate, in the units of fs. |
| `whole` | True to cover 0 to fs instead of 0 to fs/2. Default False. |
| `plot` | Not supported: must be None. Plot w and h from the result. |
| `fs` | Sampling frequency, in the units w is returned in. Default 2*pi (rad/sample). |
| `include_nyquist` | True to include fs/2 as the last frequency when worN is a count. Default False. |

**Returns:** (w, h): frequencies (floats) and the complex response at each.

### freqz_sos

```python
system.signal.freqz_sos(sos, worN=512, whole=False, fs=6.283185307179586)
```

Frequency response of a filter in second-order sections. Same as scipy.signal.freqz_sos.

| Parameter | Description |
| --- | --- |
| `sos` | Second-order sections, rows of [b0, b1, b2, 1, a1, a2]. |
| `worN` | Number of frequencies (default 512), or the frequencies to evaluate, in the units of fs. |
| `whole` | True to cover 0 to fs instead of 0 to fs/2. Default False. |
| `fs` | Sampling frequency, in the units w is returned in. Default 2*pi (rad/sample). |

**Returns:** (w, h): frequencies (floats) and the complex response at each.

### sosfreqz

```python
system.signal.sosfreqz(sos, worN, whole, fs)
```

Frequency response of a filter in second-order sections. Older name of freqz_sos, kept for compatibility.

| Parameter | Description |
| --- | --- |
| `sos` | Second-order sections, rows of [b0, b1, b2, 1, a1, a2]. |
| `worN` | Number of frequencies (default 512), or the frequencies to evaluate, in the units of fs. |
| `whole` | True to cover 0 to fs instead of 0 to fs/2. Default False. |
| `fs` | Sampling frequency, in the units w is returned in. Default 2*pi (rad/sample). |

**Returns:** (w, h): frequencies (floats) and the complex response at each.

## Filtering

### lfilter

```python
system.signal.lfilter(b, a, x, axis=-1, zi=None)
```

Filter a signal with b/a coefficients (causal: the output lags the input). Same as scipy.signal.lfilter.

| Parameter | Description |
| --- | --- |
| `b` | Numerator coefficients. |
| `a` | Denominator coefficients (a number or a list). a[0] must not be 0. |
| `x` | The signal, a list of numbers, evenly sampled. None, NaN and infinity are not allowed. |
| `axis` | -1 (default) or 0. Signals are 1-D, so both mean the same. |
| `zi` | Initial conditions, max(len(a), len(b)) - 1 numbers. Default None (start from rest). |

**Returns:** The filtered signal, a list of floats; with zi, a tuple (y, zf) with the final conditions.

### lfilter_zi

```python
system.signal.lfilter_zi(b, a)
```

Initial conditions for lfilter that start it in steady state for a step input. Same as scipy.signal.lfilter_zi.

| Parameter | Description |
| --- | --- |
| `b` | Numerator coefficients. |
| `a` | Denominator coefficients (a number or a list). a[0] must not be 0. |

**Returns:** A list of max(len(a), len(b)) - 1 floats. Multiply by the first sample to start on a signal.

### sosfilt

```python
system.signal.sosfilt(sos, x, axis=-1, zi=None)
```

Filter a signal with second-order sections (causal; accurate at any order). Same as scipy.signal.sosfilt.

| Parameter | Description |
| --- | --- |
| `sos` | Second-order sections, rows of [b0, b1, b2, 1, a1, a2], as butter(..., output='sos') returns. |
| `x` | The signal, a list of numbers, evenly sampled. None, NaN and infinity are not allowed. |
| `axis` | -1 (default) or 0. Signals are 1-D, so both mean the same. |
| `zi` | Initial conditions, one [z0, z1] row per section. Default None (start from rest). |

**Returns:** The filtered signal, a list of floats; with zi, a tuple (y, zf) with the final conditions.

### sosfilt_zi

```python
system.signal.sosfilt_zi(sos)
```

Initial conditions for sosfilt that start it in steady state for a step input. Same as scipy.signal.sosfilt_zi.

| Parameter | Description |
| --- | --- |
| `sos` | Second-order sections, rows of [b0, b1, b2, 1, a1, a2], as butter(..., output='sos') returns. |

**Returns:** One [z0, z1] row per section. Multiply by the first sample to start on a signal.

### filtfilt

```python
system.signal.filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None)
```

Filter forwards and backwards for zero lag (zero phase), with b/a coefficients. Same as scipy.signal.filtfilt.

| Parameter | Description |
| --- | --- |
| `b` | Numerator coefficients. |
| `a` | Denominator coefficients (a number or a list). a[0] must not be 0. |
| `x` | The signal, a list of numbers, evenly sampled. None, NaN and infinity are not allowed. |
| `axis` | -1 (default) or 0. Signals are 1-D, so both mean the same. |
| `padtype` | Edge extension before filtering: 'odd' (default), 'even', 'constant' or None. |
| `padlen` | Samples to extend at each end. Default: 3 times the number of filter taps. Must be less than len(x). |
| `method` | 'pad' (default). 'gust' is not supported. |
| `irlen` | Only used by method='gust'; accepted for compatibility. |

**Returns:** The filtered signal, a list of floats, in step with the input.

### sosfiltfilt

```python
system.signal.sosfiltfilt(sos, x, axis=-1, padtype='odd', padlen=None)
```

Filter forwards and backwards for zero lag (zero phase), with second-order sections. The way to clean historian data without delay. Same as scipy.signal.sosfiltfilt.

| Parameter | Description |
| --- | --- |
| `sos` | Second-order sections, rows of [b0, b1, b2, 1, a1, a2], as butter(..., output='sos') returns. |
| `x` | The signal, a list of numbers, evenly sampled. None, NaN and infinity are not allowed. |
| `axis` | -1 (default) or 0. Signals are 1-D, so both mean the same. |
| `padtype` | Edge extension before filtering: 'odd' (default), 'even', 'constant' or None. |
| `padlen` | Samples to extend at each end. Default: 3 times the number of filter taps. Must be less than len(x). |

**Returns:** The filtered signal, a list of floats, in step with the input.

## Smoothing

### savgol_coeffs

```python
system.signal.savgol_coeffs(window_length, polyorder, deriv=0, delta=1.0, pos=None, use='conv')
```

Savitzky-Golay filter coefficients. Same as scipy.signal.savgol_coeffs, but computed exactly, so they stay correct at high orders where scipy's lose accuracy.

| Parameter | Description |
| --- | --- |
| `window_length` | Number of samples in the fitting window (an integer). |
| `polyorder` | Degree of the fitted polynomial; less than window_length. |
| `deriv` | Derivative to compute: 0 (default) smooths, 1 gives the slope, 2 the curvature. |
| `delta` | Sample spacing, used to scale derivatives. Default 1.0. |
| `pos` | Window position the coefficients evaluate at. Default: the centre. |
| `use` | 'conv' (default) for convolution order, 'dot' for dot-product order. |

**Returns:** The coefficients, a list of window_length floats.

### savgol_filter

```python
system.signal.savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
```

Smooth a signal (or take its derivative) by local polynomial fits, keeping peak shapes. Same as scipy.signal.savgol_filter, with exactly computed fits.

| Parameter | Description |
| --- | --- |
| `x` | The signal, a list of numbers, evenly sampled. None, NaN and infinity are not allowed. |
| `window_length` | Number of samples in the fitting window (an integer). |
| `polyorder` | Degree of the fitted polynomial; less than window_length. |
| `deriv` | Derivative to compute: 0 (default) smooths, 1 gives the slope, 2 the curvature. |
| `delta` | Sample spacing, used to scale derivatives. Default 1.0. |
| `axis` | -1 (default) or 0. Signals are 1-D, so both mean the same. |
| `mode` | Edge handling: 'interp' (default, fits the edges), 'mirror', 'constant', 'nearest' or 'wrap'. |
| `cval` | Value beyond the edges for mode='constant'. Default 0.0. |

**Returns:** The smoothed signal (or derivative), a list of floats.

### medfilt

```python
system.signal.medfilt(volume, kernel_size=None)
```

Median filter: replaces each sample with the median of its window. Removes spikes and keeps steps sharp. Same as scipy.signal.medfilt.

| Parameter | Description |
| --- | --- |
| `volume` | The signal, a list of numbers. Integers in give integers out. |
| `kernel_size` | Window length, an odd integer. Default 3. The signal is padded with zeros beyond its ends. |

**Returns:** The filtered signal: integers if the input was all integers, otherwise floats.

## Peak finding

### find_peaks

```python
system.signal.find_peaks(x, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
```

Find peaks (local maxima) in a signal, filtered by height, threshold, distance, prominence, width or plateau size. Same as scipy.signal.find_peaks.

| Parameter | Description |
| --- | --- |
| `x` | The signal, a list of numbers. None, NaN and infinity are not allowed. |
| `height` | Required peak height: a number (minimum) or (min, max); either may be None. |
| `threshold` | Required rise over both neighbouring samples: a number (minimum) or (min, max). |
| `distance` | Minimum samples between peaks (>= 1). Lower peaks within it are removed first. |
| `prominence` | Required prominence (how far a peak stands out): a number (minimum) or (min, max). |
| `width` | Required width in samples, measured at rel_height: a number (minimum) or (min, max). |
| `wlen` | Window in samples for prominence searches. Default None: the whole signal. |
| `rel_height` | Height for widths, as a fraction of prominence below the peak. Default 0.5 (half prominence). |
| `plateau_size` | Required flat-top length in samples: a number (minimum) or (min, max). |

**Returns:** A tuple (peaks, properties): peaks is a list of sample indices; properties is a dict of lists, e.g. 'prominences', 'widths', 'peak_heights', for the conditions given.

### peak_prominences

```python
system.signal.peak_prominences(x, peaks, wlen=None)
```

How far each peak stands out from the signal around it. Same as scipy.signal.peak_prominences.

| Parameter | Description |
| --- | --- |
| `x` | The signal, a list of numbers. None, NaN and infinity are not allowed. |
| `peaks` | Indices of peaks in x, a list of integers (as find_peaks returns). |
| `wlen` | Window in samples for prominence searches. Default None: the whole signal. |

**Returns:** A tuple (prominences, left_bases, right_bases).

### peak_widths

```python
system.signal.peak_widths(x, peaks, rel_height=0.5, prominence_data=None, wlen=None)
```

Width of each peak at a height relative to its prominence. Same as scipy.signal.peak_widths.

| Parameter | Description |
| --- | --- |
| `x` | The signal, a list of numbers. None, NaN and infinity are not allowed. |
| `peaks` | Indices of peaks in x, a list of integers (as find_peaks returns). |
| `rel_height` | Height for widths, as a fraction of prominence below the peak. Default 0.5 (half prominence). |
| `prominence_data` | Optional (prominences, left_bases, right_bases), as peak_prominences returns. |
| `wlen` | Window in samples for prominence searches. Default None: the whole signal. |

**Returns:** A tuple (widths, width_heights, left_ips, right_ips); ips are interpolated sample positions.

## Spectral analysis

### welch

```python
system.signal.welch(x, fs=1.0, window='hann_periodic', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
```

Power spectral density by Welch's method (averaged, windowed segments). Same as scipy.signal.welch.

| Parameter | Description |
| --- | --- |
| `x` | The signal, a list of numbers, evenly sampled. None, NaN and infinity are not allowed. |
| `fs` | Sampling frequency (samples per second). Default 1.0. Frequencies come out in the same unit. |
| `window` | Window name (e.g. 'hann'), (name, parameter) tuple, or a list of window values. |
| `nperseg` | Samples per segment. Default 256 (or the signal length if shorter). |
| `noverlap` | Samples shared by neighbouring segments. Default: half a segment (welch), an eighth (spectrogram). |
| `nfft` | FFT length, at least nperseg (zero-padded). Default nperseg. |
| `detrend` | Detrending of each segment: 'constant' (default), 'linear' or False. |
| `return_onesided` | True (default): frequencies 0 to fs/2 only, as for real signals. |
| `scaling` | 'density' (default, power per Hz) or 'spectrum' (power, e.g. V**2). |
| `axis` | -1 (default) or 0. Signals are 1-D, so both mean the same. |
| `average` | 'mean' (default) or 'median' (robust to bursts) over segments. |

**Returns:** A tuple (f, Pxx): frequencies and power spectral density, lists of floats.

### periodogram

```python
system.signal.periodogram(x, fs=1.0, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)
```

Power spectral density of the whole signal in one segment. Same as scipy.signal.periodogram.

| Parameter | Description |
| --- | --- |
| `x` | The signal, a list of numbers, evenly sampled. None, NaN and infinity are not allowed. |
| `fs` | Sampling frequency (samples per second). Default 1.0. Frequencies come out in the same unit. |
| `window` | Window name (e.g. 'hann'), (name, parameter) tuple, or a list of window values. |
| `nfft` | FFT length, at least nperseg (zero-padded). Default nperseg. |
| `detrend` | Detrending of each segment: 'constant' (default), 'linear' or False. |
| `return_onesided` | True (default): frequencies 0 to fs/2 only, as for real signals. |
| `scaling` | 'density' (default, power per Hz) or 'spectrum' (power, e.g. V**2). |
| `axis` | -1 (default) or 0. Signals are 1-D, so both mean the same. |

**Returns:** A tuple (f, Pxx): frequencies and power spectral density, lists of floats.

### spectrogram

```python
system.signal.spectrogram(x, fs=1.0, window=('tukey_periodic', 0.25), nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')
```

Spectrum over time: one spectrum per segment. Same as scipy.signal.spectrogram.

| Parameter | Description |
| --- | --- |
| `x` | The signal, a list of numbers, evenly sampled. None, NaN and infinity are not allowed. |
| `fs` | Sampling frequency (samples per second). Default 1.0. Frequencies come out in the same unit. |
| `window` | Window name (e.g. 'hann'), (name, parameter) tuple, or a list of window values. |
| `nperseg` | Samples per segment. Default 256 (or the signal length if shorter). |
| `noverlap` | Samples shared by neighbouring segments. Default: half a segment (welch), an eighth (spectrogram). |
| `nfft` | FFT length, at least nperseg (zero-padded). Default nperseg. |
| `detrend` | Detrending of each segment: 'constant' (default), 'linear' or False. |
| `return_onesided` | True (default): frequencies 0 to fs/2 only, as for real signals. |
| `scaling` | 'density' (default, power per Hz) or 'spectrum' (power, e.g. V**2). |
| `axis` | -1 (default) or 0. Signals are 1-D, so both mean the same. |
| `mode` | 'psd' (default), 'complex', 'magnitude', 'angle' or 'phase'. |

**Returns:** A tuple (f, t, Sxx): frequencies, segment centre times, and Sxx with one row per frequency and one column per time (complex numbers for mode='complex').

### hilbert

```python
system.signal.hilbert(x, N=None, axis=-1)
```

Analytic signal by the Hilbert transform; abs() of it is the envelope. Same as scipy.signal.hilbert.

| Parameter | Description |
| --- | --- |
| `x` | The signal, a list of numbers, evenly sampled. None, NaN and infinity are not allowed. |
| `N` | FFT length. Default: len(x); the signal is truncated or zero-padded. |
| `axis` | -1 (default) or 0. Signals are 1-D, so both mean the same. |

**Returns:** The analytic signal, a list of complex numbers.

## Resampling

### resample

```python
system.signal.resample(x, num, t=None, axis=0, window=None, domain='time')
```

Resample to num samples with the FFT (assumes the signal is periodic). Same as scipy.signal.resample.

| Parameter | Description |
| --- | --- |
| `x` | The signal, a list of numbers, evenly sampled. None, NaN and infinity are not allowed. |
| `num` | Number of samples in the result. |
| `t` | Optional sample times (only the first two are used); then the new times are returned too. |
| `axis` | Signals are 1-D, so the axis has no effect. |
| `window` | resample: spectral window (name, (name, parameter) or values). resample_poly: FIR window (default ('kaiser', 5.0)) or the filter's taps as a list. |
| `domain` | 'time' (default): x is a signal. 'freq': x is a spectrum (FFT order). |

**Returns:** The resampled signal, a list of floats (complex numbers for domain='freq'); with t, a tuple (signal, new_times).

### resample_poly

```python
system.signal.resample_poly(x, up, down, axis=0, window=('kaiser', 5.0), padtype='constant', cval=None)
```

Resample by up / down with a polyphase FIR filter (good for non-periodic signals). Same as scipy.signal.resample_poly.

| Parameter | Description |
| --- | --- |
| `x` | The signal, a list of numbers, evenly sampled. None, NaN and infinity are not allowed. |
| `up` | Upsampling factor (a whole number >= 1). |
| `down` | Downsampling factor (a whole number >= 1). |
| `axis` | Signals are 1-D, so the axis has no effect. |
| `window` | resample: spectral window (name, (name, parameter) or values). resample_poly: FIR window (default ('kaiser', 5.0)) or the filter's taps as a list. |
| `padtype` | Edge handling: 'constant' (default), 'edge', 'symmetric', 'reflect', 'wrap', 'smooth', 'line', 'antisymmetric', 'antireflect', or 'mean', 'median', 'minimum', 'maximum' (subtract it, resample, add it back). |
| `cval` | Value beyond the edges for padtype='constant'. Default 0. |

**Returns:** The resampled signal, ceil(len(x) * up / down) floats.

### decimate

```python
system.signal.decimate(x, q, n=None, ftype='iir', axis=-1, zero_phase=True)
```

Lowpass filter, then keep every q-th sample (no aliasing). Same as scipy.signal.decimate.

| Parameter | Description |
| --- | --- |
| `x` | The signal, a list of numbers, evenly sampled. None, NaN and infinity are not allowed. |
| `q` | Downsampling factor (a whole number). |
| `n` | Filter order. Default 8 for 'iir', 20 * q for 'fir'. |
| `ftype` | 'iir' (default, Chebyshev type I) or 'fir' (Hamming-windowed). |
| `axis` | Signals are 1-D, so the axis has no effect. |
| `zero_phase` | True (default): filter forwards and backwards, so there is no lag. |

**Returns:** The decimated signal, ceil(len(x) / q) floats.
