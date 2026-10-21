# Signal Processing for Ignition

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Ignition 8.1](https://img.shields.io/badge/Ignition-8.1.33%2B-orange)
![Ignition 8.3](https://img.shields.io/badge/Ignition-8.3.3%2B-orange)

`scipy.signal` for Ignition scripts, under `system.signal`: filter design, zero-phase filtering, Savitzky-Golay
smoothing, peak finding, spectra and resampling, with scipy's function names, parameters and defaults. 33
functions, in every scope. Every test run checks about 65,000 calls, some 54 million samples of signal, against
SciPy 1.18.1, and 1,000,000-point signals run inside Ignition 8.1 and 8.3 as part of the test.

```python
sos = system.signal.butter(4, 2.0, fs=50.0, output="sos")
clean = system.signal.sosfiltfilt(sos, x)               # filter without lag
peaks, props = system.signal.find_peaks(clean, prominence=2.0)
f, pxx = system.signal.welch(x, fs=50.0, nperseg=1024)  # power spectral density
```

| Module | Ignition | Module ID |
| --- | --- | --- |
| Signal Processing (8.1 build) | 8.1.33 or newer | `com.nedama.ignition.signal` |
| Signal Processing (8.3 build) | 8.3.3 or newer | `com.nedama.ignition.signal` |

Documentation, with every function and the few places where the module deliberately differs from SciPy:
[docs.nedama.com](https://docs.nedama.com).

## Building

Requires JDK 17 and Maven 3.8 or newer.

```
mvn clean package
```

This builds an unsigned `.modl` in `build-81\target` and in `build-83\target`. Release files are signed by
Nedama Integrated Solutions; to install your own build, use a gateway that allows unsigned modules.

## Testing

Requires Python 3. From the `signal-core` folder:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r src\test\python\requirements.txt
mvn -q test-compile
python src\test\python\random_check.py
```

Inside Ignition: `random_check.py --export`, then paste `target\accuracy_test.py` into the Designer Script
Console, then `compare_ignition.py --results <path it prints>`. For gateway scope, see
`src\test\ignition\gateway_check.py`.

## Support

Best-effort support through [GitHub issues](../../issues). For integration work and commercial support, contact
[Nedama Integrated Solutions](https://nedama.com).

## Credits and license

The algorithms are Java ports of parts of [SciPy](https://scipy.org), the Cephes special functions it ships, and
[NumPy](https://numpy.org) behaviour, all BSD 3-Clause. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Copyright 2026 Nedama Integrated Solutions. Licensed under the [Apache License 2.0](LICENSE).
Ignition is a trademark of Inductive Automation; this module is not affiliated with or endorsed by Inductive Automation.
