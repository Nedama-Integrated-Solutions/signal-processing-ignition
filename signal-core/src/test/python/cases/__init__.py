"""
One module per function, or per group of internal building blocks. Each module defines:

    FN                                 the name used by --only and for seeding (internal groups start with _)
    reference(*args, **kwargs)         scipy plus the module's documented rules, for FN
    REFERENCES                         optional {function name: reference}, when a module covers several
    generate(rng, n, max_len)          yields random Cases compared with the reference
    properties(rng, n, max_len)        yields Cases with a check() that needs no reference

Add each new module to MODULES.
"""

from cases import (aberth, bessel_zeros, detrend, fft, filtering, fminbound, freqz, get_window, iir, order,
                   peaks, resampling, smoothing, special, spectral)

MODULES = [detrend, get_window, iir, freqz, order, filtering, smoothing, peaks, spectral, resampling, fft, special,
           fminbound, aberth, bessel_zeros]

BY_NAME = {m.FN: m for m in MODULES}

REFERENCES = {}
for _m in MODULES:
    REFERENCES.update(getattr(_m, "REFERENCES", None) or {_m.FN: _m.reference})
