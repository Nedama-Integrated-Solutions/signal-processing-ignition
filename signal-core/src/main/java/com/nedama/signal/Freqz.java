package com.nedama.signal;

/**
 * Frequency response of a digital filter, ported from scipy.signal.freqz and freqz_sos (scipy 1.18.1). Like
 * scipy, freqz uses an FFT of b when a is a single number and enough points are requested, and otherwise
 * evaluates b and a as polynomials in exp(-jw).
 */
public final class Freqz {

    private Freqz() {
    }

    /** Most frequencies accepted, to stop a mistyped worN from exhausting gateway memory. */
    public static final long MAX_POINTS = 50_000_000L;

    /** w and h. */
    public static final class Response {
        public final double[] w;
        public final Complex[] h;

        Response(double[] w, Complex[] h) {
            this.w = w;
            this.h = h;
        }
    }

    /** numpy.linspace with numpy's arithmetic, with or without the end point. */
    static double[] linspace(double start, double stop, int num, boolean endpoint) {
        double[] y = new double[num];
        int div = endpoint ? num - 1 : num;
        double delta = stop - start;
        if (div > 0) {
            double step = delta / div;
            for (int i = 0; i < num; i++) {
                y[i] = step == 0 ? (i / (double) div) * delta + start : i * step + start;
            }
        } else {
            for (int i = 0; i < num; i++) {
                y[i] = i * delta + start;
            }
        }
        if (endpoint && num > 1) {
            y[num - 1] = stop;
        }
        return y;
    }

    /** freqz at N evenly spaced frequencies. */
    public static Response freqz(double[] b, double[] a, long n, boolean whole, double fs, boolean includeNyquist) {
        if (n < 0) {
            throw SignalException.valueError("worN must be nonnegative, got " + n);
        }
        if (n > MAX_POINTS) {
            throw SignalException.valueError("worN is too large (at most " + MAX_POINTS + ")");
        }
        int num = (int) n;
        double lastpoint = whole ? 2 * Math.PI : Math.PI;
        double[] w = linspace(0, lastpoint, num, includeNyquist && !whole);
        long nfft = whole ? num : includeNyquist ? 2L * (num - 1) : 2L * num;
        Complex[] h = null;
        if (a.length == 1 && nfft >= b.length && nfft > 0) {
            double[] padded = java.util.Arrays.copyOf(b, (int) nfft);
            double[][] spec = Fft.rfft(padded);
            int keep = Math.min(num, spec[0].length);
            Complex[] half = new Complex[keep];
            Complex div = new Complex(a[0], 0.0);
            for (int i = 0; i < keep; i++) {
                half[i] = new Complex(spec[0][i], spec[1][i]).div(div);
            }
            if (whole) {
                int stop = nfft % 2 == 1 ? keep : keep - 1;
                int extra = Math.max(0, stop - 1);
                h = new Complex[keep + extra];
                System.arraycopy(half, 0, h, 0, keep);
                for (int i = 0; i < extra; i++) {
                    h[keep + i] = half[stop - 1 - i].conj();
                }
            } else {
                h = half;
            }
        }
        return finish(w, h, b, a, fs);
    }

    /** freqz at the given frequencies, in the same units as fs. */
    public static Response freqz(double[] b, double[] a, double[] freqs, double fs) {
        double[] w = new double[freqs.length];
        for (int i = 0; i < w.length; i++) {
            w[i] = 2 * Math.PI * freqs[i] / fs;
        }
        return finish(w, null, b, a, fs);
    }

    private static Response finish(double[] w, Complex[] h, double[] b, double[] a, double fs) {
        if (h == null) {
            if (b.length == 0 || a.length == 0) {
                throw SignalException.valueError("b and a must not be empty");
            }
            h = new Complex[w.length];
            for (int i = 0; i < w.length; i++) {
                Complex zm1 = new Complex(Math.cos(w[i]), -Math.sin(w[i]));
                h[i] = polyval(zm1, b).div(polyval(zm1, a));
            }
        }
        double scale = fs / (2 * Math.PI);
        double[] out = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            out[i] = w[i] * scale;
        }
        return new Response(out, h);
    }

    /** numpy.polynomial.polynomial.polyval: c[0] + c[1] x + ..., by Horner's rule from the last coefficient. */
    static Complex polyval(Complex x, double[] c) {
        Complex c0 = new Complex(c[c.length - 1], 0.0);
        for (int i = 2; i <= c.length; i++) {
            c0 = c0.times(x).plus(c[c.length - i]);
        }
        return c0;
    }

    /** Checks second-order sections as scipy does: rows of 6 with a leading denominator coefficient of exactly 1. */
    public static void validateSos(double[][] sos) {
        for (double[] row : sos) {
            if (row.length != 6) {
                throw SignalException.valueError("sos array must be shape (n_sections, 6)");
            }
        }
        if (sos.length == 0) {
            throw SignalException.valueError("sos array must be shape (n_sections, 6)");
        }
        for (double[] row : sos) {
            if (row[3] != 1) {
                throw SignalException.valueError("sos[:, 3] should be all ones");
            }
        }
    }

    /** freqz_sos: the product of each section's response. {@code worN} is a Long point count or a double[]. */
    public static Response freqzSos(double[][] sos, Object worN, boolean whole, double fs) {
        validateSos(sos);
        Complex[] h = null;
        double[] w = null;
        for (double[] row : sos) {
            double[] b = {row[0], row[1], row[2]};
            double[] a = {row[3], row[4], row[5]};
            Response r = worN instanceof double[] ? freqz(b, a, (double[]) worN, fs)
                    : freqz(b, a, (Long) worN, whole, fs, false);
            w = r.w;
            if (h == null) {
                h = new Complex[r.h.length];
                for (int i = 0; i < h.length; i++) {
                    h[i] = new Complex(1.0, 0.0).times(r.h[i]);
                }
            } else {
                for (int i = 0; i < h.length; i++) {
                    h[i] = h[i].times(r.h[i]);
                }
            }
        }
        return new Response(w, h);
    }
}
