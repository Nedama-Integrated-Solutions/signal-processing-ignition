package com.nedama.signal;

import java.util.Arrays;

/**
 * Spectral analysis, ported from scipy.signal (scipy 1.18.1): welch and periodogram (which scipy computes through
 * its ShortTimeFFT class), spectrogram (through its older _spectral_helper, with modes psd, complex, magnitude,
 * angle and phase) and hilbert. Segmenting, detrending, window scaling, one-sided doubling and averaging follow
 * scipy operation by operation; the transforms are this module's FFT, which agrees with scipy.fft to within a few
 * roundings of the segment's size.
 */
public final class Spectral {

    private Spectral() {
    }

    /** Detrending applied to each segment: none, constant (mean) or linear. */
    public enum Trend { NONE, CONSTANT, LINEAR }

    private static double[] detrend(double[] seg, Trend trend) {
        switch (trend) {
            case CONSTANT:
                return Detrend.constant(seg);
            case LINEAR:
                return Detrend.linear(seg, new long[] {0});
            default:
                return seg;
        }
    }

    /** rfft or fft of a real segment zero-padded to nfft. */
    private static double[][] transform(double[] seg, int nfft, boolean onesided) {
        double[] padded = Arrays.copyOf(seg, nfft);
        return onesided ? Fft.rfft(padded) : Fft.fft(padded, new double[nfft]);
    }

    /** numpy's rfftfreq(n, d) or fftfreq(n, d) with d = 1 / fs. */
    static double[] freqs(int n, double fs, boolean onesided) {
        double d = 1 / fs;
        double val = 1.0 / (n * d);
        if (onesided) {
            double[] f = new double[n / 2 + 1];
            for (int i = 0; i < f.length; i++) {
                f[i] = i * val;
            }
            return f;
        }
        double[] f = new double[n];
        int pos = (n - 1) / 2 + 1;
        for (int i = 0; i < pos; i++) {
            f[i] = i * val;
        }
        for (int i = pos; i < n; i++) {
            f[i] = (i - n) * val;
        }
        return f;
    }

    /** Frequencies and spectral density (or power spectrum). */
    public static final class Psd {
        public final double[] f;
        public final double[] p;

        Psd(double[] f, double[] p) {
            this.f = f;
            this.p = p;
        }
    }

    /** scipy's _median_bias: the ratio of median to mean for a chi-squared with 2 degrees of freedom. */
    static double medianBias(int n) {
        int m = (n - 1) / 2;
        double[] terms = new double[m];
        for (int i = 1; i <= m; i++) {
            double ii2 = 2.0 * i;
            terms[i - 1] = 1. / (ii2 + 1) - 1. / ii2;
        }
        return 1 + Numpy.sum(terms);
    }

    /** numpy's median of a copy of v. */
    static double median(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        int n = s.length;
        if (n % 2 == 1) {
            return s[n / 2];
        }
        return (s[n / 2 - 1] + s[n / 2]) / 2;
    }

    /**
     * welch as scipy 1.18 computes it (csd of x with itself, through ShortTimeFFT): the window is scaled once
     * (Python's sequential sum), each segment is detrended, multiplied by the scaled window and transformed,
     * |X|^2 = re^2 + im^2, one-sided interior bins doubled, then the segments are averaged (numpy's pairwise mean, or
     * the median divided by the median bias). {@code x} is non-empty, {@code win} has nperseg values.
     */
    public static Psd welch(double[] x, double fs, double[] win, int noverlap, int nfft, Trend trend,
                            boolean onesided, String scaling, String average) {
        int n = x.length;
        int nperseg = win.length;
        if (!scaling.equals("density") && !scaling.equals("spectrum")) {
            throw SignalException.valueError("Parameter scaling='" + scaling
                    + "' not in {'spectrum': 'magnitude', 'density': 'psd'}!");
        }
        if (!(fs > 0)) {
            throw SignalException.valueError("Sampling frequency fs=" + fs + " must be positive!");
        }
        int hop = nperseg - noverlap;
        // ShortTimeFFT.scale_to
        double t = 1 / fs;
        double fac;
        if (scaling.equals("density")) {
            double s = 0;
            for (double w : win) {
                s += w * w;
            }
            fac = 1 / Math.sqrt(s / t);
        } else {
            double s = 0;
            for (double w : win) {
                s += w;
            }
            fac = 1 / Math.abs(s);
        }
        double[] w = new double[nperseg];
        for (int i = 0; i < nperseg; i++) {
            w[i] = win[i] * fac;
        }
        int needed = nperseg - nperseg / 2;
        if (n < needed) {
            throw SignalException.valueError("len(x)=" + n + " must be >= ceil(m_num/2) = " + needed + "!");
        }
        int segments = Math.floorDiv(n - noverlap, hop);
        if (segments < 1) {
            throw SignalException.valueError("Invalid Parameter p0=0, p1=" + segments + ": the signal is too "
                    + "short for one segment of " + nperseg + " samples");
        }
        int nf = onesided ? nfft / 2 + 1 : nfft;
        double[][] power = new double[nf][segments];
        for (int p = 0; p < segments; p++) {
            double[] seg = detrend(Arrays.copyOfRange(x, p * hop, p * hop + nperseg), trend);
            double[] xw = new double[nperseg];
            for (int i = 0; i < nperseg; i++) {
                xw[i] = seg[i] * w[i];
            }
            double[][] spec = transform(xw, nfft, onesided);
            for (int k = 0; k < nf; k++) {
                double re = spec[0][k];
                double im = spec[1][k];
                power[k][p] = re * re + im * im;
            }
        }
        if (onesided) {
            int end = nfft % 2 == 0 ? nf - 1 : nf;
            for (int k = 1; k < end; k++) {
                for (int p = 0; p < segments; p++) {
                    power[k][p] *= 2;
                }
            }
        }
        double[] out = new double[nf];
        if (segments > 1) {
            if (average.equals("median")) {
                double bias = medianBias(segments);
                for (int k = 0; k < nf; k++) {
                    out[k] = median(power[k]) / bias;
                }
            } else if (average.equals("mean")) {
                for (int k = 0; k < nf; k++) {
                    out[k] = Numpy.sum(power[k]) / segments;
                }
            } else {
                throw SignalException.valueError("Parameter average='" + average + "' must be 'median' or 'mean'!");
            }
        } else {
            for (int k = 0; k < nf; k++) {
                out[k] = power[k][0];
            }
        }
        return new Psd(freqs(nfft, fs, onesided), out);
    }

    // spectrogram

    /** Frequencies, segment times, and the spectrogram (one row per frequency, one column per segment). */
    public static final class Spectrogram {
        public final double[] f;
        public final double[] t;
        public final double[][] real;
        public final Complex[][] complex;

        Spectrogram(double[] f, double[] t, double[][] real, Complex[][] complex) {
            this.f = f;
            this.t = t;
            this.real = real;
            this.complex = complex;
        }
    }

    /**
     * spectrogram through scipy's _spectral_helper: segments are detrended, multiplied by the window and
     * transformed; scaling is applied afterwards (numpy's pairwise sums for the window), as a square root for the
     * complex-valued modes. {@code win} has nperseg values, nperseg no more than len(x).
     */
    public static Spectrogram spectrogram(double[] x, double fs, double[] win, int noverlap, int nfft, Trend trend,
                                          boolean onesided, String scaling, String mode) {
        int n = x.length;
        int nperseg = win.length;
        if (noverlap >= nperseg) {
            throw SignalException.valueError("noverlap must be less than nperseg.");
        }
        int step = nperseg - noverlap;
        double scale;
        if (scaling.equals("density")) {
            double[] ww = new double[nperseg];
            for (int i = 0; i < nperseg; i++) {
                ww[i] = win[i] * win[i];
            }
            scale = 1.0 / (fs * Numpy.sum(ww));
        } else if (scaling.equals("spectrum")) {
            double s = Numpy.sum(win);
            scale = 1.0 / (s * s);
        } else {
            throw SignalException.valueError("Unknown scaling: '" + scaling + "'");
        }
        boolean psd = mode.equals("psd");
        if (!psd) {
            scale = Math.sqrt(scale);
        }
        if (fs == 0) {
            throw SignalException.valueError("division by zero (fs = 0)");
        }
        double[] f = freqs(nfft, fs, onesided);
        int segments = (n - nperseg) / step + 1;
        int nf = f.length;
        double[][] real = new double[nf][segments];
        Complex[][] cplx = psd ? null : new Complex[nf][segments];
        for (int p = 0; p < segments; p++) {
            double[] seg = detrend(Arrays.copyOfRange(x, p * step, p * step + nperseg), trend);
            double[] xw = new double[nperseg];
            for (int i = 0; i < nperseg; i++) {
                xw[i] = win[i] * seg[i];
            }
            double[][] spec = transform(xw, nfft, onesided);
            for (int k = 0; k < nf; k++) {
                double re = spec[0][k];
                double im = spec[1][k];
                if (psd) {
                    real[k][p] = (re * re + im * im) * scale;
                } else {
                    cplx[k][p] = new Complex(re * scale, im * scale);
                }
            }
        }
        if (psd && onesided) {
            int end = nfft % 2 == 1 ? nf : nf - 1;
            for (int k = 1; k < end; k++) {
                for (int p = 0; p < segments; p++) {
                    real[k][p] *= 2;
                }
            }
        }
        // numpy.arange(nperseg/2, n - nperseg/2 + 1, step) / fs
        double start = nperseg / 2.0;
        double stop = n - nperseg / 2.0 + 1;
        int count = (int) Math.ceil((stop - start) / step);
        double[] t = new double[Math.max(count, 0)];
        for (int i = 0; i < t.length; i++) {
            t[i] = (start + i * (double) step) / fs;
        }
        if (psd || mode.equals("complex")) {
            return new Spectrogram(f, t, psd ? real : null, cplx);
        }
        for (int k = 0; k < nf; k++) {
            for (int p = 0; p < segments; p++) {
                Complex c = cplx[k][p];
                real[k][p] = mode.equals("magnitude") ? Math.hypot(c.re, c.im) : Math.atan2(c.im, c.re);
            }
        }
        if (mode.equals("phase")) {
            unwrapColumns(real);
        }
        return new Spectrogram(f, t, real, null);
    }

    /** numpy's remainder for floats: the result has the sign of the divisor. */
    private static double mod(double a, double b) {
        double m = a % b;
        if (m != 0) {
            if ((b < 0) != (m < 0)) {
                m += b;
            }
        } else {
            m = Math.copySign(0.0, b);
        }
        return m;
    }

    /** numpy.unwrap along the frequency axis (down each column), with numpy's default period 2 pi. */
    private static void unwrapColumns(double[][] p) {
        int rows = p.length;
        if (rows < 2) {
            return;
        }
        double period = 2 * Math.PI;
        double high = period / 2;
        double low = -high;
        int cols = p[0].length;
        for (int c = 0; c < cols; c++) {
            double cum = 0;
            double prev = p[0][c];
            for (int r = 1; r < rows; r++) {
                double cur = p[r][c];
                double dd = cur - prev;
                double ddmod = mod(dd - low, period) + low;
                if (ddmod == low && dd > 0) {
                    ddmod = high;
                }
                double correct = ddmod - dd;
                if (Math.abs(dd) < high) {
                    correct = 0;
                }
                cum += correct;
                prev = cur;
                p[r][c] = cur + cum;
            }
        }
    }

    // hilbert

    /** The analytic signal of x over N points (x truncated or zero-padded to N). */
    public static Complex[] hilbert(double[] x, int nPoints) {
        if (nPoints <= 0) {
            throw SignalException.valueError("N must be positive.");
        }
        double[][] xf = Fft.fft(Arrays.copyOf(x, nPoints), new double[nPoints]);
        double[] re = xf[0];
        double[] im = xf[1];
        int n = nPoints;
        int last = n % 2 == 0 ? n / 2 : (n + 1) / 2;
        int zeroFrom = n % 2 == 0 ? n / 2 + 1 : (n + 1) / 2;
        for (int k = 1; k < last; k++) {
            re[k] *= 2.0;
            im[k] *= 2.0;
        }
        for (int k = zeroFrom; k < n; k++) {
            re[k] = 0.0;
            im[k] = 0.0;
        }
        double[][] z = Fft.ifft(re, im);
        Complex[] out = new Complex[n];
        for (int k = 0; k < n; k++) {
            out[k] = new Complex(z[0][k], z[1][k]);
        }
        return out;
    }
}
