package com.nedama.signal;

import java.util.Arrays;

/**
 * Resampling, ported from scipy.signal (scipy 1.18.1): resample (FFT, with its window and domain options),
 * resample_poly (a firwin lowpass applied by upfirdn, with every padtype) and decimate (Chebyshev IIR with
 * sosfiltfilt, or FIR through resample_poly). firwin and upfirdn are internal: firwin only as resample_poly and
 * decimate use it (one cutoff, lowpass, scaled), upfirdn with scipy's Cython loop and edge modes.
 */
public final class Resampling {

    private Resampling() {
    }

    // firwin

    /** numpy.sinc: y = pi x, with an exact zero replaced by machine epsilon, then sin(y) / y. */
    private static double sinc(double v) {
        double y = Math.PI * v;
        if (y == 0) {
            y = Math.ulp(1.0);
        }
        return Math.sin(y) / y;
    }

    /** firwin(numtaps, cutoff, window) for one cutoff (relative to Nyquist): lowpass, scaled to unit gain at DC. */
    static double[] firwin(int numtaps, double cutoff, double[] symmetricWindow) {
        if (cutoff <= 0 || cutoff >= 1) {
            throw SignalException.valueError("Invalid cutoff frequency: frequencies must be greater than 0 and less "
                    + "than fs/2.");
        }
        double alpha = 0.5 * (numtaps - 1);
        double[] h = new double[numtaps];
        for (int i = 0; i < numtaps; i++) {
            double m = i - alpha;
            h[i] = 0 + cutoff * sinc(cutoff * m);
            h[i] = h[i] - 0.0 * sinc(0.0 * m);
            h[i] *= symmetricWindow[i];
        }
        double[] hc = new double[numtaps];
        for (int i = 0; i < numtaps; i++) {
            hc[i] = h[i] * Math.cos(Math.PI * (i - alpha) * 0.0);
        }
        double s = Numpy.sum(hc);
        for (int i = 0; i < numtaps; i++) {
            h[i] /= s;
        }
        return h;
    }

    // upfirdn

    /** Edge modes of upfirdn. */
    public enum Mode { CONSTANT, SYMMETRIC, EDGE, SMOOTH, WRAP, REFLECT, ANTISYMMETRIC, ANTIREFLECT, LINE }

    /** scipy's _output_len. */
    static long outputLen(long lenH, long inLen, long up, long down) {
        return Math.floorDiv(((inLen - 1) * up + lenH) - 1, down) + 1;
    }

    private static double extendLeft(double[] x, int idx, int lenX, Mode mode, double cval) {
        switch (mode) {
            case SYMMETRIC:
                if (-idx < lenX) {
                    return x[-idx - 1];
                }
                idx = (-idx - 1) % (2 * lenX);
                return idx < lenX ? x[idx] : x[lenX - 1 - (idx - lenX)];
            case REFLECT:
                if (-idx < lenX - 1) {
                    return x[-idx];
                }
                idx = (-idx - 1) % (2 * (lenX - 1));
                return idx < lenX - 1 ? x[idx + 1] : x[lenX - 2 - (idx - (lenX - 1))];
            case WRAP:
                idx = (-idx - 1) % lenX;
                return x[lenX - idx - 1];
            case SMOOTH:
                return x[0] + idx * (x[1] - x[0]);
            case LINE: {
                double slope = (x[lenX - 1] - x[0]) / (lenX - 1);
                return x[0] + idx * slope;
            }
            case ANTISYMMETRIC:
                if (-idx < lenX) {
                    return -x[-idx - 1];
                }
                idx = (-idx - 1) % (2 * lenX);
                return idx < lenX ? -x[idx] : x[lenX - 1 - (idx - lenX)];
            case ANTIREFLECT:
                if (-idx < lenX) {
                    return x[0] - (x[-idx] - x[0]);
                }
                double le = x[0] + (x[0] - x[lenX - 1]) * ((-idx - 1) / (lenX - 1));
                idx = (-idx - 1) % (2 * (lenX - 1));
                return idx < lenX - 1 ? le - (x[idx + 1] - x[0])
                        : le - (x[lenX - 1] - x[lenX - 2 - (idx - (lenX - 1))]);
            case EDGE:
                return x[0];
            default:
                return cval;
        }
    }

    private static double extendRight(double[] x, int idx, int lenX, Mode mode, double cval) {
        switch (mode) {
            case SYMMETRIC:
                if (idx < 2 * lenX) {
                    return x[lenX - 1 - (idx - lenX)];
                }
                idx = idx % (2 * lenX);
                return idx < lenX ? x[idx] : x[lenX - 1 - (idx - lenX)];
            case REFLECT:
                if (idx < 2 * lenX - 1) {
                    return x[lenX - 2 - (idx - lenX)];
                }
                idx = idx % (2 * (lenX - 1));
                return idx < lenX - 1 ? x[idx] : x[lenX - 1 - (idx - (lenX - 1))];
            case WRAP:
                return x[idx % lenX];
            case SMOOTH:
                return x[lenX - 1] + (idx - lenX + 1) * (x[lenX - 1] - x[lenX - 2]);
            case LINE: {
                double slope = (x[lenX - 1] - x[0]) / (lenX - 1);
                return x[lenX - 1] + (idx - lenX + 1) * slope;
            }
            case EDGE:
                return x[lenX - 1];
            case ANTISYMMETRIC:
                if (idx < 2 * lenX) {
                    return -x[lenX - 1 - (idx - lenX)];
                }
                idx = idx % (2 * lenX);
                return idx < lenX ? x[idx] : -x[lenX - 1 - (idx - lenX)];
            case ANTIREFLECT:
                if (idx < 2 * lenX - 1) {
                    return x[lenX - 1] - (x[lenX - 2 - (idx - lenX)] - x[lenX - 1]);
                }
                double re = x[lenX - 1] + (x[lenX - 1] - x[0]) * (idx / (lenX - 1) - 1);
                idx = idx % (2 * (lenX - 1));
                return idx < lenX - 1 ? re + (x[idx] - x[0]) : re + (x[lenX - 1] - x[lenX - 1 - (idx - (lenX - 1))]);
            default:
                return cval;
        }
    }

    /** upfirdn: upsample by up, filter with h, downsample by down, following scipy's _apply_impl. */
    static double[] upfirdn(double[] h, double[] x, int up, int down, Mode mode, double cval) {
        if (h.length == 0) {
            throw SignalException.valueError("h must be 1-D with non-zero length");
        }
        if (up < 1 || down < 1) {
            throw SignalException.valueError("Both up and down must be >= 1");
        }
        int lenX = x.length;
        if (lenX == 1 && (mode == Mode.REFLECT || mode == Mode.ANTIREFLECT)) {
            // scipy's kernel takes a remainder modulo 2 * (len(x) - 1) here, which kills the Python process
            throw SignalException.valueError("padtype '" + mode.name().toLowerCase()
                    + "' needs at least 2 samples (scipy crashes the interpreter on 1)");
        }
        // _pad_h: pad h to a multiple of up, then per phase, reversed
        int hPad = h.length + Math.floorMod(-h.length, up);
        int perPhase = hPad / up;
        double[] full = Arrays.copyOf(h, hPad);
        double[] flip = new double[hPad];
        int k = 0;
        for (int p = 0; p < up; p++) {
            for (int r = perPhase - 1; r >= 0; r--) {
                flip[k++] = full[r * up + p];
            }
        }
        long lenOutL = outputLen(h.length, lenX, up, down);
        if (lenOutL > 200_000_000L) {
            throw SignalException.valueError("upfirdn: output of " + lenOutL + " samples is too large");
        }
        int lenOut = (int) Math.max(lenOutL, 0);
        double[] out = new double[lenOut];
        int paddedLen = lenX + perPhase - 1;
        boolean zpad = mode == Mode.CONSTANT && cval == 0;
        if (lenOut == 0) {
            return out;
        }
        int xIdx = 0;
        int yIdx = 0;
        int t = 0;
        while (xIdx < lenX) {
            int hIdx = t * perPhase;
            int conv = xIdx - perPhase + 1;
            if (conv < 0) {
                if (zpad) {
                    hIdx -= conv;
                } else {
                    for (int c = conv; c < 0; c++) {
                        out[yIdx] += extendLeft(x, c, lenX, mode, cval) * flip[hIdx];
                        hIdx++;
                    }
                }
                conv = 0;
            }
            for (int c = conv; c < xIdx + 1; c++) {
                out[yIdx] = out[yIdx] + x[c] * flip[hIdx];
                hIdx++;
            }
            yIdx++;
            if (yIdx >= lenOut) {
                return out;
            }
            t += down;
            xIdx += t / up;
            t = t % up;
        }
        while (xIdx < paddedLen) {
            int hIdx = t * perPhase;
            int conv = xIdx - perPhase + 1;
            for (int c = conv; c < xIdx + 1; c++) {
                double xval;
                if (c >= lenX) {
                    xval = extendRight(x, c, lenX, mode, cval);
                } else if (c < 0) {
                    xval = extendLeft(x, c, lenX, mode, cval);
                } else {
                    xval = x[c];
                }
                out[yIdx] += xval * flip[hIdx];
                hIdx++;
            }
            yIdx++;
            if (yIdx >= lenOut) {
                return out;
            }
            t += down;
            xIdx += t / up;
            t = t % up;
        }
        return out;
    }

    // resample_poly

    /** How resample_poly pads: an upfirdn mode, or a background value subtracted first and added back. */
    public enum Background { NONE, MEAN, MEDIAN, MINIMUM, MAXIMUM }

    /**
     * resample_poly with up and down already reduced and validated. {@code h} is a custom filter (window given as
     * values) or null with {@code designWindow} the symmetric window for firwin's 2 * 10 * max(up, down) + 1 taps.
     */
    public static double[] resamplePoly(double[] x, int up, int down, double[] h, WindowMaker designWindow,
                                        Mode mode, double cval, Background background) {
        int nIn = x.length;
        long nOutL = (long) nIn * up;
        nOutL = nOutL / down + (nOutL % down != 0 ? 1 : 0);
        int halfLen;
        if (h != null) {
            h = h.clone();
            halfLen = Math.floorDiv(h.length - 1, 2);
        } else {
            int maxRate = Math.max(up, down);
            double fc = 1. / maxRate;
            halfLen = 10 * maxRate;
            h = firwin(2 * halfLen + 1, fc, designWindow.make(2 * halfLen + 1));
        }
        for (int i = 0; i < h.length; i++) {
            h[i] *= up;
        }
        int nPrePad = down - Math.floorMod(halfLen, down);
        int nPostPad = 0;
        long nPreRemove = Math.floorDiv(halfLen + nPrePad, down);
        while (outputLen(h.length + nPrePad + nPostPad, nIn, up, down) < nOutL + nPreRemove) {
            nPostPad++;
        }
        double[] hp = new double[nPrePad + h.length + nPostPad];
        System.arraycopy(h, 0, hp, nPrePad, h.length);
        double bg = 0;
        double[] xs = x;
        if (background != Background.NONE) {
            bg = backgroundValue(x, background);
            xs = new double[nIn];
            for (int i = 0; i < nIn; i++) {
                xs[i] = x[i] - bg;
            }
            mode = Mode.CONSTANT;
            cval = 0;
        }
        double[] y = upfirdn(hp, xs, up, down, mode, cval);
        int from = (int) nPreRemove;
        int to = (int) (nPreRemove + nOutL);
        double[] keep = Arrays.copyOfRange(y, Math.min(from, y.length), Math.min(to, y.length));
        if (background != Background.NONE) {
            for (int i = 0; i < keep.length; i++) {
                keep[i] += bg;
            }
        }
        return keep;
    }

    /** Builds a symmetric window of n points (get_window(window, n, fftbins=False)). */
    public interface WindowMaker {
        double[] make(int n);
    }

    private static double backgroundValue(double[] x, Background b) {
        if (x.length == 0) {
            throw SignalException.valueError("zero-size array to reduction operation which has no identity");
        }
        switch (b) {
            case MEAN:
                return Numpy.sum(x) / x.length;
            case MEDIAN:
                return Spectral.median(x);
            case MINIMUM: {
                double m = x[0];
                for (double v : x) {
                    m = Math.min(m, v);
                }
                return m;
            }
            default: {
                double m = x[0];
                for (double v : x) {
                    m = Math.max(m, v);
                }
                return m;
            }
        }
    }

    // decimate

    /** decimate with ftype 'iir': cheby1(n, 0.05, 0.8 / q) as sections, filtered both ways (or forwards). */
    public static double[] decimateIir(double[] x, int q, int n, boolean zeroPhase) {
        if (q == 0) {
            throw SignalException.valueError("float division by zero (q = 0)");
        }
        double[][] sos = IirDesign.sos(IirDesign.design("cheby1", n, new double[] {0.8 / q}, 0.05, null, "lowpass",
                false, null, null), false);
        double[] y = zeroPhase ? Filtering.sosfiltfilt(sos, x, Filtering.PadType.ODD, null)
                : Filtering.sosfilt(sos, x, null).y;
        int count = (y.length + q - 1) / q;
        double[] out = new double[count];
        for (int i = 0; i < count; i++) {
            out[i] = y[i * q];
        }
        return out;
    }

    /** decimate with ftype 'fir': firwin(n + 1, 1 / q, 'hamming'), then resample_poly (or upfirdn forwards). */
    public static double[] decimateFir(double[] x, int q, int n, boolean zeroPhase, double[] hamming) {
        if (q == 0) {
            throw SignalException.valueError("float division by zero (q = 0)");
        }
        double[] b = firwin(n + 1, 1. / q, hamming);
        for (int i = 0; i < b.length; i++) {
            b[i] = b[i] / 1.0;
        }
        if (zeroPhase) {
            if (q < 1) {
                throw SignalException.valueError("up and down must be >= 1");
            }
            return q == 1 ? x.clone() : resamplePoly(x, 1, q, b, null, Mode.CONSTANT, 0, Background.NONE);
        }
        int nOut = x.length / q + (x.length % q != 0 ? 1 : 0);
        double[] y = upfirdn(b, x, 1, q, Mode.CONSTANT, 0);
        return Arrays.copyOf(y, Math.min(nOut, y.length));
    }

    // resample (FFT)

    /** resample's result: real (rfft path) or complex (two-sided path), and the new times if t was given. */
    public static final class Resampled {
        public final double[] real;
        public final Complex[] complex;

        Resampled(double[] real, Complex[] complex) {
            this.real = real;
            this.complex = complex;
        }
    }

    /**
     * resample. {@code w} is the spectral window over n_x bins in scipy's order (already fftshifted for a named
     * window), or null; {@code freqDomain} treats x as a spectrum.
     */
    public static Resampled resample(double[] x, int num, double[] w, boolean freqDomain) {
        int nx = x.length;
        if (num == 0) {
            throw SignalException.valueError("division by zero (num = 0)");
        }
        double sFac = (double) nx / num;
        int m = Math.min(num, nx);
        int m2 = m / 2 + 1;
        if (num < 0 || nx == 0) {
            throw SignalException.valueError("Invalid number of FFT data points (" + (nx == 0 ? 0 : num)
                    + ") specified.");
        }
        if (!freqDomain) {
            double[][] spec = Fft.rfft(x);
            double[] re = spec[0];
            double[] im = spec[1];
            int nX = re.length;
            if (w != null) {
                double[] w1 = w.clone();
                for (int l = 1; l < nX; l++) {
                    w1[l] = w[l] + w[nx - l];
                }
                for (int l = 1; l < nX; l++) {
                    w1[l] /= 2;
                }
                for (int k = 0; k < nX; k++) {
                    re[k] *= w1[k];
                    im[k] *= w1[k];
                }
            }
            int keep = Math.min(m2, nX);
            re = Arrays.copyOf(re, keep);
            im = Arrays.copyOf(im, keep);
            if (m % 2 == 0 && num != nx) {
                double f = num < nx ? 2 : 0.5;
                re[m / 2] *= f;
                im[m / 2] *= f;
            }
            for (int k = 0; k < keep; k++) {
                re[k] = re[k] / sFac;
                im[k] = im[k] / sFac;
            }
            return new Resampled(Fft.irfft(re, im, num), null);
        }
        // two-sided path: x is the spectrum (real values)
        double[] xr = x.clone();
        if (w != null) {
            for (int k = 0; k < nx; k++) {
                xr[k] = xr[k] * w[k];
            }
        }
        double[] yr = new double[num];
        System.arraycopy(xr, 0, yr, 0, Math.min(m2, Math.min(num, nx)));
        if (m2 < m) {
            int len = m - m2;
            System.arraycopy(xr, nx - len, yr, num - len, len);
        }
        if (m % 2 == 0) {
            int half = m / 2;
            if (num < nx) {
                yr[num - half] += xr[nx - half];
            } else if (nx < num) {
                yr[half] /= 2;
                yr[num - half] = yr[half];
            }
        }
        double[] yi = new double[num];
        for (int k = 0; k < num; k++) {
            yr[k] = yr[k] / sFac;
        }
        double[][] z = Fft.ifft(yr, yi);
        Complex[] out = new Complex[num];
        for (int k = 0; k < num; k++) {
            out[k] = new Complex(z[0][k], z[1][k]);
        }
        return new Resampled(null, out);
    }
}
