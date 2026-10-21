package com.nedama.signal;

import java.util.Arrays;

/**
 * The window functions behind scipy.signal.get_window, ported from scipy.signal.windows (scipy 1.18.1).
 * {@code sym = true} gives the symmetric window, for filter design; {@code sym = false} the periodic one, for
 * spectral analysis, made as scipy does by computing M + 1 points and dropping the last.
 *
 * <p>Every method expects M of at least 1; the caller checks it.
 */
public final class Windows {

    private Windows() {
    }

    private static double[] ones(int m) {
        double[] w = new double[m];
        Arrays.fill(w, 1.0);
        return w;
    }

    private static double[] truncate(double[] w, boolean sym) {
        return sym ? w : Arrays.copyOf(w, w.length - 1);
    }

    /** numpy.linspace(start, stop, num) with its arithmetic: i * step + start, with the last point set to stop. */
    static double[] linspace(double start, double stop, int num) {
        double[] y = new double[num];
        if (num == 0) {
            return y;
        }
        int div = num - 1;
        double delta = stop - start;
        if (div > 0) {
            double step = delta / div;
            if (step == 0) {
                for (int i = 0; i < num; i++) {
                    y[i] = (i / (double) div) * delta + start;
                }
            } else {
                for (int i = 0; i < num; i++) {
                    y[i] = i * step + start;
                }
            }
            y[num - 1] = stop;
        } else {
            y[0] = 0.0 * delta + start;
        }
        return y;
    }

    /** Sum of cosine terms a[k] cos(k x) over x from -pi to pi. */
    public static double[] generalCosine(int m, double[] a, boolean sym) {
        if (m <= 1) {
            return ones(m);
        }
        int n = sym ? m : m + 1;
        double[] fac = linspace(-Math.PI, Math.PI, n);
        double[] w = new double[n];
        for (int k = 0; k < a.length; k++) {
            for (int i = 0; i < n; i++) {
                w[i] += a[k] * Math.cos(k * fac[i]);
            }
        }
        return truncate(w, sym);
    }

    public static double[] generalHamming(int m, double alpha, boolean sym) {
        return generalCosine(m, new double[] {alpha, 1. - alpha}, sym);
    }

    public static double[] hann(int m, boolean sym) {
        return generalHamming(m, 0.5, sym);
    }

    public static double[] hamming(int m, boolean sym) {
        return generalHamming(m, 0.54, sym);
    }

    public static double[] blackman(int m, boolean sym) {
        return generalCosine(m, new double[] {0.42, 0.50, 0.08}, sym);
    }

    public static double[] nuttall(int m, boolean sym) {
        return generalCosine(m, new double[] {0.3635819, 0.4891775, 0.1365995, 0.0106411}, sym);
    }

    public static double[] blackmanHarris(int m, boolean sym) {
        return generalCosine(m, new double[] {0.35875, 0.48829, 0.14128, 0.01168}, sym);
    }

    public static double[] flattop(int m, boolean sym) {
        return generalCosine(m, new double[] {0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368}, sym);
    }

    public static double[] boxcar(int m, boolean sym) {
        if (m <= 1) {
            return ones(m);
        }
        return truncate(ones(sym ? m : m + 1), sym);
    }

    public static double[] triang(int m, boolean sym) {
        if (m <= 1) {
            return ones(m);
        }
        int n = sym ? m : m + 1;
        int half = (n + 1) / 2;
        double[] w = new double[n];
        if (n % 2 == 0) {
            for (int i = 0; i < half; i++) {
                double v = (2 * (i + 1.0) - 1.0) / n;
                w[i] = v;
                w[n - 1 - i] = v;
            }
        } else {
            for (int i = 0; i < half; i++) {
                double v = 2 * (i + 1.0) / (n + 1.0);
                w[i] = v;
                w[n - 1 - i] = v;
            }
        }
        return truncate(w, sym);
    }

    public static double[] parzen(int m, boolean sym) {
        if (m <= 1) {
            return ones(m);
        }
        int n = sym ? m : m + 1;
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            double x = -(n - 1) / 2.0 + i;
            double ax = Math.abs(x);
            // scipy squares with ** 2.0 (an exact square in numpy) and cubes with ** 3.0 (a pow call).
            if (ax <= (n - 1) / 4.0) {
                double r = ax / (n / 2.0);
                w[i] = 1 - 6 * (r * r) + 6 * Math.pow(r, 3.0);
            } else {
                w[i] = 2 * Math.pow(1 - ax / (n / 2.0), 3.0);
            }
        }
        return truncate(w, sym);
    }

    public static double[] bohman(int m, boolean sym) {
        if (m <= 1) {
            return ones(m);
        }
        int n = sym ? m : m + 1;
        double[] lin = linspace(-1, 1, n);
        double[] w = new double[n];
        for (int i = 1; i < n - 1; i++) {
            double fac = Math.abs(lin[i]);
            w[i] = (1 - fac) * Math.cos(Math.PI * fac) + 1.0 / Math.PI * Math.sin(Math.PI * fac);
        }
        return truncate(w, sym);
    }

    public static double[] bartlett(int m, boolean sym) {
        if (m <= 1) {
            return ones(m);
        }
        int n = sym ? m : m + 1;
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = i <= (n - 1) / 2.0 ? 2.0 * i / (n - 1) : 2.0 - 2.0 * i / (n - 1);
        }
        return truncate(w, sym);
    }

    public static double[] barthann(int m, boolean sym) {
        if (m <= 1) {
            return ones(m);
        }
        int n = sym ? m : m + 1;
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            double fac = Math.abs(i / (n - 1.0) - 0.5);
            w[i] = 0.62 - 0.48 * fac + 0.38 * Math.cos(2 * Math.PI * fac);
        }
        return truncate(w, sym);
    }

    public static double[] cosine(int m, boolean sym) {
        if (m <= 1) {
            return ones(m);
        }
        int n = sym ? m : m + 1;
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = Math.sin(Math.PI / n * (i + .5));
        }
        return truncate(w, sym);
    }

    public static double[] tukey(int m, double alpha, boolean sym) {
        if (m <= 1) {
            return ones(m);
        }
        if (alpha <= 0) {
            return ones(m);
        } else if (alpha >= 1.0) {
            return hann(m, sym);
        }
        int n = sym ? m : m + 1;
        int width = (int) Math.floor(alpha * (n - 1) / 2.0);
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            if (i < width + 1) {
                w[i] = 0.5 * (1 + Math.cos(Math.PI * (-1 + 2.0 * i / alpha / (n - 1))));
            } else if (i < n - width - 1) {
                w[i] = 1.0;
            } else {
                w[i] = 0.5 * (1 + Math.cos(Math.PI * (-2.0 / alpha + 1 + 2.0 * i / alpha / (n - 1))));
            }
        }
        return truncate(w, sym);
    }

    public static double[] kaiser(int m, double beta, boolean sym) {
        if (m <= 1) {
            return ones(m);
        }
        int n = sym ? m : m + 1;
        double alpha = (n - 1) / 2.0;
        double denom = Special.i0(beta);
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            double r = (i - alpha) / alpha;
            w[i] = Special.i0(beta * Math.sqrt(1 - r * r)) / denom;
        }
        return truncate(w, sym);
    }

    /** Kaiser-Bessel derived window: symmetric only, even M only. */
    public static double[] kaiserBesselDerived(int m, double beta, boolean sym) {
        if (!sym) {
            throw SignalException.valueError("Kaiser-Bessel Derived windows are only defined for symmetric shapes");
        }
        if (m < 1) {
            return new double[0];
        }
        if (m % 2 != 0) {
            throw SignalException.valueError(
                    "Kaiser-Bessel Derived windows are only defined for even number of points");
        }
        double[] k = kaiser(m / 2 + 1, beta, true);
        double[] csum = new double[k.length];
        double s = 0.0;
        for (int i = 0; i < k.length; i++) {
            s += k[i];
            csum[i] = s;
        }
        double total = csum[k.length - 1];
        int half = m / 2;
        double[] w = new double[m];
        for (int i = 0; i < half; i++) {
            double v = Math.sqrt(csum[i] / total);
            w[i] = v;
            w[m - 1 - i] = v;
        }
        return w;
    }

    public static double[] gaussian(int m, double std, boolean sym) {
        if (m <= 1) {
            return ones(m);
        }
        int n = sym ? m : m + 1;
        double sig2 = 2 * std * std;
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            double x = i - (n - 1.0) / 2.0;
            w[i] = Math.exp(-(x * x) / sig2);
        }
        return truncate(w, sym);
    }

    public static double[] generalGaussian(int m, double p, double sig, boolean sym) {
        if (m <= 1) {
            return ones(m);
        }
        int n = sym ? m : m + 1;
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            double x = i - (n - 1.0) / 2.0;
            w[i] = Math.exp(-0.5 * Math.pow(Math.abs(x / sig), 2 * p));
        }
        return truncate(w, sym);
    }

    /** Exponential (Poisson) window. {@code center} may be null for the middle; it must be null when symmetric. */
    public static double[] exponential(int m, Double center, double tau, boolean sym) {
        if (sym && center != null) {
            throw SignalException.valueError("If sym==True, center must be None.");
        }
        if (m <= 1) {
            return ones(m);
        }
        int n = sym ? m : m + 1;
        double c = center == null ? (n - 1) / 2.0 : center;
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = Math.exp(-Math.abs(i - c) / tau);
        }
        return truncate(w, sym);
    }

    public static double[] taylor(int m, int nbar, double sll, boolean norm, boolean sym) {
        if (m <= 1) {
            return ones(m);
        }
        int n = sym ? m : m + 1;
        double b = Math.pow(10, sll / 20);
        double a = Special.acosh(b) / Math.PI;
        double a2 = a * a;
        double s2 = (double) nbar * nbar / (a2 + (nbar - 0.5) * (nbar - 0.5));
        int count = nbar - 1;
        double[] ma = new double[count];
        double[] m2 = new double[count];
        for (int i = 0; i < count; i++) {
            ma[i] = i + 1;
            m2[i] = ma[i] * ma[i];
        }
        double[] fm = new double[count];
        for (int mi = 0; mi < count; mi++) {
            double sign = mi % 2 == 0 ? 1 : -1;
            double numer = 1.0;
            for (int j = 0; j < count; j++) {
                double d = ma[j] - 0.5;
                numer *= 1 - m2[mi] / s2 / (a2 + d * d);
            }
            numer *= sign;
            double before = 1.0;
            for (int j = 0; j < mi; j++) {
                before *= 1 - m2[mi] / m2[j];
            }
            double after = 1.0;
            for (int j = mi + 1; j < count; j++) {
                after *= 1 - m2[mi] / m2[j];
            }
            fm[mi] = numer / (2 * before * after);
        }
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = taylorTerm(fm, ma, i, n);
        }
        if (norm) {
            double scale = 1.0 / taylorTerm(fm, ma, (n - 1) / 2.0, n);
            for (int i = 0; i < n; i++) {
                w[i] *= scale;
            }
        }
        return truncate(w, sym);
    }

    private static double taylorTerm(double[] fm, double[] ma, double x, int n) {
        double s = 0.0;
        for (int j = 0; j < fm.length; j++) {
            s += fm[j] * Math.cos(2 * Math.PI * ma[j] * (x - n / 2. + 0.5) / n);
        }
        return 1 + 2 * s;
    }

    /** Dolph-Chebyshev window with sidelobes {@code at} dB below the main lobe. */
    public static double[] chebwin(int m, double at, boolean sym) {
        if (m <= 1) {
            return ones(m);
        }
        int n = sym ? m : m + 1;
        double order = n - 1.0;
        double val = Math.pow(10, Math.abs(at) / 20.);
        double beta = Math.cosh(1.0 / order * Special.acosh(val));
        double[] p = new double[n];
        for (int k = 0; k < n; k++) {
            double x = beta * Math.cos(Math.PI * k / n);
            if (x > 1) {
                p[k] = Math.cosh(order * Special.acosh(x));
            } else if (x < -1) {
                p[k] = (2 * (n % 2) - 1) * Math.cosh(order * Special.acosh(-x));
            } else {
                p[k] = Math.cos(order * Math.acos(x));
            }
        }
        double[] w = new double[n];
        if (n % 2 != 0) {
            double[][] f = Fft.fft(p, new double[n]);
            int h = (n + 1) / 2;
            for (int i = 0; i < h; i++) {
                w[h - 1 + i] = f[0][i];
                w[h - 1 - i] = f[0][i];
            }
        } else {
            double[] re = new double[n];
            double[] im = new double[n];
            for (int k = 0; k < n; k++) {
                // numpy forms 1j * pi / M as pi * (1 / M), then multiplies by k.
                double theta = Math.PI * (1.0 / n) * k;
                re[k] = p[k] * Math.cos(theta);
                im[k] = p[k] * Math.sin(theta);
            }
            double[][] f = Fft.fft(re, im);
            int h = n / 2 + 1;
            for (int i = 1; i < h; i++) {
                w[h - 1 - i] = f[0][i];
                w[h - 2 + i] = f[0][i];
            }
        }
        double max = Double.NEGATIVE_INFINITY;
        for (double v : w) {
            max = Math.max(max, v);
        }
        for (int i = 0; i < n; i++) {
            w[i] /= max;
        }
        return truncate(w, sym);
    }

    public static double[] lanczos(int m, boolean sym) {
        if (m <= 1) {
            return ones(m);
        }
        int n = sym ? m : m + 1;
        double[] w = new double[n];
        if (n % 2 == 0) {
            int h = n / 2;
            for (int i = 0; i < h; i++) {
                double v = Special.sinc(2. * (h + i) / (n - 1) - 1.0);
                w[h + i] = v;
                w[h - 1 - i] = v;
            }
        } else {
            int h = (n + 1) / 2;
            w[h - 1] = 1.0;
            for (int i = 0; i < n - h; i++) {
                double v = Special.sinc(2. * (h + i) / (n - 1) - 1.0);
                w[h + i] = v;
                w[h - 2 - i] = v;
            }
        }
        return truncate(w, sym);
    }
}
