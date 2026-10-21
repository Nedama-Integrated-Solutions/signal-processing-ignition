package com.nedama.signal;

/**
 * Discrete Fourier transforms of any length, with numpy.fft's conventions: the forward transform is unscaled
 * and the inverse is scaled by 1/n. Like numpy's pocketfft, a length whose prime factors are all small is split
 * into those factors (mixed-radix Cooley-Tukey, with every twiddle factor computed directly), and a length with
 * a large prime factor uses Bluestein's method, a power-of-two convolution. Nothing is cached, so the class is
 * safe to use from any number of gateway threads.
 *
 * <p>Complex arrays are passed as separate real and imaginary arrays.
 */
public final class Fft {

    private Fft() {
    }

    /** Forward transform; returns {re, im}. */
    public static double[][] fft(double[] re, double[] im) {
        checkLength(re.length, im.length);
        double[] r = re.clone();
        double[] i = im.clone();
        transform(r, i, false);
        return new double[][] {r, i};
    }

    /** Inverse transform scaled by 1/n; returns {re, im}. */
    public static double[][] ifft(double[] re, double[] im) {
        checkLength(re.length, im.length);
        double[] r = re.clone();
        double[] i = im.clone();
        transform(r, i, true);
        int n = r.length;
        for (int k = 0; k < n; k++) {
            r[k] /= n;
            i[k] /= n;
        }
        return new double[][] {r, i};
    }

    /** Transform of a real signal: the first n / 2 + 1 terms; returns {re, im}. */
    public static double[][] rfft(double[] x) {
        checkLength(x.length, x.length);
        double[] r = x.clone();
        double[] i = new double[x.length];
        transform(r, i, false);
        int m = x.length / 2 + 1;
        return new double[][] {java.util.Arrays.copyOf(r, m), java.util.Arrays.copyOf(i, m)};
    }

    /**
     * Inverse of {@link #rfft}: a real signal of length n from its first n / 2 + 1 terms. Shorter input is
     * padded with zeros and longer input truncated, as numpy.fft.irfft does; the imaginary parts of the zero
     * and Nyquist terms are ignored.
     */
    public static double[] irfft(double[] re, double[] im, int n) {
        if (n < 1) {
            throw SignalException.valueError("Invalid number of FFT data points (" + n + ") specified.");
        }
        int m = n / 2 + 1;
        double[] r = new double[n];
        double[] i = new double[n];
        for (int k = 0; k < m && k < re.length; k++) {
            r[k] = re[k];
            i[k] = im[k];
        }
        i[0] = 0.0;
        if (n % 2 == 0) {
            i[n / 2] = 0.0;
        }
        for (int k = 1; k < (n + 1) / 2; k++) {
            r[n - k] = r[k];
            i[n - k] = -i[k];
        }
        transform(r, i, true);
        for (int k = 0; k < n; k++) {
            r[k] /= n;
        }
        return r;
    }

    private static void checkLength(int nRe, int nIm) {
        if (nRe != nIm) {
            throw SignalException.valueError("Real and imaginary parts must have the same length.");
        }
        if (nRe < 1) {
            throw SignalException.valueError("Invalid number of FFT data points (" + nRe + ") specified.");
        }
    }

    /**
     * cos and sin of 2 pi num / den, into {@code out}. The angle is reduced to the first octant using exact integer
     * arithmetic before calling cos and sin, as pocketfft does, so every root of unity is accurate to about one
     * rounding: computing 2 pi k / n directly loses up to 6 roundings near 2 pi.
     */
    static void unitRoot(long num, long den, double[] out) {
        num %= den;
        if (num < 0) {
            num += den;
        }
        boolean negSin = false;
        boolean negCos = false;
        boolean swap = false;
        // (pi, 2 pi): reflect to 2 pi - angle, which negates the sine.
        if (2 * num > den) {
            num = den - num;
            negSin = true;
        }
        // (pi/2, pi]: pi - angle, as the fraction (den - 2 num) / (2 den); negates the cosine.
        if (4 * num > den) {
            num = den - 2 * num;
            den = 2 * den;
            negCos = true;
        }
        // (pi/4, pi/2]: pi/2 - angle, as (den - 4 num) / (4 den); swaps cosine and sine.
        if (8 * num > den) {
            num = den - 4 * num;
            den = 4 * den;
            swap = true;
        }
        double angle = 2.0 * Math.PI * num / den;
        double c = Math.cos(angle);
        double sn = Math.sin(angle);
        if (swap) {
            double t = c;
            c = sn;
            sn = t;
        }
        out[0] = negCos ? -c : c;
        out[1] = negSin ? -sn : sn;
    }

    /** Largest prime factor handled directly; lengths with a larger one use Bluestein's method. */
    private static final int MAX_DIRECT_FACTOR = 61;

    /** In-place unscaled transform; {@code inverse} selects the sign of the exponent. */
    static void transform(double[] re, double[] im, boolean inverse) {
        int n = re.length;
        if (n <= 1) {
            return;
        }
        if ((n & (n - 1)) == 0) {
            radix2(re, im, inverse);
            return;
        }
        int[] factors = factor(n);
        if (factors[factors.length - 1] > MAX_DIRECT_FACTOR) {
            // Bluestein's convolution spreads the rounding of a large zero-frequency term across every output,
            // and plant signals often sit on a large offset. The transform is linear, so take the mean out first
            // and put it back as n times the mean in term 0, which is exact in either direction.
            double meanRe = Sums.mean(re, 0, n);
            double meanIm = Sums.mean(im, 0, n);
            for (int k = 0; k < n; k++) {
                re[k] -= meanRe;
                im[k] -= meanIm;
            }
            bluestein(re, im, inverse);
            re[0] += n * meanRe;
            im[0] += n * meanIm;
        } else {
            mixedRadix(re, im, inverse, factors);
        }
    }

    /** Prime factors of n in ascending order, with fours taken first so most passes are radix 4. */
    private static int[] factor(int n) {
        java.util.List<Integer> f = new java.util.ArrayList<>();
        int m = n;
        while (m % 4 == 0) {
            f.add(4);
            m /= 4;
        }
        for (int p = 2; (long) p * p <= m; p++) {
            while (m % p == 0) {
                f.add(p);
                m /= p;
            }
        }
        if (m > 1) {
            f.add(m);
        }
        int[] out = new int[f.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = f.get(i);
        }
        java.util.Arrays.sort(out);
        return out;
    }

    /**
     * Decimation-in-time Cooley-Tukey for any factorisation. Each pass combines p sub-transforms of length m
     * with an O(p) butterfly per output, using twiddles w^e = exp(-+2 pi i e / n) read from one table whose
     * entries are computed directly, so the error does not grow with the number of passes.
     */
    private static void mixedRadix(double[] re, double[] im, boolean inverse, int[] factors) {
        int n = re.length;
        double[] wr = new double[n];
        double[] wi = new double[n];
        double sign = inverse ? 1.0 : -1.0;
        double[] cs = new double[2];
        for (int k = 0; k < n; k++) {
            unitRoot(k, n, cs);
            wr[k] = cs[0];
            wi[k] = sign * cs[1];
        }
        double[] outR = new double[n];
        double[] outI = new double[n];
        dit(re, im, 0, 1, n, outR, outI, 0, factors, 0, wr, wi, 1, new double[n], new double[n]);
        System.arraycopy(outR, 0, re, 0, n);
        System.arraycopy(outI, 0, im, 0, n);
    }

    /**
     * Transform of the n inputs in[offset + j * stride] into out[outOff .. outOff + n). {@code wStride} maps
     * this level's exponents onto the full table; {@code tr}/{@code ti} are scratch space of the full length.
     */
    private static void dit(double[] inR, double[] inI, int offset, int stride, int n, double[] outR, double[] outI,
                            int outOff, int[] factors, int fi, double[] wr, double[] wi, int wStride,
                            double[] tr, double[] ti) {
        if (n == 1) {
            outR[outOff] = inR[offset];
            outI[outOff] = inI[offset];
            return;
        }
        int p = factors[fi];
        int m = n / p;
        for (int r = 0; r < p; r++) {
            dit(inR, inI, offset + r * stride, stride * p, m, outR, outI, outOff + r * m, factors, fi + 1,
                    wr, wi, wStride * p, tr, ti);
        }
        int full = wr.length;
        for (int k = 0; k < m; k++) {
            for (int q = 0; q < p; q++) {
                int kq = k + q * m;
                double sr = 0.0;
                double si = 0.0;
                for (int r = 0; r < p; r++) {
                    double yr = outR[outOff + r * m + k];
                    double yi = outI[outOff + r * m + k];
                    int e = (int) (((long) r * kq * wStride) % full);
                    double cr = wr[e];
                    double ci = wi[e];
                    sr += yr * cr - yi * ci;
                    si += yr * ci + yi * cr;
                }
                tr[outOff + kq] = sr;
                ti[outOff + kq] = si;
            }
        }
        System.arraycopy(tr, outOff, outR, outOff, n);
        System.arraycopy(ti, outOff, outI, outOff, n);
    }

    private static void radix2(double[] re, double[] im, boolean inverse) {
        int n = re.length;
        int levels = Integer.numberOfTrailingZeros(n);
        for (int k = 0; k < n; k++) {
            int j = Integer.reverse(k) >>> (32 - levels);
            if (j > k) {
                double t = re[k];
                re[k] = re[j];
                re[j] = t;
                t = im[k];
                im[k] = im[j];
                im[j] = t;
            }
        }
        double[] cos = new double[n / 2];
        double[] sin = new double[n / 2];
        double sign = inverse ? 1.0 : -1.0;
        double[] cs = new double[2];
        for (int k = 0; k < n / 2; k++) {
            unitRoot(k, n, cs);
            cos[k] = cs[0];
            sin[k] = sign * cs[1];
        }
        for (int size = 2; size <= n; size <<= 1) {
            int half = size / 2;
            int step = n / size;
            for (int start = 0; start < n; start += size) {
                for (int j = 0; j < half; j++) {
                    int a = start + j;
                    int b = a + half;
                    double wr = cos[j * step];
                    double wi = sin[j * step];
                    double xr = re[b] * wr - im[b] * wi;
                    double xi = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                }
            }
        }
    }

    private static void bluestein(double[] re, double[] im, boolean inverse) {
        int n = re.length;
        int m = Integer.highestOneBit(2 * n - 1);
        if (m < 2 * n - 1) {
            m <<= 1;
        }
        // Chirp w_k = exp(-i pi k^2 / n), with k^2 reduced mod 2n so the angle stays exact for long signals.
        double[] wr = new double[n];
        double[] wi = new double[n];
        double sign = inverse ? 1.0 : -1.0;
        long twoN = 2L * n;
        double[] cs = new double[2];
        for (int k = 0; k < n; k++) {
            // exp(-i pi k^2 / n) = exp(-2 pi i (k^2 mod 2n) / 2n)
            unitRoot(((long) k * k) % twoN, twoN, cs);
            wr[k] = cs[0];
            wi[k] = sign * cs[1];
        }
        double[] ar = new double[m];
        double[] ai = new double[m];
        for (int k = 0; k < n; k++) {
            ar[k] = re[k] * wr[k] - im[k] * wi[k];
            ai[k] = re[k] * wi[k] + im[k] * wr[k];
        }
        double[] br = new double[m];
        double[] bi = new double[m];
        br[0] = wr[0];
        bi[0] = -wi[0];
        for (int k = 1; k < n; k++) {
            br[k] = wr[k];
            bi[k] = -wi[k];
            br[m - k] = wr[k];
            bi[m - k] = -wi[k];
        }
        radix2(ar, ai, false);
        radix2(br, bi, false);
        for (int k = 0; k < m; k++) {
            double r = ar[k] * br[k] - ai[k] * bi[k];
            double i = ar[k] * bi[k] + ai[k] * br[k];
            ar[k] = r;
            ai[k] = i;
        }
        radix2(ar, ai, true);
        for (int k = 0; k < n; k++) {
            double cr = ar[k] / m;
            double ci = ai[k] / m;
            re[k] = cr * wr[k] - ci * wi[k];
            im[k] = cr * wi[k] + ci * wr[k];
        }
    }
}
