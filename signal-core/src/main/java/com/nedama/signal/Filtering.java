package com.nedama.signal;

import java.util.Arrays;

/**
 * Linear filtering, ported from scipy.signal (scipy 1.18.1): lfilter (the C++ direct form II transposed loop, or a
 * convolution when a has one coefficient, as scipy does), lfilter_zi, sosfilt (the Cython section loop), sosfilt_zi,
 * and the forward-backward filters filtfilt and sosfiltfilt with odd, even or constant edge extension. Every loop
 * performs scipy's operations in scipy's order, so results agree with scipy to the last bit on platforms where
 * scipy's compiled code does not fuse multiply-adds.
 */
public final class Filtering {

    private Filtering() {
    }

    /** The output and, when initial conditions were given, the final conditions. */
    public static final class Result {
        public final double[] y;
        public final double[] zf;

        Result(double[] y, double[] zf) {
            this.y = y;
            this.zf = zf;
        }
    }

    /** Output and final section states of sosfilt. */
    public static final class SosResult {
        public final double[] y;
        public final double[][] zf;

        SosResult(double[] y, double[][] zf) {
            this.y = y;
            this.zf = zf;
        }
    }

    /**
     * lfilter. {@code zi} may be null; with zi, it must have max(len(a), len(b)) - 1 values and the final conditions
     * are returned.
     */
    public static Result lfilter(double[] b, double[] a, double[] x, double[] zi) {
        if (b.length == 0) {
            throw SignalException.valueError("Parameter b is not a non-empty 1d array");
        }
        if (a.length == 0) {
            throw SignalException.valueError("Parameter a is not a non-empty 1d array");
        }
        if (a.length == 1) {
            return lfilterFir(b, a[0], x, zi);
        }
        // The C++ driver compares a[0] with zero bytewise, so -0.0 is not caught there.
        if (Double.doubleToRawLongBits(a[0]) == 0L) {
            throw SignalException.valueError("BUG: filter coefficient a[0] == 0 not supported yet");
        }
        int nfilt = Math.max(a.length, b.length);
        if (zi != null && zi.length != nfilt - 1) {
            throw SignalException.valueError("Unexpected shape for zi: expected (" + (nfilt - 1) + ",), found ("
                    + zi.length + ",).");
        }
        double[] bb = Arrays.copyOf(b, nfilt);
        double[] aa = Arrays.copyOf(a, nfilt);
        double a0 = aa[0];
        for (int n = 0; n < nfilt; n++) {
            bb[n] /= a0;
            aa[n] /= a0;
        }
        double[] z = zi == null ? new double[nfilt - 1] : zi.clone();
        double[] y = new double[x.length];
        for (int k = 0; k < x.length; k++) {
            double xn = x[k];
            double yn = z[0] + bb[0] * xn;
            for (int n = 0; n < nfilt - 2; n++) {
                z[n] = z[n + 1] + xn * bb[n + 1] - yn * aa[n + 1];
            }
            z[nfilt - 2] = xn * bb[nfilt - 1] - yn * aa[nfilt - 1];
            y[k] = yn;
        }
        return new Result(y, zi == null ? null : z);
    }

    /** scipy's path for a single denominator coefficient: b / a[0] convolved with x, plus zi at the start. */
    private static Result lfilterFir(double[] b, double a0, double[] x, double[] zi) {
        if (x.length == 0) {
            throw SignalException.valueError("v cannot be empty");
        }
        if (zi != null && zi.length != b.length - 1) {
            throw SignalException.valueError("Unexpected shape for parameter zi: expected (" + (b.length - 1)
                    + ",), found (" + zi.length + ",).");
        }
        double[] bb = new double[b.length];
        for (int i = 0; i < b.length; i++) {
            bb[i] = b[i] / a0;
        }
        int full = x.length + b.length - 1;
        double[] out = new double[full];
        for (int n = 0; n < full; n++) {
            int kLo = Math.max(0, n - x.length + 1);
            int kHi = Math.min(b.length - 1, n);
            double s = 0.0;
            for (int k = kLo; k <= kHi; k++) {
                s += bb[k] * x[n - k];
            }
            out[n] = s;
        }
        if (zi != null) {
            for (int i = 0; i < zi.length; i++) {
                out[i] += zi[i];
            }
        }
        double[] y = Arrays.copyOf(out, x.length);
        double[] zf = zi == null ? null : Arrays.copyOfRange(out, x.length, full);
        return new Result(y, zf);
    }

    /** Initial conditions for lfilter's steady state under a unit step (scipy 1.18's closed form). */
    public static double[] lfilterZi(double[] bIn, double[] aIn) {
        if (bIn.length == 0 || aIn.length == 0) {
            throw SignalException.valueError("b and a must not be empty");
        }
        double[] b = bIn.clone();
        double[] a = aIn.clone();
        if (a[0] == 0) {
            throw SignalException.valueError("First coefficient of parameter `a` must be non-zero!");
        }
        if (a[0] != 1) {
            double a0 = a[0];
            for (int i = 0; i < b.length; i++) {
                b[i] = b[i] / a0;
            }
            for (int i = 0; i < a.length; i++) {
                a[i] = a[i] / a0;
            }
        }
        double sumA = Numpy.sum(a);
        if (sumA == 0) {
            throw SignalException.valueError("Filter not stable due to sum(a) == 0, i.e., having a pole at z = 1!");
        }
        double yInf = Numpy.sum(b) / sumA;
        int n = Math.max(a.length, b.length);
        double[] bp = Arrays.copyOf(b, n);
        double[] ap = Arrays.copyOf(a, n);
        double[] out = new double[n - 1];
        double acc = 0.0;
        for (int i = n - 1; i >= 1; i--) {
            double d = bp[i] - yInf * ap[i];
            acc = i == n - 1 ? d : acc + d;
            out[i - 1] = acc;
        }
        return out;
    }

    /** Checks second-order sections for filtering: rows of 6 with sos[:, 3] exactly 1. */
    static void validateSos(double[][] sos) {
        Freqz.validateSos(sos);
    }

    /** sosfilt. {@code zi} may be null; with zi (one row of 2 per section), the final states are returned. */
    public static SosResult sosfilt(double[][] sos, double[] x, double[][] zi) {
        validateSos(sos);
        int ns = sos.length;
        double[][] z = new double[ns][2];
        if (zi != null) {
            if (zi.length != ns) {
                throw SignalException.valueError("Invalid zi shape: zi must have shape (" + ns + ", 2), got ("
                        + zi.length + ", " + (zi.length > 0 ? zi[0].length : 0) + ").");
            }
            for (int s = 0; s < ns; s++) {
                if (zi[s].length != 2) {
                    throw SignalException.valueError("Invalid zi shape: zi must have shape (" + ns + ", 2).");
                }
                z[s][0] = zi[s][0];
                z[s][1] = zi[s][1];
            }
        }
        double[] y = new double[x.length];
        for (int n = 0; n < x.length; n++) {
            double xCur = x[n];
            for (int s = 0; s < ns; s++) {
                double[] c = sos[s];
                double[] zs = z[s];
                double xNew = c[0] * xCur + zs[0];
                zs[0] = c[1] * xCur - c[4] * xNew + zs[1];
                zs[1] = c[2] * xCur - c[5] * xNew;
                xCur = xNew;
            }
            y[n] = xCur;
        }
        return new SosResult(y, zi == null ? null : z);
    }

    /** Initial section states for sosfilt's steady state under a unit step. */
    public static double[][] sosfiltZi(double[][] sos) {
        for (double[] row : sos) {
            if (row.length != 6) {
                throw SignalException.valueError("sos must be shape (n_sections, 6)");
            }
        }
        if (sos.length == 0) {
            throw SignalException.valueError("sos must be shape (n_sections, 6)");
        }
        double[][] zi = new double[sos.length][];
        double scale = 1.0;
        for (int s = 0; s < sos.length; s++) {
            double[] b = Arrays.copyOfRange(sos[s], 0, 3);
            double[] a = Arrays.copyOfRange(sos[s], 3, 6);
            double[] z = lfilterZi(b, a);
            zi[s] = new double[] {scale * z[0], scale * z[1]};
            scale *= Numpy.sum(b) / Numpy.sum(a);
        }
        return zi;
    }

    /** Edge extension types for the forward-backward filters. */
    public enum PadType { ODD, EVEN, CONSTANT, NONE }

    /** scipy's _validate_pad: returns the edge length; {@code padlen} null means 3 * ntaps. */
    private static int edge(PadType padtype, Long padlen, int length, int ntaps) {
        long edge;
        if (padtype == PadType.NONE) {
            edge = 0;
        } else if (padlen == null) {
            edge = 3L * ntaps;
        } else {
            edge = padlen;
        }
        if (length <= edge) {
            throw SignalException.valueError("The length of the input vector x must be greater than padlen, which is "
                    + edge + ".");
        }
        return (int) Math.max(edge, 0);
    }

    private static double[] extend(double[] x, int n, PadType padtype) {
        if (n < 1 || padtype == PadType.NONE) {
            return x;
        }
        int len = x.length;
        double[] ext = new double[len + 2 * n];
        double first = x[0];
        double last = x[len - 1];
        for (int i = 0; i < n; i++) {
            double left = x[n - i];
            double right = x[len - 2 - i];
            switch (padtype) {
                case ODD:
                    ext[i] = 2 * first - left;
                    ext[len + n + i] = 2 * last - right;
                    break;
                case EVEN:
                    ext[i] = left;
                    ext[len + n + i] = right;
                    break;
                default:
                    ext[i] = first;
                    ext[len + n + i] = last;
            }
        }
        System.arraycopy(x, 0, ext, n, len);
        return ext;
    }

    private static double[] reverse(double[] v) {
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[v.length - 1 - i];
        }
        return out;
    }

    private static double[] scaled(double[] v, double s) {
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i] * s;
        }
        return out;
    }

    private static double[][] scaled(double[][] v, double s) {
        double[][] out = new double[v.length][];
        for (int i = 0; i < v.length; i++) {
            out[i] = scaled(v[i], s);
        }
        return out;
    }

    /** filtfilt with method 'pad': zero-phase forward-backward filtering. */
    public static double[] filtfilt(double[] b, double[] a, double[] x, PadType padtype, Long padlen) {
        int e = edge(padtype, padlen, x.length, Math.max(a.length, b.length));
        double[] ext = extend(x, e, padtype);
        double[] zi = lfilterZi(b, a);
        double[] y = lfilter(b, a, ext, scaled(zi, ext[0])).y;
        double y0 = y[y.length - 1];
        y = reverse(lfilter(b, a, reverse(y), scaled(zi, y0)).y);
        return e > 0 ? Arrays.copyOfRange(y, e, y.length - e) : y;
    }

    /** sosfiltfilt: zero-phase forward-backward filtering with second-order sections. */
    public static double[] sosfiltfilt(double[][] sos, double[] x, PadType padtype, Long padlen) {
        validateSos(sos);
        int zerosB = 0;
        int zerosA = 0;
        for (double[] row : sos) {
            zerosB += row[2] == 0 ? 1 : 0;
            zerosA += row[5] == 0 ? 1 : 0;
        }
        int ntaps = 2 * sos.length + 1 - Math.min(zerosB, zerosA);
        int e = edge(padtype, padlen, x.length, ntaps);
        double[] ext = extend(x, e, padtype);
        double[][] zi = sosfiltZi(sos);
        double[] y = sosfilt(sos, ext, scaled(zi, ext[0])).y;
        double y0 = y[y.length - 1];
        y = reverse(sosfilt(sos, reverse(y), scaled(zi, y0)).y);
        return e > 0 ? Arrays.copyOfRange(y, e, y.length - e) : y;
    }
}
