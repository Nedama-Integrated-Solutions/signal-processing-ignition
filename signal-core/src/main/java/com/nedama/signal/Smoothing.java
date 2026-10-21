package com.nedama.signal;

import java.math.BigInteger;
import java.util.Arrays;

/**
 * Savitzky-Golay smoothing and differentiation (scipy.signal.savgol_coeffs and savgol_filter) and median filtering
 * (scipy.signal.medfilt), from scipy 1.18.1.
 *
 * <p>The least-squares problems (the filter coefficients, and the polynomial fits at the edges in 'interp' mode)
 * are solved exactly in rational arithmetic and rounded once, so they are correctly rounded. scipy solves them with
 * LAPACK, whose error grows with the Vandermonde matrix's condition number; for window 5, order 2 it returns
 * -0.08571428571428574 and -0.08571428571428584 at the two ends, where the exact value is -3/35. The convolution
 * follows scipy.ndimage.convolve1d's C loop operation by operation, including its edge modes and its shortcut for
 * symmetric weights.
 */
public final class Smoothing {

    private Smoothing() {
    }

    // Savitzky-Golay coefficients

    /** float_factorial(deriv) / delta ** deriv, with Python's errors for a zero or overflowing power. */
    private static double derivScale(int deriv, double delta) {
        if (deriv < 0) {
            throw SignalException.valueError("factorial() not defined for negative values");
        }
        // scipy's float_factorial: the exact factorial rounded once, infinity from 171 up.
        double fact;
        if (deriv >= 171) {
            fact = Double.POSITIVE_INFINITY;
        } else {
            BigInteger f = BigInteger.ONE;
            for (int i = 2; i <= deriv; i++) {
                f = f.multiply(BigInteger.valueOf(i));
            }
            fact = f.doubleValue();
        }
        double power = Math.pow(delta, deriv);
        if (Double.isInfinite(power)) {
            throw SignalException.valueError("delta ** deriv is out of range");
        }
        if (power == 0.0) {
            throw SignalException.valueError("float division by zero (delta ** deriv is 0)");
        }
        return fact / power;
    }

    /** savgol_coeffs: the filter coefficients, correctly rounded. */
    public static double[] coeffs(int windowLength, int polyorder, int deriv, double delta, Double pos, String use) {
        if (polyorder >= windowLength) {
            throw SignalException.valueError("polyorder must be less than window_length.");
        }
        int halflen = Math.floorDiv(windowLength, 2);
        int rem = Math.floorMod(windowLength, 2);
        double p = pos != null ? pos : (rem == 0 ? halflen - 0.5 : halflen);
        if (!(0 <= p && p < windowLength)) {
            throw SignalException.valueError("pos must be nonnegative and less than window_length.");
        }
        if (!use.equals("conv") && !use.equals("dot")) {
            throw SignalException.valueError("`use` must be 'conv' or 'dot'");
        }
        if (deriv > polyorder) {
            if (windowLength < 0) {
                throw SignalException.valueError("negative dimensions are not allowed");
            }
            return new double[windowLength];
        }
        double yDeriv = derivScale(deriv, delta);
        // Each position x_j = -pos + j is a double, so x_j = m_j / 2^s exactly for integers m_j and one shift s.
        // With U[i][j] = m_j^i and D = diag(2^(s i)), the minimum-norm solution of A c = y (A[i][j] = x_j^i) is
        // c = U^T (U U^T)^-1 D y, so the large Gram matrix U U^T is built in integers and only the small solve
        // needs rational arithmetic.
        double[] xd = new double[windowLength];
        for (int i = 0; i < windowLength; i++) {
            xd[use.equals("conv") ? windowLength - 1 - i : i] = -p + i;
        }
        int shift = 0;
        for (double v : xd) {
            shift = Math.max(shift, fractionBits(v));
        }
        BigInteger[] m = new BigInteger[windowLength];
        for (int j = 0; j < windowLength; j++) {
            m[j] = new java.math.BigDecimal(xd[j]).multiply(new java.math.BigDecimal(BigInteger.ONE.shiftLeft(shift)))
                    .toBigIntegerExact();
        }
        int rows = polyorder + 1;
        BigInteger[][] u = new BigInteger[rows][windowLength];
        for (int j = 0; j < windowLength; j++) {
            BigInteger v = BigInteger.ONE;
            for (int i = 0; i < rows; i++) {
                u[i][j] = v;
                v = v.multiply(m[j]);
            }
        }
        Rational[][] g = new Rational[rows][rows];
        for (int i = 0; i < rows; i++) {
            for (int k = i; k < rows; k++) {
                BigInteger sum = BigInteger.ZERO;
                for (int j = 0; j < windowLength; j++) {
                    sum = sum.add(u[i][j].multiply(u[k][j]));
                }
                Rational r = Rational.of(sum, BigInteger.ONE);
                g[i][k] = r;
                g[k][i] = r;
            }
        }
        Rational[] e = new Rational[rows];
        Arrays.fill(e, Rational.ZERO);
        e[deriv] = Rational.of(yDeriv).multiply(Rational.of(BigInteger.ONE.shiftLeft(shift * deriv), BigInteger.ONE));
        Rational[] z = Rational.solve(g, e);
        // Put z over one denominator so each coefficient is an integer sum divided once.
        BigInteger den = BigInteger.ONE;
        for (Rational r : z) {
            den = den.divide(den.gcd(r.den)).multiply(r.den);
        }
        BigInteger[] zn = new BigInteger[rows];
        for (int i = 0; i < rows; i++) {
            zn[i] = z[i].num.multiply(den.divide(z[i].den));
        }
        double[] c = new double[windowLength];
        for (int j = 0; j < windowLength; j++) {
            BigInteger s = BigInteger.ZERO;
            for (int i = 0; i < rows; i++) {
                s = s.add(u[i][j].multiply(zn[i]));
            }
            c[j] = Rational.of(s, den).toDouble();
        }
        return c;
    }

    /** Number of binary digits after the point in a finite double (0 for an integer). */
    private static int fractionBits(double v) {
        if (v == 0 || v == Math.rint(v)) {
            return 0;
        }
        long bits = Double.doubleToRawLongBits(v);
        int exp = (int) ((bits >> 52) & 0x7ff);
        long mant = bits & 0xfffffffffffffL;
        if (exp == 0) {
            exp = 1;
        } else {
            mant |= 1L << 52;
        }
        int e = exp - 1075;
        return Math.max(0, -(e + Long.numberOfTrailingZeros(mant)));
    }

    // Savitzky-Golay filter

    /** savgol_filter for mode 'mirror', 'constant', 'nearest', 'wrap' or 'interp'. */
    public static double[] filter(double[] x, int windowLength, int polyorder, int deriv, double delta, String mode,
                                  double cval) {
        if (!mode.equals("mirror") && !mode.equals("constant") && !mode.equals("nearest") && !mode.equals("interp")
                && !mode.equals("wrap")) {
            throw SignalException.valueError("mode must be 'mirror', 'constant', 'nearest' 'wrap' or 'interp'.");
        }
        double[] c = coeffs(windowLength, polyorder, deriv, delta, null, "conv");
        if (mode.equals("interp")) {
            if (windowLength > x.length) {
                throw SignalException.valueError("If mode is 'interp', window_length must be less than or equal to "
                        + "the size of x.");
            }
            double[] y = convolve1d(x, c, "constant", 0.0);
            if (polyorder < 0) {
                throw SignalException.valueError("expected deg >= 0");
            }
            int halflen = windowLength / 2;
            fitEdge(x, 0, windowLength, 0, halflen, polyorder, deriv, delta, y);
            int n = x.length;
            fitEdge(x, n - windowLength, n, n - halflen, n, polyorder, deriv, delta, y);
            return y;
        }
        return convolve1d(x, c, mode, cval);
    }

    /**
     * scipy's _fit_edge: a least-squares polynomial through x[windowStart..windowStop), differentiated deriv times,
     * evaluated at interpStart..interpStop and divided by delta ** deriv. Solved exactly and rounded once.
     */
    private static void fitEdge(double[] x, int windowStart, int windowStop, int interpStart, int interpStop,
                                int polyorder, int deriv, double delta, double[] y) {
        int m = windowStop - windowStart;
        int order = polyorder + 1;
        // Normal equations in increasing powers: sum_t t^(i+k) c_k = sum_t t^i x_t.
        Rational[] powerSums = new Rational[2 * order - 1];
        Arrays.fill(powerSums, Rational.ZERO);
        Rational[] rhs = new Rational[order];
        Arrays.fill(rhs, Rational.ZERO);
        for (int t = 0; t < m; t++) {
            BigInteger tt = BigInteger.valueOf(t);
            BigInteger pw = BigInteger.ONE;
            Rational xv = Rational.of(x[windowStart + t]);
            for (int k = 0; k < 2 * order - 1; k++) {
                Rational r = Rational.of(pw, BigInteger.ONE);
                powerSums[k] = powerSums[k].add(r);
                if (k < order) {
                    rhs[k] = rhs[k].add(r.multiply(xv));
                }
                pw = pw.multiply(tt);
            }
        }
        Rational[][] g = new Rational[order][order];
        for (int i = 0; i < order; i++) {
            for (int k = 0; k < order; k++) {
                g[i][k] = powerSums[i + k];
            }
        }
        Rational[] c = Rational.solve(g, rhs);
        // Differentiate deriv times (c[k] is the coefficient of t^k).
        Rational[] d = c;
        for (int r = 0; r < deriv; r++) {
            if (d.length <= 1) {
                d = new Rational[] {Rational.ZERO};
                break;
            }
            Rational[] nd = new Rational[d.length - 1];
            for (int k = 1; k < d.length; k++) {
                nd[k - 1] = d[k].multiply(Rational.of(k));
            }
            d = nd;
        }
        double power = Math.pow(delta, deriv);
        if (Double.isInfinite(power)) {
            throw SignalException.valueError("delta ** deriv is out of range");
        }
        Rational div = Rational.of(power);
        for (int i = interpStart; i < interpStop; i++) {
            Rational t = Rational.of(i - windowStart);
            Rational v = Rational.ZERO;
            for (int k = d.length - 1; k >= 0; k--) {
                v = v.multiply(t).add(d[k]);
            }
            if (div.isZero()) {
                // numpy divides the values by 0.0: 0/0 is NaN, anything else infinite
                y[i] = v.isZero() ? Double.NaN : (v.num.signum() > 0 ? Double.POSITIVE_INFINITY
                        : Double.NEGATIVE_INFINITY);
            } else {
                y[i] = v.divide(div).toDouble();
            }
        }
    }

    /** scipy.ndimage's NI_ExtendLine for one line: {@code before} and {@code after} values around x. */
    static double[] extendLine(double[] x, int before, int after, String mode, double cval) {
        int n = x.length;
        double[] ext = new double[before + n + after];
        System.arraycopy(x, 0, ext, before, n);
        int first = before;
        int last = before + n;
        if (n == 1 && mode.equals("mirror")) {
            mode = "nearest";
        }
        switch (mode) {
            case "nearest": {
                for (int i = 0; i < before; i++) {
                    ext[i] = ext[first];
                }
                for (int i = 0; i < after; i++) {
                    ext[last + i] = ext[last - 1];
                }
                break;
            }
            case "wrap": {
                int src = last - 1;
                int dst = first - 1;
                for (int i = 0; i < before; i++) {
                    ext[dst--] = ext[src--];
                }
                src = first;
                dst = last;
                for (int i = 0; i < after; i++) {
                    ext[dst++] = ext[src++];
                }
                break;
            }
            case "mirror": {
                int src = first + 1;
                int dst = first - 1;
                int sb = before;
                while (sb > 0 && src < last) {
                    ext[dst--] = ext[src++];
                    sb--;
                }
                src = last - 2;
                while (sb-- > 0) {
                    ext[dst--] = ext[src--];
                }
                src = last - 2;
                dst = last;
                int sa = after;
                while (sa > 0 && src >= first) {
                    ext[dst++] = ext[src--];
                    sa--;
                }
                src = first + 1;
                while (sa-- > 0) {
                    ext[dst++] = ext[src++];
                }
                break;
            }
            default: {
                for (int i = 0; i < before; i++) {
                    ext[i] = cval;
                }
                for (int i = 0; i < after; i++) {
                    ext[last + i] = cval;
                }
            }
        }
        return ext;
    }

    /** scipy.ndimage.convolve1d (origin 0) for one line, following NI_Correlate1D's loops. */
    static double[] convolve1d(double[] x, double[] coeffs, String mode, double cval) {
        int n = x.length;
        if (n == 0) {
            return new double[0];
        }
        int size = coeffs.length;
        double[] fw = new double[size];
        for (int i = 0; i < size; i++) {
            fw[i] = coeffs[size - 1 - i];
        }
        int origin = (size & 1) == 0 ? -1 : 0;
        int size1 = size / 2;
        int size2 = size - size1 - 1;
        int symmetric = 0;
        if ((size & 1) == 1) {
            symmetric = 1;
            for (int i = 1; i <= size / 2; i++) {
                if (Math.abs(fw[i + size1] - fw[size1 - i]) > Math.ulp(1.0)) {
                    symmetric = 0;
                    break;
                }
            }
            if (symmetric == 0) {
                symmetric = -1;
                for (int i = 1; i <= size / 2; i++) {
                    if (Math.abs(fw[size1 + i] + fw[size1 - i]) > Math.ulp(1.0)) {
                        symmetric = 0;
                        break;
                    }
                }
            }
        }
        double[] ext = extendLine(x, size1 + origin, size2 - origin, mode, cval);
        double[] out = new double[n];
        for (int ll = 0; ll < n; ll++) {
            int b = size1 + ll;
            double s;
            if (symmetric > 0) {
                s = ext[b] * fw[size1];
                for (int jj = -size1; jj < 0; jj++) {
                    s += (ext[b + jj] + ext[b - jj]) * fw[size1 + jj];
                }
            } else if (symmetric < 0) {
                s = ext[b] * fw[size1];
                for (int jj = -size1; jj < 0; jj++) {
                    s += (ext[b + jj] - ext[b - jj]) * fw[size1 + jj];
                }
            } else {
                s = ext[b + size2] * fw[size1 + size2];
                for (int jj = -size1; jj < size2; jj++) {
                    s += ext[b + jj] * fw[size1 + jj];
                }
            }
            out[ll] = s;
        }
        return out;
    }

    // Median filter

    /** medfilt for floats: the median of each window of kernel samples, zero outside the signal. */
    public static double[] medfilt(double[] x, int kernel) {
        int n = x.length;
        int h = kernel / 2;
        double[] window = new double[kernel];
        for (int k = 0; k < kernel; k++) {
            int i = k - h;
            window[k] = i >= 0 && i < n ? x[i] : 0.0;
        }
        Arrays.sort(window);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = window[h];
            if (i + 1 < n) {
                int leaving = i - h;
                int entering = i + h + 1;
                double outVal = leaving >= 0 && leaving < n ? x[leaving] : 0.0;
                double inVal = entering < n ? x[entering] : 0.0;
                replace(window, outVal, inVal);
            }
        }
        return out;
    }

    /** medfilt for integers, which keeps integer values as scipy does. */
    public static long[] medfilt(long[] x, int kernel) {
        int n = x.length;
        int h = kernel / 2;
        long[] window = new long[kernel];
        for (int k = 0; k < kernel; k++) {
            int i = k - h;
            window[k] = i >= 0 && i < n ? x[i] : 0L;
        }
        Arrays.sort(window);
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            out[i] = window[h];
            if (i + 1 < n) {
                int leaving = i - h;
                int entering = i + h + 1;
                long outVal = leaving >= 0 && leaving < n ? x[leaving] : 0L;
                long inVal = entering < n ? x[entering] : 0L;
                replace(window, outVal, inVal);
            }
        }
        return out;
    }

    /** Removes one copy of outVal from the sorted window and inserts inVal, keeping it sorted. */
    private static void replace(double[] w, double outVal, double inVal) {
        // Arrays.sort and binarySearch order -0.0 before 0.0, so this must compare the same way: with ==, removing
        // a 0.0 could take a -0.0 instead, which then cannot be found when it leaves the window.
        int i = Arrays.binarySearch(w, outVal);
        if (i < 0) {
            throw new IllegalStateException("medfilt window lost track of a value");
        }
        while (i > 0 && Double.compare(w[i - 1], outVal) == 0) {
            i--;
        }
        System.arraycopy(w, i + 1, w, i, w.length - i - 1);
        int j = Arrays.binarySearch(w, 0, w.length - 1, inVal);
        if (j < 0) {
            j = -j - 1;
        }
        System.arraycopy(w, j, w, j + 1, w.length - 1 - j);
        w[j] = inVal;
    }

    private static void replace(long[] w, long outVal, long inVal) {
        int i = Arrays.binarySearch(w, outVal);
        while (i > 0 && w[i - 1] == outVal) {
            i--;
        }
        System.arraycopy(w, i + 1, w, i, w.length - i - 1);
        int j = Arrays.binarySearch(w, 0, w.length - 1, inVal);
        if (j < 0) {
            j = -j - 1;
        }
        System.arraycopy(w, j, w, j + 1, w.length - 1 - j);
        w[j] = inVal;
    }
}
