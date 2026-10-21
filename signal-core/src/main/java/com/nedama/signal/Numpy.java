package com.nedama.signal;

/**
 * numpy's reduction orders, for ports where the order of additions matters. numpy sums arrays pairwise: below
 * 8 values in a plain loop, otherwise with 8 running sums (4 for complex) combined as a tree, recursing above 128.
 * Matching this keeps sums within a rounding or two of scipy, which matters for its convergence tests such as
 * _bessel_zeros checking that the zeros sum to -1 within 1e-15.
 */
final class Numpy {

    private Numpy() {
    }

    /** numpy.sum of x[from..to). */
    static double sum(double[] x, int from, int to) {
        int n = to - from;
        if (n < 8) {
            double res = 0.0;
            for (int i = from; i < to; i++) {
                res += x[i];
            }
            return res;
        }
        if (n <= 128) {
            double[] r = new double[8];
            for (int j = 0; j < 8; j++) {
                r[j] = x[from + j];
            }
            int i;
            for (i = 8; i < n - (n % 8); i += 8) {
                for (int j = 0; j < 8; j++) {
                    r[j] += x[from + i + j];
                }
            }
            double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for (; i < n; i++) {
                res += x[from + i];
            }
            return res;
        }
        int n2 = n / 2;
        n2 -= n2 % 8;
        return sum(x, from, from + n2) + sum(x, from + n2, to);
    }

    static double sum(double[] x) {
        return sum(x, 0, x.length);
    }

    /** numpy.sum of complex values z[from..to); numpy counts the real and imaginary parts as 2 values each. */
    static Complex sum(Complex[] z, int from, int to) {
        int count = to - from;
        int n = 2 * count;
        if (n < 8) {
            double rr = 0.0;
            double ri = 0.0;
            for (int i = from; i < to; i++) {
                rr += z[i].re;
                ri += z[i].im;
            }
            return new Complex(rr, ri);
        }
        if (n <= 128) {
            double[] ar = new double[4];
            double[] ai = new double[4];
            for (int j = 0; j < 4; j++) {
                ar[j] = z[from + j].re;
                ai[j] = z[from + j].im;
            }
            int i;
            int full = count - (count % 4);
            for (i = 4; i < full; i += 4) {
                for (int j = 0; j < 4; j++) {
                    ar[j] += z[from + i + j].re;
                    ai[j] += z[from + i + j].im;
                }
            }
            double rr = (ar[0] + ar[1]) + (ar[2] + ar[3]);
            double ri = (ai[0] + ai[1]) + (ai[2] + ai[3]);
            for (; i < count; i++) {
                rr += z[from + i].re;
                ri += z[from + i].im;
            }
            return new Complex(rr, ri);
        }
        int n2 = n / 2;
        n2 -= n2 % 8;
        int c2 = n2 / 2;
        return sum(z, from, from + c2).plus(sum(z, from + c2, to));
    }

    static Complex sum(Complex[] z) {
        return sum(z, 0, z.length);
    }

    /** numpy.prod of complex values, a plain left-to-right product (1 for an empty array). */
    static Complex prod(Complex[] z) {
        Complex p = Complex.ONE;
        for (Complex v : z) {
            p = p.times(v);
        }
        return p;
    }

    /** numpy.prod of real values. */
    static double prod(double[] x) {
        double p = 1.0;
        for (double v : x) {
            p *= v;
        }
        return p;
    }

    /** Inverse hyperbolic sine, accurate to about one rounding. */
    static double asinh(double x) {
        double a = Math.abs(x);
        double r;
        if (a > 1e8) {
            r = Math.log(a) + Math.log(2.0);
        } else {
            r = Math.log1p(a + a * a / (1.0 + Math.sqrt(1.0 + a * a)));
        }
        return Math.copySign(r, x);
    }
}
