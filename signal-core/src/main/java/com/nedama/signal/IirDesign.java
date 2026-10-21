package com.nedama.signal;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.UnaryOperator;

/**
 * IIR filter design, ported from scipy.signal._filter_design (scipy 1.18.1): the analog prototypes (buttap,
 * cheb1ap, cheb2ap, ellipap, besselap), the frequency transforms, the bilinear transform, and the conversions to
 * transfer-function and second-order-section form. Each step follows scipy's arithmetic, including numpy's
 * complex division and summation order, so designs agree with scipy to within a few roundings.
 */
public final class IirDesign {

    private IirDesign() {
    }

    /** scipy's EPSILON in _filter_design. */
    private static final double EPSILON = 2e-16;

    /** Zeros, poles and gain. {@code zReal} mirrors scipy's dtype: true when scipy keeps the zeros in a real array. */
    public static final class Zpk {
        public final Complex[] z;
        public final boolean zReal;
        public final Complex[] p;
        public final double k;

        Zpk(Complex[] z, boolean zReal, Complex[] p, double k) {
            this.z = z;
            this.zReal = zReal;
            this.p = p;
            this.k = k;
        }
    }

    /** A polynomial's coefficients, highest power first; real when the roots come in exact conjugate pairs. */
    public static final class Poly {
        public final double[] real;
        public final Complex[] complex;

        Poly(double[] real, Complex[] complex) {
            this.real = real;
            this.complex = complex;
        }

        public boolean isReal() {
            return real != null;
        }
    }

    // Analog prototypes

    static Zpk buttap(int n) {
        Complex[] p = new Complex[n];
        // numpy: 1j * pi * m / (2 * N), the division done as a complex multiply by 1 / (2N)
        Complex jpi = new Complex(0.0, Math.PI);
        for (int i = 0; i < n; i++) {
            double m = -n + 1 + 2.0 * i;
            Complex theta = jpi.times(new Complex(m, 0.0)).divReal(2.0 * n);
            p[i] = theta.exp().negate();
        }
        return new Zpk(new Complex[0], true, p, 1.0);
    }

    static Zpk cheb1ap(int n, double rp) {
        if (n == 0) {
            return new Zpk(new Complex[0], true, new Complex[0], Math.pow(10, -rp / 20));
        }
        double eps = Math.sqrt(Math.pow(10, 0.1 * rp) - 1.0);
        if (eps == 0.0) {
            // scipy: 1 / eps raises ZeroDivisionError for rp = 0
            throw SignalException.valueError("cheby1(): rp must be greater than 0");
        }
        double mu = 1.0 / n * Numpy.asinh(1 / eps);
        Complex[] p = new Complex[n];
        for (int i = 0; i < n; i++) {
            double m = -n + 1 + 2.0 * i;
            double theta = Math.PI * m / (2 * n);
            p[i] = new Complex(mu, theta).sinh().negate();
        }
        Complex[] negP = negateAll(p);
        double k = Numpy.prod(negP).re;
        if (n % 2 == 0) {
            k = k / Math.sqrt(1 + eps * eps);
        }
        return new Zpk(new Complex[0], true, p, k);
    }

    static Zpk cheb2ap(int n, double rs) {
        if (n == 0) {
            return new Zpk(new Complex[0], false, new Complex[0], 1.0);
        }
        double root = Math.sqrt(Math.pow(10, 0.1 * rs) - 1);
        if (root == 0.0) {
            // scipy: 1.0 / sqrt(...) raises ZeroDivisionError for rs = 0
            throw SignalException.valueError("cheby2(): rs must be greater than 0");
        }
        double de = 1.0 / root;
        double mu = Numpy.asinh(1.0 / de) / n;
        List<Double> ms = new ArrayList<>();
        for (int m = -n + 1; m < n; m += 2) {
            if (n % 2 == 1 && m == 0) {
                continue;
            }
            ms.add((double) m);
        }
        Complex[] z = new Complex[ms.size()];
        for (int i = 0; i < z.length; i++) {
            double s = Math.sin(ms.get(i) * Math.PI / (2 * n));
            z[i] = new Complex(0.0, 1.0).divReal(s);
        }
        Complex[] p = new Complex[n];
        for (int i = 0; i < n; i++) {
            double m1 = -n + 1 + 2.0 * i;
            double theta1 = Math.PI * m1 / (2 * n);
            p[i] = new Complex(-1.0, 0.0).div(new Complex(mu, theta1).sinh());
        }
        double k = Numpy.prod(negateAll(p)).div(Numpy.prod(negateAll(z))).re;
        return new Zpk(z, false, p, k);
    }

    static Zpk ellipap(int n, double rp, double rs) {
        if (n == 0) {
            return new Zpk(new Complex[0], false, new Complex[0], Math.pow(10, -rp / 20));
        }
        if (n == 1) {
            double d = pow10m1(0.1 * rp);
            if (d == 0.0) {
                throw SignalException.valueError("ellip(): rp must be greater than 0");
            }
            double p = -Math.sqrt(1.0 / d);
            return new Zpk(new Complex[0], false, new Complex[] {new Complex(p, 0.0)}, -p);
        }
        double epsSq = pow10m1(0.1 * rp);
        double eps = Math.sqrt(epsSq);
        double rsTerm = pow10m1(0.1 * rs);
        if (rsTerm == 0.0) {
            // scipy: eps_sq / 0.0 raises ZeroDivisionError for rs = 0
            throw SignalException.valueError("ellip(): rs must be greater than 0");
        }
        double ck1Sq = epsSq / rsTerm;
        if (ck1Sq == 0) {
            throw SignalException.valueError("Cannot design a filter with given rp and rs specifications.");
        }
        double val0 = Special.ellipk(ck1Sq);
        double m = ellipdeg(n, ck1Sq);
        double capk = Special.ellipk(m);

        int jj = 0;
        for (int j = 1 - n % 2; j < n; j += 2) {
            jj++;
        }
        double[] s = new double[jj];
        double[] c = new double[jj];
        double[] d = new double[jj];
        int idx = 0;
        for (int j = 1 - n % 2; j < n; j += 2) {
            double[] r = Special.ellipj(j * capk / n, m);
            s[idx] = r[0];
            c[idx] = r[1];
            d[idx] = r[2];
            idx++;
        }
        List<Complex> zl = new ArrayList<>();
        double sqrtM = Math.sqrt(m);
        for (double sv : s) {
            if (Math.abs(sv) > EPSILON) {
                double zv = 1.0 / (sqrtM * sv);
                zl.add(new Complex(0.0, 1.0).times(new Complex(zv, 0.0)));
            }
        }
        Complex[] z = new Complex[2 * zl.size()];
        for (int i = 0; i < zl.size(); i++) {
            z[i] = zl.get(i);
            z[zl.size() + i] = zl.get(i).conj();
        }

        double r = arcJacSc1(1. / eps, ck1Sq);
        double v0 = capk * r / (n * val0);
        double[] e = Special.ellipj(v0, 1 - m);
        double sv = e[0];
        double cv = e[1];
        double dv = e[2];
        Complex[] p = new Complex[jj];
        for (int i = 0; i < jj; i++) {
            double ds = d[i] * sv;
            Complex num = new Complex(c[i] * d[i] * sv * cv, s[i] * dv).negate();
            p[i] = num.divReal(1 - ds * ds);
        }
        Complex[] pAll;
        if (n % 2 == 1) {
            Complex[] pp = new Complex[jj];
            for (int i = 0; i < jj; i++) {
                pp[i] = p[i].times(p[i].conj());
            }
            double limit = EPSILON * Math.sqrt(Numpy.sum(pp).re);
            List<Complex> all = new ArrayList<>(Arrays.asList(p));
            for (Complex v : p) {
                if (Math.abs(v.im) > limit) {
                    all.add(v.conj());
                }
            }
            pAll = all.toArray(new Complex[0]);
        } else {
            pAll = new Complex[2 * jj];
            for (int i = 0; i < jj; i++) {
                pAll[i] = p[i];
                pAll[jj + i] = p[i].conj();
            }
        }
        double k = Numpy.prod(negateAll(pAll)).div(Numpy.prod(negateAll(z))).re;
        if (n % 2 == 0) {
            k = k / Math.sqrt(1 + epsSq);
        }
        return new Zpk(z, false, pAll, k);
    }

    static Zpk besselap(int n, String norm) {
        if (n == 0) {
            return new Zpk(new Complex[0], true, new Complex[0], 1.0);
        }
        Complex[] zeros = besselZeros(n);
        Complex[] p = new Complex[n];
        for (int i = 0; i < n; i++) {
            p[i] = Complex.ONE.div(zeros[i]);
        }
        BigInteger aLast = fallingFactorial(2 * n, n).shiftRight(n);
        double k;
        if (norm.equals("delay") || norm.equals("mag")) {
            k = aLast.doubleValue();
            if (norm.equals("mag")) {
                double nf = normFactor(p, aLast.doubleValue());
                for (int i = 0; i < n; i++) {
                    p[i] = p[i].divReal(nf);
                }
                k = Math.pow(nf, -n) * aLast.doubleValue();
            }
        } else if (norm.equals("phase")) {
            double scale = Math.pow(10, -Math.log10(aLast.doubleValue()) / n);
            for (int i = 0; i < n; i++) {
                p[i] = p[i].times(new Complex(scale, 0.0));
            }
            k = 1;
        } else {
            throw SignalException.valueError("normalization not understood");
        }
        return new Zpk(new Complex[0], true, p, k);
    }

    // Bessel helpers

    static BigInteger fallingFactorial(int x, int n) {
        BigInteger v = BigInteger.ONE;
        for (int k = x - n + 1; k <= x; k++) {
            v = v.multiply(BigInteger.valueOf(k));
        }
        return v;
    }

    /** numpy.polynomial.polynomial.polyval: coefficients in increasing order, Horner from the last. */
    private static double nppPolyval(double x, double[] c) {
        double c0 = c[c.length - 1];
        for (int i = 2; i <= c.length; i++) {
            c0 = c[c.length - i] + c0 * x;
        }
        return c0;
    }

    /** Campos and Calderon's starting points for the Bessel polynomial zeros. */
    static Complex[] camposZeros(int n) {
        if (n == 1) {
            return new Complex[] {new Complex(-1, 0)};
        }
        double s = nppPolyval(n, new double[] {0, 0, 2, 0, -3, 1});
        double b3 = nppPolyval(n, new double[] {16, -8}) / s;
        double b2 = nppPolyval(n, new double[] {-24, -12, 12}) / s;
        double b1 = nppPolyval(n, new double[] {8, 24, -12, -2}) / s;
        double b0 = nppPolyval(n, new double[] {0, -6, 0, 5, -1}) / s;
        double r = nppPolyval(n, new double[] {0, 0, 2, 1});
        double a1 = nppPolyval(n, new double[] {-6, -6}) / r;
        double a2 = 6 / r;
        Complex[] out = new Complex[n];
        for (int k = 1; k <= n; k++) {
            double x = nppPolyval(k, new double[] {0, a1, a2});
            double y = nppPolyval(k, new double[] {b0, b1, b2, b3});
            out[k - 1] = new Complex(x, y);
        }
        return out;
    }

    /**
     * kve(m + 1/2, 1/x): the exponentially scaled modified Bessel function of the second kind at half-integer order,
     * at z = 1/x. At these orders it has the exact form sqrt(pi x / 2) * sum_k c_k (x/2)^k with integer
     * c_k = (m + k)! / (k! (m - k)!). scipy evaluates it with the AMOS library, which stays accurate near its zeros.
     * A double-precision sum of these terms does not: near a zero the terms cancel, leaving a rounding floor (1e-15
     * at m = 7, 1e-12 at m = 12) above scipy's 1e-15 stopping test, so the root finder never converges. So the
     * sum is taken in double-double arithmetic (about 32 digits) up to m = 30, which measured within 1.1 eps of
     * scipy's zeros to m = 36, and exactly in decimal arithmetic with 34 + m digits above that, up to scipy's own
     * limit. x/2 and the coefficients are exact, and the sum is rounded once. The coefficients are built once per
     * order, since the root finder evaluates the function thousands of times.
     */
    static final class KveHalf {
        private static final int DOUBLE_DOUBLE_MAX = 30;
        private final int m;
        private final double[][] ddCoef;
        private final java.math.BigDecimal[] decCoef;
        private final java.math.MathContext mc;

        KveHalf(int m) {
            this.m = m;
            BigInteger[] c = coefficients(m);
            if (m <= DOUBLE_DOUBLE_MAX) {
                ddCoef = new double[m + 1][];
                for (int k = 0; k <= m; k++) {
                    ddCoef[k] = dd(c[k]);
                }
                decCoef = null;
                mc = null;
            } else {
                ddCoef = null;
                decCoef = new java.math.BigDecimal[m + 1];
                for (int k = 0; k <= m; k++) {
                    decCoef[k] = new java.math.BigDecimal(c[k]);
                }
                mc = new java.math.MathContext(34 + m);
            }
        }

        Complex at(Complex x) {
            Complex sum = ddCoef != null ? sumDoubleDouble(x) : sumDecimal(x);
            Complex pre = new Complex(Math.PI * x.re / 2, Math.PI * x.im / 2).sqrt();
            return pre.times(sum);
        }

        private Complex sumDoubleDouble(Complex x) {
            double wr = x.re / 2;
            double wi = x.im / 2;
            double[] yr = ddCoef[m];
            double[] yi = {0.0, 0.0};
            for (int k = m - 1; k >= 0; k--) {
                double[] nr = ddAdd(ddMul(yr, wr), ddMul(yi, -wi));
                double[] ni = ddAdd(ddMul(yr, wi), ddMul(yi, wr));
                yr = ddAdd(nr, ddCoef[k]);
                yi = ni;
            }
            return new Complex(yr[0] + yr[1], yi[0] + yi[1]);
        }

        private Complex sumDecimal(Complex x) {
            java.math.BigDecimal two = java.math.BigDecimal.valueOf(2);
            java.math.BigDecimal wr = new java.math.BigDecimal(x.re).divide(two);
            java.math.BigDecimal wi = new java.math.BigDecimal(x.im).divide(two);
            java.math.BigDecimal yr = decCoef[m];
            java.math.BigDecimal yi = java.math.BigDecimal.ZERO;
            for (int k = m - 1; k >= 0; k--) {
                java.math.BigDecimal nr = yr.multiply(wr, mc).subtract(yi.multiply(wi, mc), mc);
                java.math.BigDecimal ni = yr.multiply(wi, mc).add(yi.multiply(wr, mc), mc);
                yr = nr.add(decCoef[k], mc);
                yi = ni;
            }
            return new Complex(yr.doubleValue(), yi.doubleValue());
        }
    }

    private static BigInteger[] coefficients(int m) {
        BigInteger[] c = new BigInteger[m + 1];
        for (int k = 0; k <= m; k++) {
            BigInteger den = BigInteger.ONE;
            for (int i = 2; i <= k; i++) {
                den = den.multiply(BigInteger.valueOf(i));
            }
            c[k] = fallingFactorial(m + k, 2 * k).divide(den);
        }
        return c;
    }

    /** An integer as an unevaluated sum hi + lo of two doubles. */
    private static double[] dd(BigInteger v) {
        double hi = v.doubleValue();
        double lo = new java.math.BigDecimal(v).subtract(new java.math.BigDecimal(hi)).doubleValue();
        return new double[] {hi, lo};
    }

    private static double[] ddAdd(double[] a, double[] b) {
        double s = a[0] + b[0];
        double bb = s - a[0];
        double err = (a[0] - (s - bb)) + (b[0] - bb);
        err += a[1] + b[1];
        double hi = s + err;
        return new double[] {hi, err - (hi - s)};
    }

    private static double[] ddMul(double[] a, double b) {
        double p = a[0] * b;
        double err = Math.fma(a[0], b, -p) + a[1] * b;
        double hi = p + err;
        return new double[] {hi, err - (hi - p)};
    }

    /** Zeros of the reverse Bessel polynomial, as scipy's _bessel_zeros finds them. */
    static Complex[] besselZeros(int n) {
        if (n == 0) {
            return new Complex[0];
        }
        Complex[] x0 = camposZeros(n);
        KveHalf below = new KveHalf(n - 1);
        KveHalf at = new KveHalf(n);
        KveHalf above = new KveHalf(n + 1);
        UnaryOperator<Complex> f = at::at;
        UnaryOperator<Complex> fp = x -> {
            Complex x2 = x.square();
            Complex twoX2 = new Complex(2.0, 0.0).times(x2);
            return below.at(x).div(twoX2).minus(at.at(x).div(x2)).plus(above.at(x).div(twoX2));
        };
        Complex[] x = Aberth.solve(f, fp, x0, 1e-15, 50);
        for (int i = 0; i < x.length; i++) {
            x[i] = newton(f, fp, x[i], 1e-15, 50);
        }
        Complex[] avg = new Complex[n];
        for (int i = 0; i < n; i++) {
            Complex s = x[i].plus(x[n - 1 - i].conj());
            avg[i] = new Complex(s.re / 2.0, s.im / 2.0);
        }
        if (Numpy.sum(avg).plus(1.0).abs() > 1e-15) {
            throw SignalException.valueError("Generated zeros are inaccurate");
        }
        return avg;
    }

    /** scipy.optimize.newton with a derivative, for a complex root; raises when it does not converge. */
    static Complex newton(UnaryOperator<Complex> f, UnaryOperator<Complex> fp, Complex x0, double tol, int maxiter) {
        Complex p0 = x0;
        for (int itr = 0; itr < maxiter; itr++) {
            Complex fval = f.apply(p0);
            if (fval.re == 0 && fval.im == 0) {
                return p0;
            }
            Complex fder = fp.apply(p0);
            if (fder.re == 0 && fder.im == 0) {
                throw SignalException.valueError("Derivative was zero.");
            }
            Complex p = p0.minus(fval.div(fder));
            if (p.minus(p0).abs() <= tol) {
                return p;
            }
            p0 = p;
        }
        throw SignalException.valueError("Failed to converge after " + maxiter + " iterations");
    }

    /** The frequency where a Bessel filter's magnitude is 1/sqrt(2), by scipy's secant method (tol 1.48e-8). */
    static double normFactor(Complex[] p, double k) {
        double target = 1 / Math.sqrt(2);
        java.util.function.DoubleUnaryOperator cutoff = w -> {
            Complex prod = Complex.ONE;
            Complex jw = new Complex(0.0, w);
            for (Complex pi : p) {
                prod = prod.times(jw.minus(pi));
            }
            return new Complex(k, 0.0).div(prod).abs() - target;
        };
        return secant(cutoff, 1.5, 1.48e-8, 50);
    }

    /** scipy.optimize.newton without a derivative: the secant method, step for step. */
    static double secant(java.util.function.DoubleUnaryOperator func, double x0, double tol, int maxiter) {
        double p0 = x0;
        double eps = 1e-4;
        double p1 = x0 * (1 + eps);
        p1 += (p1 >= 0 ? eps : -eps);
        double q0 = func.applyAsDouble(p0);
        double q1 = func.applyAsDouble(p1);
        if (Math.abs(q1) < Math.abs(q0)) {
            double t = p0;
            p0 = p1;
            p1 = t;
            t = q0;
            q0 = q1;
            q1 = t;
        }
        for (int itr = 0; itr < maxiter; itr++) {
            double p;
            if (q1 == q0) {
                if (p1 != p0) {
                    throw SignalException.valueError("Tolerance of " + (p1 - p0) + " reached.");
                }
                return (p1 + p0) / 2.0;
            } else if (Math.abs(q1) > Math.abs(q0)) {
                p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1);
            } else {
                p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0);
            }
            if (Math.abs(p - p1) <= tol) {
                return p;
            }
            p0 = p1;
            q0 = q1;
            p1 = p;
            q1 = func.applyAsDouble(p1);
        }
        throw SignalException.valueError("Failed to converge after " + maxiter + " iterations");
    }

    // Elliptic helpers

    static double pow10m1(double x) {
        return Math.expm1(Math.log(10) * x);
    }

    /** Solves the degree equation for the elliptic modulus, from nomes (scipy _ellipdeg, 7 terms). */
    static double ellipdeg(int n, double m1) {
        double k1 = Special.ellipk(m1);
        double k1p = Special.ellipkm1(m1);
        double q1 = Math.exp(-Math.PI * k1p / k1);
        double q = Math.pow(q1, 1.0 / n);
        double[] num = new double[8];
        double[] den = new double[8];
        for (int m = 0; m <= 7; m++) {
            num[m] = Math.pow(q, m * (m + 1));
            den[m] = Math.pow(q, (m + 1) * (m + 1));
        }
        double ratio = Numpy.sum(num) / (1 + 2 * Numpy.sum(den));
        return 16 * q * Math.pow(ratio, 4);
    }

    /**
     * Imaginary part of the inverse Jacobi sc function, scipy's _arc_jac_sc1, which evaluates _arc_jac_sn at 1j*w.
     * Along the imaginary axis every Landen step stays imaginary, so this follows scipy's complex arithmetic
     * with the real values it produces.
     */
    static double arcJacSc1(double w, double m) {
        double k = Math.pow(m, 0.5);
        if (k > 1) {
            // scipy's _arc_jac_sn returns nan, and nan's imaginary part is 0.0
            return 0.0;
        } else if (k == 1) {
            return Math.atan(w);
        }
        List<Double> ks = new ArrayList<>();
        ks.add(k);
        int niter = 0;
        while (ks.get(ks.size() - 1) != 0) {
            double kk = ks.get(ks.size() - 1);
            double kp = Math.pow((1 - kk) * (1 + kk), 0.5);
            ks.add((1 - kp) / (1 + kp));
            niter += 1;
            if (niter > 10) {
                throw SignalException.valueError("Landen transformation not converging");
            }
        }
        double prod = 1.0;
        for (int i = 1; i < ks.size(); i++) {
            prod *= 1 + ks.get(i);
        }
        double capK = prod * Math.PI / 2;
        double y = w;
        for (int i = 0; i + 1 < ks.size(); i++) {
            double kn = ks.get(i);
            double knext = ks.get(i + 1);
            double ky = kn * y;
            double c = Math.pow(1.0 - (-ky) * ky, 0.5);
            y = (2 * y) / ((1 + knext) * (1 + c));
        }
        return capK * (2 / Math.PI * Numpy.asinh(y));
    }

    // Frequency transforms

    private static int relativeDegree(Zpk f) {
        int degree = f.p.length - f.z.length;
        if (degree < 0) {
            throw SignalException.valueError("Improper transfer function. Must have at least as many poles as zeros.");
        }
        return degree;
    }

    static Zpk lp2lp(Zpk f, double wo) {
        int degree = relativeDegree(f);
        Complex w = new Complex(wo, 0.0);
        Complex[] z = new Complex[f.z.length];
        for (int i = 0; i < z.length; i++) {
            z[i] = f.zReal ? new Complex(wo * f.z[i].re, 0.0) : w.times(f.z[i]);
        }
        Complex[] p = new Complex[f.p.length];
        for (int i = 0; i < p.length; i++) {
            p[i] = w.times(f.p[i]);
        }
        return new Zpk(z, f.zReal, p, f.k * Math.pow(wo, degree));
    }

    static Zpk lp2hp(Zpk f, double wo) {
        int degree = relativeDegree(f);
        Complex w = new Complex(wo, 0.0);
        Complex[] z = new Complex[f.z.length + degree];
        for (int i = 0; i < f.z.length; i++) {
            z[i] = f.zReal ? new Complex(wo / f.z[i].re, 0.0) : w.div(f.z[i]);
        }
        for (int i = f.z.length; i < z.length; i++) {
            z[i] = Complex.ZERO;
        }
        Complex[] p = new Complex[f.p.length];
        for (int i = 0; i < p.length; i++) {
            p[i] = w.div(f.p[i]);
        }
        double k = f.k * prodNeg(f.z, f.zReal).div(Numpy.prod(negateAll(f.p))).re;
        return new Zpk(z, f.zReal, p, k);
    }

    static Zpk lp2bp(Zpk f, double wo, double bw) {
        int degree = relativeDegree(f);
        double wo2 = wo * wo;
        Complex[] z = new Complex[2 * f.z.length + degree];
        int nz = f.z.length;
        for (int i = 0; i < nz; i++) {
            Complex zl = f.z[i].times(new Complex(bw, 0.0)).divReal(2);
            Complex root = zl.square().minus(new Complex(wo2, 0.0)).sqrt();
            z[i] = zl.plus(root);
            z[nz + i] = zl.minus(root);
        }
        for (int i = 2 * nz; i < z.length; i++) {
            z[i] = Complex.ZERO;
        }
        int np = f.p.length;
        Complex[] p = new Complex[2 * np];
        for (int i = 0; i < np; i++) {
            Complex pl = f.p[i].times(new Complex(bw, 0.0)).divReal(2);
            Complex root = pl.square().minus(new Complex(wo2, 0.0)).sqrt();
            p[i] = pl.plus(root);
            p[np + i] = pl.minus(root);
        }
        return new Zpk(z, false, p, f.k * Math.pow(bw, degree));
    }

    static Zpk lp2bs(Zpk f, double wo, double bw) {
        int degree = relativeDegree(f);
        double wo2 = wo * wo;
        Complex half = new Complex(bw / 2, 0.0);
        int nz = f.z.length;
        Complex[] z = new Complex[2 * nz + 2 * degree];
        for (int i = 0; i < nz; i++) {
            Complex zh = half.div(f.z[i]);
            Complex root = zh.square().minus(new Complex(wo2, 0.0)).sqrt();
            z[i] = zh.plus(root);
            z[nz + i] = zh.minus(root);
        }
        for (int i = 0; i < degree; i++) {
            z[2 * nz + i] = new Complex(0.0, wo);
            z[2 * nz + degree + i] = new Complex(0.0, -wo);
        }
        int np = f.p.length;
        Complex[] p = new Complex[2 * np];
        for (int i = 0; i < np; i++) {
            Complex ph = half.div(f.p[i]);
            Complex root = ph.square().minus(new Complex(wo2, 0.0)).sqrt();
            p[i] = ph.plus(root);
            p[np + i] = ph.minus(root);
        }
        double k = f.k * prodNeg(f.z, f.zReal).div(Numpy.prod(negateAll(f.p))).re;
        return new Zpk(z, false, p, k);
    }

    static Zpk bilinear(Zpk f, double fs) {
        int degree = relativeDegree(f);
        double fs2 = 2.0 * fs;
        Complex two = new Complex(fs2, 0.0);
        Complex[] z = new Complex[f.z.length + degree];
        for (int i = 0; i < f.z.length; i++) {
            if (f.zReal) {
                double v = f.z[i].re;
                z[i] = new Complex((fs2 + v) / (fs2 - v), 0.0);
            } else {
                z[i] = two.plus(f.z[i]).div(two.minus(f.z[i]));
            }
        }
        for (int i = f.z.length; i < z.length; i++) {
            z[i] = new Complex(-1.0, 0.0);
        }
        Complex[] p = new Complex[f.p.length];
        Complex[] den = new Complex[f.p.length];
        for (int i = 0; i < p.length; i++) {
            p[i] = two.plus(f.p[i]).div(two.minus(f.p[i]));
            den[i] = two.minus(f.p[i]);
        }
        Complex num;
        if (f.zReal) {
            double prod = 1.0;
            for (Complex v : f.z) {
                prod *= fs2 - v.re;
            }
            num = new Complex(prod, 0.0);
        } else {
            Complex[] zz = new Complex[f.z.length];
            for (int i = 0; i < zz.length; i++) {
                zz[i] = two.minus(f.z[i]);
            }
            num = Numpy.prod(zz);
        }
        double k = f.k * num.div(Numpy.prod(den)).re;
        return new Zpk(z, f.zReal, p, k);
    }

    private static Complex prodNeg(Complex[] z, boolean real) {
        if (real) {
            double prod = 1.0;
            for (Complex v : z) {
                prod *= -v.re;
            }
            return new Complex(prod, 0.0);
        }
        return Numpy.prod(negateAll(z));
    }

    private static Complex[] negateAll(Complex[] v) {
        Complex[] out = new Complex[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i].negate();
        }
        return out;
    }

    // Conversions

    /**
     * Polynomial with the given roots, as scipy's poly builds it (convolving [1, -root] in turn). Complex roots give
     * real coefficients when their imaginary parts are an exact mirror image, as for conjugate pairs.
     */
    static Poly poly(Complex[] roots, boolean realDtype) {
        if (roots.length == 0) {
            return new Poly(new double[] {1.0}, null);
        }
        if (realDtype) {
            double[] a = {1.0};
            for (Complex r : roots) {
                double neg = -r.re;
                double[] next = new double[a.length + 1];
                next[0] = a[0];
                for (int i = 1; i < a.length; i++) {
                    next[i] = a[i] + a[i - 1] * neg;
                }
                next[a.length] = a[a.length - 1] * neg;
                a = next;
            }
            return new Poly(a, null);
        }
        Complex[] a = {Complex.ONE};
        for (Complex r : roots) {
            Complex neg = r.negate();
            Complex[] next = new Complex[a.length + 1];
            next[0] = a[0];
            for (int i = 1; i < a.length; i++) {
                next[i] = a[i].plus(a[i - 1].times(neg));
            }
            next[a.length] = a[a.length - 1].times(neg);
            a = next;
        }
        double[] im = new double[roots.length];
        double[] imConj = new double[roots.length];
        for (int i = 0; i < roots.length; i++) {
            im[i] = roots[i].im;
            imConj[i] = -roots[i].im;
        }
        Arrays.sort(im);
        Arrays.sort(imConj);
        boolean mirrored = true;
        for (int i = 0; i < im.length; i++) {
            if (im[i] != imConj[i]) {
                mirrored = false;
                break;
            }
        }
        if (mirrored) {
            double[] real = new double[a.length];
            for (int i = 0; i < a.length; i++) {
                real[i] = a[i].re;
            }
            return new Poly(real, null);
        }
        return new Poly(null, a);
    }

    /** Transfer function (b, a) from zeros, poles and gain. */
    public static Poly[] zpk2tf(Complex[] z, boolean zReal, Complex[] p, double k) {
        Poly bz = poly(z, zReal);
        Poly b;
        if (bz.isReal()) {
            double[] v = new double[bz.real.length];
            for (int i = 0; i < v.length; i++) {
                v[i] = k * bz.real[i];
            }
            b = new Poly(v, null);
        } else {
            Complex[] v = new Complex[bz.complex.length];
            Complex kc = new Complex(k, 0.0);
            for (int i = 0; i < v.length; i++) {
                v[i] = kc.times(bz.complex[i]);
            }
            b = new Poly(null, v);
        }
        return new Poly[] {b, poly(p, false)};
    }

    /** Splits values into complex ones with positive imaginary part and real ones, as scipy's _cplxreal. */
    static Complex[][] cplxreal(Complex[] z) {
        if (z.length == 0) {
            return new Complex[][] {new Complex[0], new Complex[0]};
        }
        double tol = 100 * Math.ulp(1.0);
        Complex[] sorted = z.clone();
        Arrays.sort(sorted, Comparator.<Complex>comparingDouble(c -> c.re).thenComparingDouble(c -> Math.abs(c.im)));
        List<Complex> zr = new ArrayList<>();
        List<Complex> rest = new ArrayList<>();
        for (Complex c : sorted) {
            if (Math.abs(c.im) <= tol * c.abs()) {
                zr.add(new Complex(c.re, 0.0));
            } else {
                rest.add(c);
            }
        }
        if (rest.isEmpty()) {
            return new Complex[][] {new Complex[0], zr.toArray(new Complex[0])};
        }
        List<Complex> zp = new ArrayList<>();
        List<Complex> zn = new ArrayList<>();
        for (Complex c : rest) {
            if (c.im > 0) {
                zp.add(c);
            } else if (c.im < 0) {
                zn.add(c);
            }
        }
        if (zp.size() != zn.size()) {
            throw SignalException.valueError("Array contains complex value with no matching conjugate.");
        }
        int n = zp.size();
        int i = 0;
        while (i < n) {
            int j = i;
            while (j + 1 < n && zp.get(j + 1).re - zp.get(j).re <= tol * zp.get(j).abs()) {
                j++;
            }
            if (j > i) {
                Comparator<Complex> byImag = Comparator.comparingDouble(c -> Math.abs(c.im));
                zp.subList(i, j + 1).sort(byImag);
                zn.subList(i, j + 1).sort(byImag);
            }
            i = j + 1;
        }
        Complex[] zc = new Complex[n];
        for (int q = 0; q < n; q++) {
            if (zp.get(q).minus(zn.get(q).conj()).abs() > tol * zn.get(q).abs()) {
                throw SignalException.valueError("Array contains complex value with no matching conjugate.");
            }
            Complex s = zp.get(q).plus(zn.get(q).conj());
            zc[q] = s.divReal(2);
        }
        return new Complex[][] {zc, zr.toArray(new Complex[0])};
    }

    /** Second-order sections of a designed filter, as scipy's iirfilter returns them for output='sos'. */
    public static double[][] sos(Zpk f, boolean analog) {
        return zpk2sos(f.z, f.p, f.k, analog);
    }

    /** Second-order sections from zeros, poles and gain, with scipy's pairing ('nearest' digital, 'minimal' analog). */
    static double[][] zpk2sos(Complex[] zIn, Complex[] pIn, double k, boolean analog) {
        String pairing = analog ? "minimal" : "nearest";
        if (zIn.length == 0 && pIn.length == 0) {
            return analog ? new double[][] {{0., 0., k, 0., 0., 1.}} : new double[][] {{k, 0., 0., 1., 0., 0.}};
        }
        List<Complex> z = new ArrayList<>(Arrays.asList(zIn));
        List<Complex> p = new ArrayList<>(Arrays.asList(pIn));
        int nSections;
        if (!pairing.equals("minimal")) {
            while (p.size() < z.size()) {
                p.add(Complex.ZERO);
            }
            while (z.size() < p.size()) {
                z.add(Complex.ZERO);
            }
            nSections = (Math.max(p.size(), z.size()) + 1) / 2;
            if (p.size() % 2 == 1) {
                p.add(Complex.ZERO);
                z.add(Complex.ZERO);
            }
        } else {
            if (p.size() < z.size()) {
                throw SignalException.valueError("for analog zpk2sos conversion, must have len(p)>=len(z)");
            }
            nSections = (p.size() + 1) / 2;
        }
        z = concat(cplxreal(z.toArray(new Complex[0])));
        p = concat(cplxreal(p.toArray(new Complex[0])));

        double[][] sos = new double[nSections][6];
        for (int si = nSections - 1; si >= 0; si--) {
            int p1Idx = idxWorst(p, analog);
            Complex p1 = p.remove(p1Idx);
            long realLeft = p.stream().filter(Complex::isReal).count();
            if (p1.isReal() && realLeft == 0) {
                if (!pairing.equals("minimal")) {
                    Complex z1 = z.remove(nearest(z, p1, "real"));
                    sos[si] = singleZpkSos(new Complex[] {z1, Complex.ZERO}, new Complex[] {p1, Complex.ZERO});
                } else if (!z.isEmpty()) {
                    Complex z1 = z.remove(nearest(z, p1, "real"));
                    sos[si] = singleZpkSos(new Complex[] {z1}, new Complex[] {p1});
                } else {
                    sos[si] = singleZpkSos(new Complex[0], new Complex[] {p1});
                }
            } else if (p.size() + 1 == z.size() && !p1.isReal() && realLeft == 1
                    && z.stream().filter(Complex::isReal).count() == 1) {
                Complex z1 = z.remove(nearest(z, p1, "complex"));
                sos[si] = singleZpkSos(new Complex[] {z1, z1.conj()}, new Complex[] {p1, p1.conj()});
            } else {
                Complex p2;
                if (p1.isReal()) {
                    List<Integer> realIdx = new ArrayList<>();
                    List<Complex> realP = new ArrayList<>();
                    for (int i = 0; i < p.size(); i++) {
                        if (p.get(i).isReal()) {
                            realIdx.add(i);
                            realP.add(p.get(i));
                        }
                    }
                    int p2Idx = realIdx.get(idxWorst(realP, analog));
                    p2 = p.remove(p2Idx);
                } else {
                    p2 = p1.conj();
                }
                if (!z.isEmpty()) {
                    Complex z1 = z.remove(nearest(z, p1, "any"));
                    if (!z1.isReal()) {
                        sos[si] = singleZpkSos(new Complex[] {z1, z1.conj()}, new Complex[] {p1, p2});
                    } else if (!z.isEmpty()) {
                        Complex z2 = z.remove(nearest(z, p1, "real"));
                        if (!z2.isReal()) {
                            // scipy: assert np.isreal(z2)
                            throw SignalException.valueError("cannot pair the zeros into second-order sections");
                        }
                        sos[si] = singleZpkSos(new Complex[] {z1, z2}, new Complex[] {p1, p2});
                    } else {
                        sos[si] = singleZpkSos(new Complex[] {z1}, new Complex[] {p1, p2});
                    }
                } else {
                    sos[si] = singleZpkSos(new Complex[0], new Complex[] {p1, p2});
                }
            }
        }
        if (!p.isEmpty() || !z.isEmpty()) {
            // scipy: assert len(p) == len(z) == 0. NaN poles or zeros (a degenerate design) are dropped by the
            // conjugate split, which leaves the counts unbalanced.
            throw SignalException.valueError("cannot pair the zeros and poles into second-order sections");
        }
        for (int i = 0; i < 3; i++) {
            sos[0][i] *= k;
        }
        return sos;
    }

    private static List<Complex> concat(Complex[][] parts) {
        List<Complex> out = new ArrayList<>(Arrays.asList(parts[0]));
        out.addAll(Arrays.asList(parts[1]));
        return out;
    }

    /**
     * numpy.argmin of |1 - |p|| (digital) or |Re p| (analog): the first minimum, or the first NaN if any. An empty
     * list raises, as numpy's argmin does; it happens when a design's poles are NaN (an elliptic filter with rs
     * below rp), because the conjugate split drops NaN values.
     */
    private static int idxWorst(List<Complex> p, boolean analog) {
        if (p.isEmpty()) {
            throw SignalException.valueError("attempt to get argmin of an empty sequence");
        }
        int best = 0;
        double bestVal = Double.POSITIVE_INFINITY;
        for (int i = 0; i < p.size(); i++) {
            double v = analog ? Math.abs(p.get(i).re) : Math.abs(1 - p.get(i).abs());
            if (Double.isNaN(v)) {
                return i;
            }
            if (v < bestVal) {
                bestVal = v;
                best = i;
            }
        }
        return best;
    }

    private static int nearest(List<Complex> fro, Complex to, String which) {
        if (fro.isEmpty()) {
            // scipy: indexing an empty argsort result raises IndexError
            throw SignalException.valueError("cannot pair the zeros into second-order sections: no zero left");
        }
        Integer[] order = new Integer[fro.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, Comparator.comparingDouble(i -> fro.get(i).minus(to).abs()));
        if (which.equals("any")) {
            return order[0];
        }
        for (int i : order) {
            boolean real = fro.get(i).isReal();
            if (which.equals("real") == real) {
                return i;
            }
        }
        // scipy fails here too (an IndexError in _nearest_real_complex_idx): the zeros cannot be paired.
        throw SignalException.valueError("cannot pair the zeros into second-order sections: no " + which
                + " zero left to pair with");
    }

    private static double[] singleZpkSos(Complex[] z, Complex[] p) {
        Poly[] ba = zpk2tf(z, false, p, 1);
        double[] sos = new double[6];
        double[] b = realPart(ba[0]);
        double[] a = realPart(ba[1]);
        System.arraycopy(b, 0, sos, 3 - b.length, b.length);
        System.arraycopy(a, 0, sos, 6 - a.length, a.length);
        return sos;
    }

    private static double[] realPart(Poly poly) {
        if (poly.isReal()) {
            return poly.real;
        }
        double[] out = new double[poly.complex.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = poly.complex[i].re;
        }
        return out;
    }

    // Design

    /**
     * scipy's iirfilter after argument parsing: Wn already divided by fs/2 when fs was given, and every string
     * already resolved. {@code ftype} is butter, cheby1, cheby2, ellip or bessel; {@code band} is lowpass,
     * highpass, bandpass or bandstop.
     */
    public static Zpk design(String ftype, int n, double[] wn, Double rp, Double rs, String band, boolean analog,
                             String norm, Double fs) {
        Zpk proto;
        switch (ftype) {
            case "butter":
                proto = buttap(n);
                break;
            case "cheby1":
                proto = cheb1ap(n, rp);
                break;
            case "cheby2":
                proto = cheb2ap(n, rs);
                break;
            case "ellip":
                proto = ellipap(n, rp, rs);
                break;
            case "bessel":
                proto = besselap(n, norm);
                break;
            default:
                throw new IllegalArgumentException(ftype);
        }
        double[] warped = new double[wn.length];
        double fsDesign = 0;
        if (!analog) {
            for (double w : wn) {
                if (w <= 0 || w >= 1) {
                    if (fs != null) {
                        throw SignalException.valueError("Digital filter critical frequencies must be 0 < Wn < fs/2 "
                                + "(fs=" + fs + " -> fs/2=" + (fs / 2) + ")");
                    }
                    throw SignalException.valueError("Digital filter critical frequencies must be 0 < Wn < 1");
                }
            }
            fsDesign = 2.0;
            for (int i = 0; i < wn.length; i++) {
                warped[i] = 2 * fsDesign * Math.tan(Math.PI * wn[i] / fsDesign);
            }
        } else {
            System.arraycopy(wn, 0, warped, 0, wn.length);
        }
        Zpk f;
        if (band.equals("lowpass") || band.equals("highpass")) {
            if (wn.length != 1) {
                throw SignalException.valueError("Must specify a single critical frequency Wn for lowpass or "
                        + "highpass filter");
            }
            f = band.equals("lowpass") ? lp2lp(proto, warped[0]) : lp2hp(proto, warped[0]);
        } else {
            if (wn.length < 2) {
                throw SignalException.valueError("Wn must specify start and stop frequencies for bandpass or "
                        + "bandstop filter");
            }
            double bw = warped[1] - warped[0];
            double wo = Math.sqrt(warped[0] * warped[1]);
            f = band.equals("bandpass") ? lp2bp(proto, wo, bw) : lp2bs(proto, wo, bw);
        }
        if (!analog) {
            f = bilinear(f, fsDesign);
        }
        return f;
    }
}
