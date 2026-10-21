package com.nedama.signal;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.math.RoundingMode;

/**
 * An exact rational number, for the small least-squares problems of Savitzky-Golay filtering. Their inputs (sample
 * positions and doubles) are exact rationals, so solving exactly and rounding once gives correctly rounded results,
 * where a floating-point solver loses accuracy as the Vandermonde matrix grows ill-conditioned.
 */
final class Rational {

    static final Rational ZERO = new Rational(BigInteger.ZERO, BigInteger.ONE);
    static final Rational ONE = new Rational(BigInteger.ONE, BigInteger.ONE);

    final BigInteger num;
    final BigInteger den;

    private Rational(BigInteger num, BigInteger den) {
        this.num = num;
        this.den = den;
    }

    static Rational of(BigInteger num, BigInteger den) {
        if (den.signum() == 0) {
            throw new ArithmeticException("division by zero");
        }
        if (den.signum() < 0) {
            num = num.negate();
            den = den.negate();
        }
        BigInteger g = num.gcd(den);
        if (!g.equals(BigInteger.ONE) && g.signum() != 0) {
            num = num.divide(g);
            den = den.divide(g);
        }
        return new Rational(num, den);
    }

    static Rational of(long v) {
        return new Rational(BigInteger.valueOf(v), BigInteger.ONE);
    }

    /** The exact value of a finite double. */
    static Rational of(double v) {
        BigDecimal d = new BigDecimal(v);
        BigInteger unscaled = d.unscaledValue();
        int scale = d.scale();
        if (scale <= 0) {
            return new Rational(unscaled.multiply(BigInteger.TEN.pow(-scale)), BigInteger.ONE);
        }
        return of(unscaled, BigInteger.TEN.pow(scale));
    }

    Rational add(Rational o) {
        if (den.equals(o.den)) {
            return of(num.add(o.num), den);
        }
        return of(num.multiply(o.den).add(o.num.multiply(den)), den.multiply(o.den));
    }

    Rational subtract(Rational o) {
        return add(o.negate());
    }

    Rational multiply(Rational o) {
        return of(num.multiply(o.num), den.multiply(o.den));
    }

    Rational divide(Rational o) {
        return of(num.multiply(o.den), den.multiply(o.num));
    }

    Rational negate() {
        return new Rational(num.negate(), den);
    }

    boolean isZero() {
        return num.signum() == 0;
    }

    /** Rounded to the nearest double (60 significant digits first, then to double). */
    double toDouble() {
        if (num.signum() == 0) {
            return 0.0;
        }
        return new BigDecimal(num).divide(new BigDecimal(den), new MathContext(60, RoundingMode.HALF_EVEN))
                .doubleValue();
    }

    /** Solves m x = b exactly by Gaussian elimination (m square and non-singular). */
    static Rational[] solve(Rational[][] m, Rational[] b) {
        int n = b.length;
        Rational[][] a = new Rational[n][];
        for (int i = 0; i < n; i++) {
            a[i] = new Rational[n + 1];
            System.arraycopy(m[i], 0, a[i], 0, n);
            a[i][n] = b[i];
        }
        for (int k = 0; k < n; k++) {
            int p = k;
            while (p < n && a[p][k].isZero()) {
                p++;
            }
            if (p == n) {
                throw SignalException.valueError("singular least-squares system");
            }
            Rational[] t = a[k];
            a[k] = a[p];
            a[p] = t;
            for (int i = k + 1; i < n; i++) {
                if (a[i][k].isZero()) {
                    continue;
                }
                Rational f = a[i][k].divide(a[k][k]);
                for (int j = k; j <= n; j++) {
                    a[i][j] = a[i][j].subtract(f.multiply(a[k][j]));
                }
            }
        }
        Rational[] x = new Rational[n];
        for (int i = n - 1; i >= 0; i--) {
            Rational s = a[i][n];
            for (int j = i + 1; j < n; j++) {
                s = s.subtract(a[i][j].multiply(x[j]));
            }
            x[i] = s.divide(a[i][i]);
        }
        return x;
    }
}
