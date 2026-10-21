package com.nedama.signal;

import java.util.function.DoubleUnaryOperator;

/**
 * Bounded scalar minimisation by Brent's method, ported from scipy.optimize._optimize._minimize_scalar_bounded
 * (scipy 1.18.1), which scipy.optimize.fminbound calls. The order-selection functions (buttord and the others)
 * use it, so this follows scipy step for step: same golden-section and parabolic steps, same stopping test.
 */
final class Fminbound {

    private Fminbound() {
    }

    /** The outcome, as fminbound(..., full_output=True) returns it. */
    static final class Result {
        final double x;
        final double fun;
        /** 0: converged; 1: maximum number of function calls reached; 2: NaN encountered. */
        final int status;
        final int nfev;

        Result(double x, double fun, int status, int nfev) {
            this.x = x;
            this.fun = fun;
            this.status = status;
            this.nfev = nfev;
        }
    }

    static Result minimize(DoubleUnaryOperator func, double x1, double x2, double xatol, int maxfun) {
        if (!Double.isFinite(x1) || !Double.isFinite(x2)) {
            throw SignalException.valueError("Optimization bounds must be finite scalars.");
        }
        if (x1 > x2) {
            throw SignalException.valueError("The lower bound exceeds the upper bound.");
        }
        int flag = 0;
        double sqrtEps = Math.sqrt(2.2e-16);
        double goldenMean = 0.5 * (3.0 - Math.sqrt(5.0));
        double a = x1;
        double b = x2;
        double fulc = a + goldenMean * (b - a);
        double nfc = fulc;
        double xf = fulc;
        double rat = 0.0;
        double e = 0.0;
        double x = xf;
        double fx = func.applyAsDouble(x);
        int num = 1;
        double fu = Double.POSITIVE_INFINITY;

        double ffulc = fx;
        double fnfc = fx;
        double xm = 0.5 * (a + b);
        double tol1 = sqrtEps * Math.abs(xf) + xatol / 3.0;
        double tol2 = 2.0 * tol1;

        while (Math.abs(xf - xm) > (tol2 - 0.5 * (b - a))) {
            boolean golden = true;
            // Check for a parabolic fit.
            if (Math.abs(e) > tol1) {
                golden = false;
                double r = (xf - nfc) * (fx - ffulc);
                double q = (xf - fulc) * (fx - fnfc);
                double p = (xf - fulc) * q - (xf - nfc) * r;
                q = 2.0 * (q - r);
                if (q > 0.0) {
                    p = -p;
                }
                q = Math.abs(q);
                r = e;
                e = rat;

                // Check the parabola is acceptable.
                if ((Math.abs(p) < Math.abs(0.5 * q * r)) && (p > q * (a - xf)) && (p < q * (b - xf))) {
                    rat = (p + 0.0) / q;
                    x = xf + rat;
                    if (((x - a) < tol2) || ((b - x) < tol2)) {
                        double si = Math.signum(xm - xf) + ((xm - xf) == 0 ? 1 : 0);
                        rat = tol1 * si;
                    }
                } else {
                    golden = true;
                }
            }

            if (golden) {
                if (xf >= xm) {
                    e = a - xf;
                } else {
                    e = b - xf;
                }
                rat = goldenMean * e;
            }

            double si = Math.signum(rat) + (rat == 0 ? 1 : 0);
            x = xf + si * Math.max(Math.abs(rat), tol1);
            fu = func.applyAsDouble(x);
            num += 1;

            if (fu <= fx) {
                if (x >= xf) {
                    a = xf;
                } else {
                    b = xf;
                }
                fulc = nfc;
                ffulc = fnfc;
                nfc = xf;
                fnfc = fx;
                xf = x;
                fx = fu;
            } else {
                if (x < xf) {
                    a = x;
                } else {
                    b = x;
                }
                if ((fu <= fnfc) || (nfc == xf)) {
                    fulc = nfc;
                    ffulc = fnfc;
                    nfc = x;
                    fnfc = fu;
                } else if ((fu <= ffulc) || (fulc == xf) || (fulc == nfc)) {
                    fulc = x;
                    ffulc = fu;
                }
            }

            xm = 0.5 * (a + b);
            tol1 = sqrtEps * Math.abs(xf) + xatol / 3.0;
            tol2 = 2.0 * tol1;

            if (num >= maxfun) {
                flag = 1;
                break;
            }
        }

        if (Double.isNaN(xf) || Double.isNaN(fx) || Double.isNaN(fu)) {
            flag = 2;
        }
        return new Result(xf, fx, flag, num);
    }
}
