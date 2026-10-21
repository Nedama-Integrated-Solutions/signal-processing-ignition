package com.nedama.signal;

import java.util.function.UnaryOperator;

/**
 * Aberth-Ehrlich simultaneous root finding, ported from scipy.signal._filter_design._aberth (scipy 1.18.1).
 * Used to find the poles of Bessel filters. Every root moves by a Newton step corrected for the "repulsion" of
 * the other roots, all at once, until every Newton step is no larger than {@code tol}.
 */
final class Aberth {

    private Aberth() {
    }

    /**
     * Refines the starting points {@code x0} to zeros of {@code f}, whose derivative is {@code fp}.
     *
     * @throws SignalException if a value stops being finite, or the steps do not shrink below {@code tol}
     *                         within {@code maxiter} rounds (scipy raises RuntimeError and Exception)
     */
    static Complex[] solve(UnaryOperator<Complex> f, UnaryOperator<Complex> fp, Complex[] x0, double tol,
                           int maxiter) {
        int n = x0.length;
        Complex[] x = x0.clone();
        Complex[] alpha = new Complex[n];
        Complex[] beta = new Complex[n];
        for (int iteration = 0; iteration < maxiter; iteration++) {
            for (int k = 0; k < n; k++) {
                alpha[k] = f.apply(x[k]).negate().div(fp.apply(x[k]));
            }
            for (int k = 0; k < n; k++) {
                Complex[] later = new Complex[n - k - 1];
                for (int j = k + 1; j < n; j++) {
                    later[j - k - 1] = Complex.ONE.div(x[k].minus(x[j]));
                }
                Complex[] earlier = new Complex[k];
                for (int j = 0; j < k; j++) {
                    earlier[j] = Complex.ONE.div(x[k].minus(x[j]));
                }
                // numpy sums each array pairwise, then adds the two sums.
                beta[k] = Numpy.sum(later).plus(Numpy.sum(earlier));
            }
            boolean finite = true;
            for (int k = 0; k < n; k++) {
                x[k] = x[k].plus(alpha[k].div(alpha[k].times(beta[k]).plus(1.0)));
                finite &= x[k].isFinite();
            }
            if (!finite) {
                throw SignalException.valueError("Root-finding calculation failed");
            }
            boolean converged = true;
            for (int k = 0; k < n; k++) {
                if (!(alpha[k].abs() <= tol)) {
                    converged = false;
                    break;
                }
            }
            if (converged) {
                return x;
            }
        }
        throw SignalException.valueError("Zeros failed to converge");
    }
}
