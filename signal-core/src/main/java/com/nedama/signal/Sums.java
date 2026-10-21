package com.nedama.signal;

/**
 * Compensated (Neumaier) summation. A plain loop loses about n * 1e-16 relative accuracy over n terms,
 * which matters for long historian series; this keeps the error near one rounding regardless of length.
 */
final class Sums {

    private Sums() {
    }

    /** Sum of {@code x[from..to)}. */
    static double sum(double[] x, int from, int to) {
        double s = 0.0;
        double c = 0.0;
        for (int i = from; i < to; i++) {
            double v = x[i];
            double t = s + v;
            if (Math.abs(s) >= Math.abs(v)) {
                c += (s - t) + v;
            } else {
                c += (v - t) + s;
            }
            s = t;
        }
        return s + c;
    }

    /** Mean of {@code x[from..to)}; NaN for an empty range. */
    static double mean(double[] x, int from, int to) {
        return sum(x, from, to) / (to - from);
    }
}
