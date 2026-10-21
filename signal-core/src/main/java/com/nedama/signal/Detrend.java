package com.nedama.signal;

import java.util.Arrays;

/**
 * Removes a constant or linear trend, as {@code scipy.signal.detrend} does for 1-D data.
 *
 * <p>scipy fits each segment with a general least-squares solver. Here the straight-line fit is done in closed
 * form on a centred sample index, with compensated sums, which gives the same residuals to within rounding.
 */
public final class Detrend {

    private Detrend() {
    }

    /** Subtracts the mean. An empty input gives an empty result, as in scipy. */
    public static double[] constant(double[] data) {
        double[] out = new double[data.length];
        if (data.length == 0) {
            return out;
        }
        double mean = Sums.mean(data, 0, data.length);
        for (int i = 0; i < data.length; i++) {
            out[i] = data[i] - mean;
        }
        return out;
    }

    /**
     * Subtracts a least-squares straight line from each segment between consecutive breakpoints.
     * Breakpoints are sample indices from 0 to {@code data.length}, in any order, repeats allowed.
     */
    public static double[] linear(double[] data, long[] breakpoints) {
        int n = data.length;
        for (long b : breakpoints) {
            if (b > n) {
                throw SignalException.valueError("Breakpoints must be less than length of data along given axis.");
            }
            if (b < 0) {
                throw SignalException.valueError("Breakpoints must not be negative.");
            }
        }
        if (n == 0) {
            throw SignalException.valueError("A linear detrend needs at least one sample.");
        }
        int[] cuts = cutPoints(breakpoints, n);
        double[] out = new double[n];
        for (int k = 0; k + 1 < cuts.length; k++) {
            fitSegment(data, cuts[k], cuts[k + 1], out);
        }
        return out;
    }

    /** 0, the breakpoints and n, sorted with repeats removed. */
    private static int[] cutPoints(long[] breakpoints, int n) {
        int[] all = new int[breakpoints.length + 2];
        all[0] = 0;
        all[1] = n;
        for (int i = 0; i < breakpoints.length; i++) {
            all[i + 2] = (int) breakpoints[i];
        }
        Arrays.sort(all);
        int m = 1;
        for (int i = 1; i < all.length; i++) {
            if (all[i] != all[m - 1]) {
                all[m++] = all[i];
            }
        }
        return Arrays.copyOf(all, m);
    }

    /** Residuals of the straight-line fit to {@code y[from..to)}, written into {@code out}. */
    private static void fitSegment(double[] y, int from, int to, double[] out) {
        int m = to - from;
        if (m == 1) {
            // A line through one point fits it exactly.
            out[from] = 0.0;
            return;
        }
        double yMean = Sums.mean(y, from, to);
        // Centred index u = k - (m - 1) / 2 sums to zero, so the slope needs no intercept term.
        double uOffset = (m - 1) / 2.0;
        double uu = m * ((double) m * m - 1.0) / 12.0;
        double s = 0.0;
        double c = 0.0;
        for (int k = 0; k < m; k++) {
            double v = (k - uOffset) * (y[from + k] - yMean);
            double t = s + v;
            if (Math.abs(s) >= Math.abs(v)) {
                c += (s - t) + v;
            } else {
                c += (v - t) + s;
            }
            s = t;
        }
        double slope = (s + c) / uu;
        for (int k = 0; k < m; k++) {
            out[from + k] = (y[from + k] - yMean) - slope * (k - uOffset);
        }
    }
}
