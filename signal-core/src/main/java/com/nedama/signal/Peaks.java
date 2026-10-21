package com.nedama.signal;

import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Peak finding, ported from scipy.signal (scipy 1.18.1): find_peaks with its height, threshold, distance,
 * prominence, width and plateau_size conditions, and peak_prominences and peak_widths. The loops are scipy's
 * Cython kernels; results use only comparisons, differences and one interpolation, so they match scipy exactly.
 * The prominence and width walks are replaced by lookups in a range-minimum tree that give the same answers in
 * O(log n) per peak, so trending signals do not cost time proportional to the square of their length.
 *
 * <p>One deliberate difference: when peaks of equal height compete under {@code distance}, scipy ranks them with
 * numpy's default (unstable) sort, whose order for ties depends on the computer's vector instructions. Here ties
 * are ranked by position, the later peak first, as numpy's stable sort would.
 */
public final class Peaks {

    private Peaks() {
    }

    /** One border of a condition: none, a number, or a list compared element by element (numpy broadcasting). */
    public static final class Border {
        final Double scalar;
        final double[] values;

        private Border(Double scalar, double[] values) {
            this.scalar = scalar;
            this.values = values;
        }

        public static Border of(double v) {
            return new Border(v, null);
        }

        public static Border of(double[] v) {
            return v.length == 1 ? new Border(v[0], null) : new Border(null, v);
        }

        /** Element i of this border against n properties, as numpy broadcasts a list against an array. */
        double at(int i, int n) {
            if (scalar != null) {
                return scalar;
            }
            if (values.length != n) {
                throw SignalException.valueError("operands could not be broadcast together with shapes ("
                        + n + ",) (" + values.length + ",)");
            }
            return values[i];
        }
    }

    /** A condition (min, max); either border may be null. */
    public static final class Condition {
        final Border min;
        final Border max;

        public Condition(Border min, Border max) {
            this.min = min;
            this.max = max;
        }

        boolean[] keep(double[] property) {
            int n = property.length;
            boolean[] keep = new boolean[n];
            Arrays.fill(keep, true);
            if (min != null) {
                if (min.values != null && min.values.length != n) {
                    min.at(0, n);
                }
                for (int i = 0; i < n; i++) {
                    keep[i] &= min.at(i, n) <= property[i];
                }
            }
            if (max != null) {
                if (max.values != null && max.values.length != n) {
                    max.at(0, n);
                }
                for (int i = 0; i < n; i++) {
                    keep[i] &= property[i] <= max.at(i, n);
                }
            }
            return keep;
        }
    }

    /**
     * A range-minimum segment tree over the signal, so the walks in scipy's prominence and width loops become
     * O(log n) lookups with the same results: the walks are what make scipy's cost grow with the square of the
     * length on trending signals (a rising, noisy 100,000-point signal took 1.4 s in scipy, 1,000,000 points
     * about 140 s).
     */
    static final class RangeMin {
        private final double[] t;
        private final int size;
        private final int n;

        RangeMin(double[] x) {
            n = x.length;
            int s = 1;
            while (s < Math.max(1, n)) {
                s <<= 1;
            }
            size = s;
            t = new double[2 * s];
            Arrays.fill(t, Double.POSITIVE_INFINITY);
            System.arraycopy(x, 0, t, s, n);
            for (int i = s - 1; i >= 1; i--) {
                t[i] = Math.min(t[2 * i], t[2 * i + 1]);
            }
        }

        /** Minimum of x[l..r] (inclusive). */
        double min(int l, int r) {
            double m = Double.POSITIVE_INFINITY;
            for (int a = l + size, b = r + size + 1; a < b; a >>= 1, b >>= 1) {
                if ((a & 1) == 1) {
                    m = Math.min(m, t[a++]);
                }
                if ((b & 1) == 1) {
                    m = Math.min(m, t[--b]);
                }
            }
            return m;
        }

        /** Smallest i in [l, r] with x[i] <= h, or -1. */
        int leftmostAtMost(int l, int r, double h) {
            return l > r ? -1 : first(1, 0, size - 1, l, r, h);
        }

        /** Largest i in [l, r] with x[i] <= h, or -1. */
        int rightmostAtMost(int l, int r, double h) {
            return l > r ? -1 : last(1, 0, size - 1, l, r, h);
        }

        private int first(int node, int nl, int nr, int l, int r, double h) {
            if (nr < l || nl > r || !(t[node] <= h)) {
                return -1;
            }
            if (nl == nr) {
                return nl;
            }
            int mid = (nl + nr) >>> 1;
            int f = first(2 * node, nl, mid, l, r, h);
            return f >= 0 ? f : first(2 * node + 1, mid + 1, nr, l, r, h);
        }

        private int last(int node, int nl, int nr, int l, int r, double h) {
            if (nr < l || nl > r || !(t[node] <= h)) {
                return -1;
            }
            if (nl == nr) {
                return nl;
            }
            int mid = (nl + nr) >>> 1;
            int f = last(2 * node + 1, mid + 1, nr, l, r, h);
            return f >= 0 ? f : last(2 * node, nl, mid, l, r, h);
        }
    }

    /** For each index, the nearest index on the left (prev) and right (next) with a strictly greater value. */
    static int[][] nearestGreater(double[] x) {
        int n = x.length;
        int[] prev = new int[n];
        int[] next = new int[n];
        int[] stack = new int[n];
        int top = 0;
        for (int i = 0; i < n; i++) {
            while (top > 0 && x[stack[top - 1]] <= x[i]) {
                top--;
            }
            prev[i] = top > 0 ? stack[top - 1] : -1;
            stack[top++] = i;
        }
        top = 0;
        for (int i = n - 1; i >= 0; i--) {
            while (top > 0 && x[stack[top - 1]] <= x[i]) {
                top--;
            }
            next[i] = top > 0 ? stack[top - 1] : n;
            stack[top++] = i;
        }
        return new int[][] {prev, next};
    }

    /** _local_maxima_1d: midpoints, left edges and right edges of every local maximum (plateaus included). */
    static long[][] localMaxima(double[] x) {
        int cap = x.length / 2;
        long[] mid = new long[cap];
        long[] left = new long[cap];
        long[] right = new long[cap];
        int m = 0;
        int i = 1;
        int iMax = x.length - 1;
        while (i < iMax) {
            if (x[i - 1] < x[i]) {
                int ahead = i + 1;
                while (ahead < iMax && x[ahead] == x[i]) {
                    ahead++;
                }
                if (x[ahead] < x[i]) {
                    left[m] = i;
                    right[m] = ahead - 1;
                    mid[m] = (left[m] + right[m]) / 2;
                    m++;
                    i = ahead;
                }
            }
            i++;
        }
        return new long[][] {Arrays.copyOf(mid, m), Arrays.copyOf(left, m), Arrays.copyOf(right, m)};
    }

    /**
     * _select_by_peak_distance, with ties in priority ranked by position (a stable sort): from the highest
     * priority down, each kept peak removes its neighbours closer than ceil(distance).
     */
    static boolean[] selectByDistance(long[] peaks, double[] priority, double distance) {
        int n = peaks.length;
        double c = Math.ceil(distance);
        // scipy casts with <np.intp_t>; out-of-range values become INT64_MIN on x86
        long d = c >= 9.223372036854775807E18 || c < -9.223372036854775808E18 ? Long.MIN_VALUE : (long) c;
        boolean[] keep = new boolean[n];
        Arrays.fill(keep, true);
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, Comparator.comparingDouble((Integer k) -> priority[k] == 0.0 ? 0.0 : priority[k]));
        for (int i = n - 1; i >= 0; i--) {
            int j = order[i];
            if (!keep[j]) {
                continue;
            }
            int k = j - 1;
            while (0 <= k && peaks[j] - peaks[k] < d) {
                keep[k] = false;
                k--;
            }
            k = j + 1;
            while (k < n && peaks[k] - peaks[j] < d) {
                keep[k] = false;
                k++;
            }
        }
        return keep;
    }

    /** Prominences with their left and right bases. */
    public static final class Prominences {
        public final double[] prominences;
        public final long[] leftBases;
        public final long[] rightBases;

        Prominences(double[] prominences, long[] leftBases, long[] rightBases) {
            this.prominences = prominences;
            this.leftBases = leftBases;
            this.rightBases = rightBases;
        }
    }

    /**
     * _peak_prominences; {@code wlen} below 2 means the whole signal. scipy walks left (then right) from the peak
     * while samples are no higher than the peak, keeping the first strictly lower minimum it meets. The walk stops
     * at the nearest strictly higher sample (or the window's edge), and the minimum it keeps is the occurrence of
     * the range's minimum closest to the peak, so both are looked up instead of walked.
     */
    public static Prominences prominences(double[] x, long[] peaks, long wlen) {
        int n = peaks.length;
        for (long peak : peaks) {
            if (!(0 <= peak && peak <= x.length - 1)) {
                throw SignalException.valueError("peak " + peak + " is not a valid index for `x`");
            }
        }
        double[] prom = new double[n];
        long[] lb = new long[n];
        long[] rb = new long[n];
        if (n == 0) {
            return new Prominences(prom, lb, rb);
        }
        RangeMin tree = new RangeMin(x);
        int[][] greater = nearestGreater(x);
        for (int p = 0; p < n; p++) {
            int peak = (int) peaks[p];
            long iMin = 0;
            long iMax = x.length - 1;
            if (2 <= wlen) {
                iMin = Math.max(peak - wlen / 2, iMin);
                iMax = Math.min(peak + wlen / 2, iMax);
            }
            int lo = (int) Math.max(iMin, greater[0][peak] + 1L);
            double leftMin = tree.min(lo, peak);
            lb[p] = tree.rightmostAtMost(lo, peak, leftMin);
            int hi = (int) Math.min(iMax, greater[1][peak] - 1L);
            double rightMin = tree.min(peak, hi);
            rb[p] = tree.leftmostAtMost(peak, hi, rightMin);
            prom[p] = x[peak] - Math.max(leftMin, rightMin);
        }
        return new Prominences(prom, lb, rb);
    }

    /** Widths, the heights they were measured at, and the interpolated left and right crossing positions. */
    public static final class Widths {
        public final double[] widths;
        public final double[] widthHeights;
        public final double[] leftIps;
        public final double[] rightIps;

        Widths(double[] widths, double[] widthHeights, double[] leftIps, double[] rightIps) {
            this.widths = widths;
            this.widthHeights = widthHeights;
            this.leftIps = leftIps;
            this.rightIps = rightIps;
        }
    }

    /** _peak_widths. */
    public static Widths widths(double[] x, long[] peaks, double relHeight, double[] prominences, long[] leftBases,
                                long[] rightBases) {
        if (relHeight < 0) {
            throw SignalException.valueError("`rel_height` must be greater or equal to 0.0");
        }
        int n = peaks.length;
        if (!(n == prominences.length && n == leftBases.length && n == rightBases.length)) {
            throw SignalException.valueError("arrays in `prominence_data` must have the same shape as `peaks`");
        }
        double[] widths = new double[n];
        double[] heights = new double[n];
        double[] leftIps = new double[n];
        double[] rightIps = new double[n];
        for (int p = 0; p < n; p++) {
            long iMin = leftBases[p];
            long iMax = rightBases[p];
            long peak = peaks[p];
            if (!(0 <= iMin && iMin <= peak && peak <= iMax && iMax < x.length)) {
                throw SignalException.valueError("prominence data is invalid for peak " + peak);
            }
        }
        RangeMin tree = n == 0 ? null : new RangeMin(x);
        for (int p = 0; p < n; p++) {
            int iMin = (int) leftBases[p];
            int iMax = (int) rightBases[p];
            int peak = (int) peaks[p];
            double height = x[peak] - prominences[p] * relHeight;
            heights[p] = height;
            // scipy walks left while iMin < i and height < x[i]: it stops at the nearest sample at or below the
            // height in (iMin, peak], or at iMin.
            int found = tree.rightmostAtMost(iMin + 1, peak, height);
            int i = found >= 0 ? found : iMin;
            double leftIp = i;
            if (x[i] < height) {
                if (x[i + 1] != x[i]) {
                    leftIp += (height - x[i]) / (x[i + 1] - x[i]);
                }
            }
            // ... and right while i < iMax and height < x[i]: the nearest sample at or below it in [peak, iMax),
            // or iMax.
            found = tree.leftmostAtMost(peak, iMax - 1, height);
            i = found >= 0 ? found : iMax;
            double rightIp = i;
            if (x[i] < height) {
                if (x[i - 1] != x[i]) {
                    rightIp -= (height - x[i]) / (x[i - 1] - x[i]);
                }
            }
            widths[p] = rightIp - leftIp;
            leftIps[p] = leftIp;
            rightIps[p] = rightIp;
        }
        return new Widths(widths, heights, leftIps, rightIps);
    }

    // find_peaks

    /** The peaks found and their properties, in scipy's order and with scipy's names. */
    public static final class Found {
        public final long[] peaks;
        public final Map<String, Object> properties;

        Found(long[] peaks, Map<String, Object> properties) {
            this.peaks = peaks;
            this.properties = properties;
        }
    }

    private static long[] select(long[] a, boolean[] keep) {
        int m = 0;
        for (boolean k : keep) {
            m += k ? 1 : 0;
        }
        long[] out = new long[m];
        for (int i = 0, j = 0; i < a.length; i++) {
            if (keep[i]) {
                out[j++] = a[i];
            }
        }
        return out;
    }

    private static double[] select(double[] a, boolean[] keep) {
        int m = 0;
        for (boolean k : keep) {
            m += k ? 1 : 0;
        }
        double[] out = new double[m];
        for (int i = 0, j = 0; i < a.length; i++) {
            if (keep[i]) {
                out[j++] = a[i];
            }
        }
        return out;
    }

    private static void selectAll(Map<String, Object> props, boolean[] keep) {
        for (Map.Entry<String, Object> e : props.entrySet()) {
            Object v = e.getValue();
            e.setValue(v instanceof long[] ? select((long[]) v, keep) : select((double[]) v, keep));
        }
    }

    private static double[] asDouble(long[] v) {
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i];
        }
        return out;
    }

    /**
     * find_peaks. Conditions may be null (not applied); {@code distance} null means no distance condition;
     * {@code wlen} is already converted (-1 for None) and only used with prominence or width.
     */
    public static Found findPeaks(double[] x, Condition height, Condition threshold, Double distance,
                                  Condition prominence, Condition width, long wlen, double relHeight,
                                  Condition plateauSize) {
        if (distance != null && distance < 1) {
            throw SignalException.valueError("`distance` must be greater or equal to 1");
        }
        long[][] lm = localMaxima(x);
        long[] peaks = lm[0];
        Map<String, Object> props = new LinkedHashMap<>();
        if (plateauSize != null) {
            long[] sizes = new long[peaks.length];
            for (int i = 0; i < sizes.length; i++) {
                sizes[i] = lm[2][i] - lm[1][i] + 1;
            }
            boolean[] keep = plateauSize.keep(asDouble(sizes));
            peaks = select(peaks, keep);
            props.put("plateau_sizes", sizes);
            props.put("left_edges", lm[1]);
            props.put("right_edges", lm[2]);
            selectAll(props, keep);
        }
        if (height != null) {
            double[] heights = new double[peaks.length];
            for (int i = 0; i < heights.length; i++) {
                heights[i] = x[(int) peaks[i]];
            }
            boolean[] keep = height.keep(heights);
            peaks = select(peaks, keep);
            props.put("peak_heights", heights);
            selectAll(props, keep);
        }
        if (threshold != null) {
            double[] left = new double[peaks.length];
            double[] right = new double[peaks.length];
            double[] lo = new double[peaks.length];
            double[] hi = new double[peaks.length];
            for (int i = 0; i < peaks.length; i++) {
                int p = (int) peaks[i];
                left[i] = x[p] - x[p - 1];
                right[i] = x[p] - x[p + 1];
                lo[i] = Math.min(left[i], right[i]);
                hi[i] = Math.max(left[i], right[i]);
            }
            boolean[] keep = new boolean[peaks.length];
            Arrays.fill(keep, true);
            if (threshold.min != null) {
                boolean[] k = new Condition(threshold.min, null).keep(lo);
                for (int i = 0; i < keep.length; i++) {
                    keep[i] &= k[i];
                }
            }
            if (threshold.max != null) {
                boolean[] k = new Condition(null, threshold.max).keep(hi);
                for (int i = 0; i < keep.length; i++) {
                    keep[i] &= k[i];
                }
            }
            peaks = select(peaks, keep);
            props.put("left_thresholds", left);
            props.put("right_thresholds", right);
            selectAll(props, keep);
        }
        if (distance != null) {
            double[] priority = new double[peaks.length];
            for (int i = 0; i < priority.length; i++) {
                priority[i] = x[(int) peaks[i]];
            }
            boolean[] keep = selectByDistance(peaks, priority, distance);
            peaks = select(peaks, keep);
            selectAll(props, keep);
        }
        if (prominence != null || width != null) {
            Prominences pr = prominences(x, peaks, wlen);
            props.put("prominences", pr.prominences);
            props.put("left_bases", pr.leftBases);
            props.put("right_bases", pr.rightBases);
        }
        if (prominence != null) {
            boolean[] keep = prominence.keep((double[]) props.get("prominences"));
            peaks = select(peaks, keep);
            selectAll(props, keep);
        }
        if (width != null) {
            Widths w = widths(x, peaks, relHeight, (double[]) props.get("prominences"),
                    (long[]) props.get("left_bases"), (long[]) props.get("right_bases"));
            props.put("widths", w.widths);
            props.put("width_heights", w.widthHeights);
            props.put("left_ips", w.leftIps);
            props.put("right_ips", w.rightIps);
            boolean[] keep = width.keep(w.widths);
            peaks = select(peaks, keep);
            selectAll(props, keep);
        }
        return new Found(peaks, props);
    }
}
