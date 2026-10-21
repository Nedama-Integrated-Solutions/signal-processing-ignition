package com.nedama.signal.api;

import com.nedama.signal.SignalException;
import com.nedama.signal.Smoothing;

import java.math.BigInteger;
import java.util.List;

/**
 * savgol_coeffs, savgol_filter and medfilt, with scipy 1.18.1's arguments.
 *
 * <p>Differences from scipy: window_length, polyorder and deriv must be integers (scipy accepts a float
 * window_length in some modes); signals are 1-D; numbers must be finite.
 */
final class SmoothingFunctions {

    private SmoothingFunctions() {
    }

    static void register(Registry r) {
        r.define(Signature.of("savgol_coeffs", Param.required("window_length"), Param.required("polyorder"),
                        Param.optional("deriv", 0L), Param.optional("delta", 1.0), Param.optional("pos", null),
                        Param.optional("use", "conv")),
                SmoothingFunctions::savgolCoeffs);
        r.define(Signature.of("savgol_filter", Param.required("x"), Param.required("window_length"),
                        Param.required("polyorder"), Param.optional("deriv", 0L), Param.optional("delta", 1.0),
                        Param.optional("axis", -1L), Param.optional("mode", "interp"), Param.optional("cval", 0.0)),
                SmoothingFunctions::savgolFilter);
        r.define(Signature.of("medfilt", Param.required("volume"), Param.optional("kernel_size", null)),
                SmoothingFunctions::medfilt);
    }

    private static int small(Args a, String name) {
        long v = a.integer(name);
        if (v > 10_000_000 || v < -10_000_000) {
            throw SignalException.valueError(name + " is out of range: " + v);
        }
        return (int) v;
    }

    private static Object savgolCoeffs(Args a) {
        int w = small(a, "window_length");
        int p = small(a, "polyorder");
        int d = small(a, "deriv");
        double delta = a.real("delta");
        Double pos = a.isNone("pos") ? null : a.real("pos");
        Object use = a.raw("use");
        if (!(use instanceof String)) {
            throw SignalException.valueError("`use` must be 'conv' or 'dot'");
        }
        return Smoothing.coeffs(w, p, d, delta, pos, (String) use);
    }

    private static Object savgolFilter(Args a) {
        Object mode = a.raw("mode");
        if (!(mode instanceof String)) {
            throw SignalException.valueError("mode must be 'mirror', 'constant', 'nearest' 'wrap' or 'interp'.");
        }
        double[] x = a.array("x");
        int w = small(a, "window_length");
        int p = small(a, "polyorder");
        int d = small(a, "deriv");
        double delta = a.real("delta");
        a.axis("axis");
        double cval = a.real("cval");
        return Smoothing.filter(x, w, p, d, delta, (String) mode, cval);
    }

    /** kernel_size: None (3), an odd positive integer, or a list holding one. */
    private static int kernel(Args a) {
        Object v = a.raw("kernel_size");
        if (v == null) {
            return 3;
        }
        long k;
        if (v instanceof List || v instanceof double[]) {
            long[] ks = a.integers("kernel_size");
            if (ks.length != 1) {
                throw SignalException.valueError("sequence argument must have length equal to input rank");
            }
            k = ks[0];
        } else {
            k = a.integer("kernel_size");
        }
        if (Math.floorMod(k, 2L) != 1) {
            throw SignalException.valueError("Each element of kernel_size should be odd.");
        }
        if (k < 0) {
            throw SignalException.valueError("negative dimensions are not allowed");
        }
        if (k > 10_000_001) {
            throw SignalException.valueError("kernel_size is too large");
        }
        return (int) k;
    }

    /**
     * medfilt keeps scipy's types: integers in, integers out; all booleans are refused (numpy's bool dtype); any
     * float makes the whole signal float. A single number counts as a one-sample signal.
     */
    private static Object medfilt(Args a) {
        Object v = a.raw("volume");
        int k = kernel(a);
        boolean scalar = v instanceof Long || v instanceof Double || v instanceof BigInteger;
        if (scalar) {
            v = java.util.Collections.singletonList(v);
        }
        if (v instanceof List) {
            List<?> list = (List<?>) v;
            boolean allBool = !list.isEmpty();
            boolean anyFloat = false;
            for (Object o : list) {
                allBool &= o instanceof Boolean;
                anyFloat |= o instanceof Double;
                if (o instanceof BigInteger) {
                    throw SignalException.valueError("dtype=object is not supported by medfilt");
                }
            }
            if (allBool) {
                throw SignalException.valueError("dtype=bool is not supported by medfilt");
            }
            if (!anyFloat && !list.isEmpty()) {
                boolean ints = true;
                for (Object o : list) {
                    ints &= o instanceof Long || o instanceof Boolean;
                }
                if (ints) {
                    long[] xs = new long[list.size()];
                    for (int i = 0; i < xs.length; i++) {
                        Object o = list.get(i);
                        xs[i] = o instanceof Boolean ? ((Boolean) o ? 1L : 0L) : (Long) o;
                    }
                    return Smoothing.medfilt(xs, k);
                }
            }
        }
        if (scalar && ((List<?>) v).get(0) instanceof Long) {
            return Smoothing.medfilt(new long[] {(Long) ((List<?>) v).get(0)}, k);
        }
        if (scalar) {
            double d = ((Number) ((List<?>) v).get(0)).doubleValue();
            if (!Double.isFinite(d)) {
                throw SignalException.valueError("medfilt(): volume must not contain None, NaN or infinite values");
            }
            return Smoothing.medfilt(new double[] {d}, k);
        }
        return Smoothing.medfilt(a.array("volume"), k);
    }
}
