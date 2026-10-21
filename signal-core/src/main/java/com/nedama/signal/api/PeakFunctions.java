package com.nedama.signal.api;

import com.nedama.signal.Peaks;
import com.nedama.signal.SignalException;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

/**
 * find_peaks, peak_prominences and peak_widths, with scipy 1.18.1's arguments and results.
 *
 * <p>Differences from scipy: when equal-height peaks compete under {@code distance}, ties are ranked by position
 * (scipy's order depends on numpy's unstable sort); peak_widths accepts prominence_data as lists (scipy needs numpy
 * arrays, which Ignition scripts cannot make); per-sample condition arrays (numpy arrays of len(x) in scipy) are not
 * available, as in scipy a plain list is compared element by element against the peaks left.
 */
final class PeakFunctions {

    private PeakFunctions() {
    }

    static void register(Registry r) {
        r.define(Signature.of("find_peaks", Param.required("x"), Param.optional("height", null),
                        Param.optional("threshold", null), Param.optional("distance", null),
                        Param.optional("prominence", null), Param.optional("width", null),
                        Param.optional("wlen", null), Param.optional("rel_height", 0.5),
                        Param.optional("plateau_size", null)),
                PeakFunctions::findPeaks);
        r.define(Signature.of("peak_prominences", Param.required("x"), Param.required("peaks"),
                        Param.optional("wlen", null)),
                PeakFunctions::peakProminences);
        r.define(Signature.of("peak_widths", Param.required("x"), Param.required("peaks"),
                        Param.optional("rel_height", 0.5), Param.optional("prominence_data", null),
                        Param.optional("wlen", null)),
                PeakFunctions::peakWidths);
    }

    // Arguments

    private static double number(Object o, String name) {
        if (o instanceof Boolean) {
            return (Boolean) o ? 1.0 : 0.0;
        }
        if (o instanceof Long || o instanceof Double || o instanceof BigInteger) {
            double d = ((Number) o).doubleValue();
            if (!Double.isFinite(d)) {
                throw SignalException.valueError(name + " must be finite");
            }
            return d;
        }
        throw SignalException.typeError("'<=' not supported for " + name + " of type "
                + (o == null ? "None" : o.getClass().getSimpleName()));
    }

    private static List<?> sequence(Object o) {
        if (o instanceof List) {
            return (List<?>) o;
        }
        if (o instanceof double[]) {
            List<Object> l = new ArrayList<>();
            for (double d : (double[]) o) {
                l.add(d);
            }
            return l;
        }
        if (o instanceof long[]) {
            List<Object> l = new ArrayList<>();
            for (long d : (long[]) o) {
                l.add(d);
            }
            return l;
        }
        return null;
    }

    private static Peaks.Border border(Object o, String name) {
        if (o == null) {
            return null;
        }
        List<?> seq = sequence(o);
        if (seq != null) {
            double[] v = new double[seq.size()];
            for (int i = 0; i < v.length; i++) {
                if (sequence(seq.get(i)) != null) {
                    throw SignalException.valueError(name + ": a border must be a number or a flat list");
                }
                v[i] = number(seq.get(i), name);
            }
            return Peaks.Border.of(v);
        }
        return Peaks.Border.of(number(o, name));
    }

    /** scipy's _unpack_condition_args: a pair is (min, max); anything else is the minimum. */
    private static Peaks.Condition condition(Args a, String name) {
        Object v = a.raw(name);
        if (v == null) {
            return null;
        }
        if (v instanceof String) {
            throw SignalException.typeError("'<=' not supported between instances of 'str' and 'float' (" + name
                    + ")");
        }
        List<?> seq = sequence(v);
        if (seq != null && seq.size() == 2) {
            return new Peaks.Condition(border(seq.get(0), name), border(seq.get(1), name));
        }
        return new Peaks.Condition(border(v, name), null);
    }

    /** scipy's _arg_wlen_as_expected: None is -1; otherwise more than 1, rounded up if a float. */
    private static long wlen(Args a) {
        Object v = a.raw("wlen");
        if (v == null) {
            return -1;
        }
        if (v instanceof String || sequence(v) != null) {
            throw SignalException.typeError("'<' not supported between instances of 'int' and '"
                    + (v instanceof String ? "str" : "list") + "'");
        }
        double d = number(v, "wlen");
        if (!(1 < d)) {
            throw SignalException.valueError("`wlen` must be larger than 1, was " + v);
        }
        if (v instanceof Double) {
            double c = Math.ceil(d);
            if (c >= 9.223372036854775807E18) {
                throw SignalException.valueError("wlen is too large");
            }
            return (long) c;
        }
        if (v instanceof BigInteger) {
            throw SignalException.valueError("wlen is too large");
        }
        return (Long) v;
    }

    /** scipy's _arg_peaks_as_expected: integers (or booleans) only, a 1-D list. */
    private static long[] peaks(Object v, String name) {
        List<?> seq = sequence(v);
        if (v instanceof double[]) {
            throw SignalException.typeError("cannot safely cast `" + name + "` to dtype('intp')");
        }
        if (seq == null) {
            if (v instanceof Long || v instanceof Boolean) {
                throw SignalException.valueError("`" + name + "` must be a 1-D array");
            }
            throw SignalException.typeError("cannot safely cast `" + name + "` to dtype('intp')");
        }
        long[] out = new long[seq.size()];
        boolean nested = false;
        for (int i = 0; i < out.length; i++) {
            Object o = seq.get(i);
            if (sequence(o) != null) {
                nested = true;
                continue;
            }
            if (o instanceof Long) {
                out[i] = (Long) o;
            } else if (o instanceof Boolean) {
                out[i] = (Boolean) o ? 1 : 0;
            } else {
                throw SignalException.typeError("cannot safely cast `" + name + "` to dtype('intp')");
            }
        }
        if (nested) {
            throw SignalException.valueError("`" + name + "` must be a 1-D array");
        }
        return out;
    }

    // Functions

    private static Object findPeaks(Args a) {
        double[] x = a.array("x");
        Object d = a.raw("distance");
        Double distance = null;
        if (d != null) {
            if (d instanceof String || sequence(d) != null) {
                throw SignalException.typeError("'<' not supported between instances of '"
                        + (d instanceof String ? "str" : "list") + "' and 'int'");
            }
            distance = number(d, "distance");
        }
        Peaks.Condition plateau = condition(a, "plateau_size");
        Peaks.Condition height = condition(a, "height");
        Peaks.Condition threshold = condition(a, "threshold");
        Peaks.Condition prominence = condition(a, "prominence");
        Peaks.Condition width = condition(a, "width");
        long w = prominence != null || width != null ? wlen(a) : -1;
        double relHeight = a.real("rel_height");
        Peaks.Found f = Peaks.findPeaks(x, height, threshold, distance, prominence, width, w, relHeight, plateau);
        return Tuple.of(f.peaks, f.properties);
    }

    private static Object peakProminences(Args a) {
        double[] x = a.array("x");
        long[] p = peaks(a.raw("peaks"), "peaks");
        Peaks.Prominences r = Peaks.prominences(x, p, wlen(a));
        return Tuple.of(r.prominences, r.leftBases, r.rightBases);
    }

    private static Object peakWidths(Args a) {
        double[] x = a.array("x");
        long[] p = peaks(a.raw("peaks"), "peaks");
        double relHeight = a.real("rel_height");
        Object data = a.raw("prominence_data");
        Peaks.Widths w;
        if (data == null) {
            Peaks.Prominences r = Peaks.prominences(x, p, wlen(a));
            w = Peaks.widths(x, p, relHeight, r.prominences, r.leftBases, r.rightBases);
        } else {
            List<?> seq = sequence(data);
            if (seq == null || seq.size() != 3) {
                throw SignalException.typeError("prominence_data must be (prominences, left_bases, right_bases)");
            }
            List<?> pr = sequence(seq.get(0));
            if (pr == null) {
                throw SignalException.typeError("prominences in prominence_data must be a list of numbers");
            }
            double[] prom = new double[pr.size()];
            for (int i = 0; i < prom.length; i++) {
                prom[i] = number(pr.get(i), "prominences");
            }
            long[] lb = peaks(seq.get(1), "left_bases");
            long[] rb = peaks(seq.get(2), "right_bases");
            w = Peaks.widths(x, p, relHeight, prom, lb, rb);
        }
        return Tuple.of(w.widths, w.widthHeights, w.leftIps, w.rightIps);
    }
}
