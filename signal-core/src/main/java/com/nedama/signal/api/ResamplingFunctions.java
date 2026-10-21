package com.nedama.signal.api;

import com.nedama.signal.Resampling;
import com.nedama.signal.SignalException;
import com.nedama.signal.Windows;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

/**
 * resample, resample_poly and decimate, with scipy 1.18.1's arguments.
 *
 * <p>Differences from scipy: a window given as a list of numbers is used as the window's values (scipy needs a
 * numpy array for that); num, q and n must be integers; decimate's ftype is 'iir' or 'fir' (scipy also takes a dlti
 * system); resample_poly with padtype 'reflect' or 'antireflect' on a 1-sample signal raises a ValueError, where
 * scipy crashes the Python process; signals are 1-D and numbers finite.
 */
final class ResamplingFunctions {

    private ResamplingFunctions() {
    }

    static void register(Registry r) {
        r.define(Signature.of("resample", Param.required("x"), Param.required("num"), Param.optional("t", null),
                        Param.optional("axis", 0L), Param.optional("window", null), Param.optional("domain", "time")),
                ResamplingFunctions::resample);
        List<Object> kaiser = new ArrayList<>();
        kaiser.add("kaiser");
        kaiser.add(5.0);
        r.define(Signature.of("resample_poly", Param.required("x"), Param.required("up"), Param.required("down"),
                        Param.optional("axis", 0L), Param.optional("window", kaiser),
                        Param.optional("padtype", "constant"), Param.optional("cval", null)),
                ResamplingFunctions::resamplePoly);
        r.define(Signature.of("decimate", Param.required("x"), Param.required("q"), Param.optional("n", null),
                        Param.optional("ftype", "iir"), Param.optional("axis", -1L),
                        Param.optional("zero_phase", Boolean.TRUE)),
                ResamplingFunctions::decimate);
    }

    private static List<?> list(Object v) {
        if (v instanceof List) {
            return (List<?>) v;
        }
        if (v instanceof double[]) {
            List<Object> l = new ArrayList<>();
            for (double d : (double[]) v) {
                l.add(d);
            }
            return l;
        }
        return null;
    }

    private static boolean isNumber(Object o) {
        return o instanceof Long || o instanceof Double || o instanceof Boolean || o instanceof BigInteger;
    }

    private static double number(Object o) {
        return o instanceof Boolean ? ((Boolean) o ? 1 : 0) : ((Number) o).doubleValue();
    }

    /** A window as values (a flat list of finite numbers), or null when the window is a spec. */
    private static double[] windowValues(Object window) {
        List<?> l = list(window);
        if (l == null || (!l.isEmpty() && l.get(0) instanceof String)) {
            return null;
        }
        double[] w = new double[l.size()];
        for (int i = 0; i < w.length; i++) {
            Object o = l.get(i);
            if (list(o) != null) {
                throw SignalException.valueError("window must be 1-D");
            }
            if (!isNumber(o)) {
                throw SignalException.valueError("window values must be numbers");
            }
            w[i] = number(o);
            if (!Double.isFinite(w[i])) {
                throw SignalException.valueError("window values must be finite");
            }
        }
        return w;
    }

    /** get_window(window, n, fftbins) for a spec: a name, a (name, parameters) tuple, or a number (Kaiser beta). */
    private static double[] specWindow(Object window, int n, boolean fftbins) {
        if (isNumber(window)) {
            double beta = number(window);
            if (n <= 0) {
                throw SignalException.valueError("get_window(): Nx must be a positive integer, not " + n);
            }
            return Windows.kaiser(n, beta, !fftbins);
        }
        return WindowFunctions.window(window, n, fftbins);
    }

    private static int intArg(Args a, String name) {
        long v = a.integer(name);
        if (v > Integer.MAX_VALUE / 4 || v < -(Integer.MAX_VALUE / 4)) {
            throw SignalException.valueError(name + " is out of range: " + v);
        }
        return (int) v;
    }

    // resample

    private static Object resample(Args a) {
        Object domain = a.raw("domain");
        if (!"time".equals(domain) && !"freq".equals(domain)) {
            throw SignalException.valueError("Parameter domain='" + domain + "' not in ('time', 'freq')!");
        }
        double[] x = a.array("x");
        int num = intArg(a, "num");
        a.integer("axis");
        int nx = x.length;
        Object window = a.raw("window");
        double[] w = null;
        if (window != null) {
            w = windowValues(window);
            if (w != null) {
                if (w.length != nx) {
                    throw SignalException.valueError("window.shape=(" + w.length + ",) != (" + nx + ",), i.e., "
                            + "window length is not equal to number of frequency bins!");
                }
            } else {
                double[] g = specWindow(window, nx, true);
                w = new double[nx];
                int shift = nx / 2;
                for (int i = 0; i < nx; i++) {
                    w[(i + shift) % nx] = g[i];
                }
            }
        }
        Resampling.Resampled r = Resampling.resample(x, num, w, "freq".equals(domain));
        Object y = r.real != null ? r.real : r.complex;
        if (a.isNone("t")) {
            return y;
        }
        double[] t = a.array("t");
        if (t.length < 2) {
            throw SignalException.valueError("t must hold at least two times (index out of range)");
        }
        double sFac = (double) nx / num;
        double step = (t[1] - t[0]) * sFac;
        double[] tn = new double[num];
        for (int k = 0; k < num; k++) {
            tn[k] = t[0] + step * k;
        }
        return Tuple.of(y, tn);
    }

    // resample_poly

    /** up and down: integers, or floats with no fractional part. */
    private static long ratio(Args a, String name) {
        Object v = a.raw(name);
        if (v instanceof Long) {
            return (Long) v;
        }
        if (v instanceof Boolean) {
            return (Boolean) v ? 1 : 0;
        }
        if (v instanceof Double) {
            double d = (Double) v;
            if (d != Math.rint(d) || Double.isInfinite(d)) {
                throw SignalException.valueError(name + " must be an integer");
            }
            if (Math.abs(d) > 1e9) {
                throw SignalException.valueError(name + " is out of range");
            }
            return (long) d;
        }
        if (v instanceof BigInteger) {
            throw SignalException.valueError(name + " is out of range");
        }
        throw SignalException.typeError(name + " must be an integer, not " + Args.typeName(v));
    }

    private static Object resamplePoly(Args a) {
        double[] x = a.array("x");
        long up = ratio(a, "up");
        long down = ratio(a, "down");
        if (up < 1 || down < 1) {
            throw SignalException.valueError("up and down must be >= 1");
        }
        Object padtype = a.raw("padtype");
        if (!a.isNone("cval") && !"constant".equals(padtype)) {
            throw SignalException.valueError("cval has no effect when padtype is " + padtype);
        }
        long g = BigInteger.valueOf(up).gcd(BigInteger.valueOf(down)).longValue();
        up /= g;
        down /= g;
        a.axis("axis");
        if (up == 1 && down == 1) {
            Object raw = a.raw("x");
            List<?> l = list(raw);
            if (l != null && !l.isEmpty() && l.stream().allMatch(o -> o instanceof Long)) {
                long[] ints = new long[l.size()];
                for (int i = 0; i < ints.length; i++) {
                    ints[i] = (Long) l.get(i);
                }
                return ints;
            }
            return x;
        }
        if (up > 100_000 || down > 100_000) {
            throw SignalException.valueError("up and down must be at most 100000 after reduction");
        }
        Resampling.Mode mode = Resampling.Mode.CONSTANT;
        Resampling.Background bg = Resampling.Background.NONE;
        double cval = a.isNone("cval") ? 0 : a.real("cval");
        String pt = padtype instanceof String ? (String) padtype : "";
        switch (pt) {
            case "mean": bg = Resampling.Background.MEAN; break;
            case "median": bg = Resampling.Background.MEDIAN; break;
            case "minimum": bg = Resampling.Background.MINIMUM; break;
            case "maximum": bg = Resampling.Background.MAXIMUM; break;
            case "constant": mode = Resampling.Mode.CONSTANT; break;
            case "wrap": mode = Resampling.Mode.WRAP; break;
            case "edge": mode = Resampling.Mode.EDGE; break;
            case "smooth": mode = Resampling.Mode.SMOOTH; break;
            case "symmetric": mode = Resampling.Mode.SYMMETRIC; break;
            case "reflect": mode = Resampling.Mode.REFLECT; break;
            case "antisymmetric": mode = Resampling.Mode.ANTISYMMETRIC; break;
            case "antireflect": mode = Resampling.Mode.ANTIREFLECT; break;
            case "line": mode = Resampling.Mode.LINE; break;
            default:
                throw SignalException.valueError("padtype must be one of: maximum, mean, median, minimum, constant, "
                        + "wrap, edge, smooth, symmetric, reflect, antisymmetric, antireflect, line");
        }
        Object window = a.raw("window");
        double[] h = windowValues(window);
        if (x.length == 0) {
            if (bg == Resampling.Background.MINIMUM || bg == Resampling.Background.MAXIMUM) {
                throw SignalException.valueError("zero-size array to reduction operation which has no identity");
            }
            if (h != null && h.length == 0) {
                throw SignalException.valueError("h must be 1-D with non-zero length");
            }
            return new double[0];
        }
        if (h != null && h.length == 0) {
            return emptyWindowResult(x, (int) up, (int) down, mode, cval, bg);
        }
        Resampling.WindowMaker maker = n -> specWindow(window, n, false);
        return Resampling.resamplePoly(x, (int) up, (int) down, h, h == null ? maker : null, mode, cval, bg);
    }

    /**
     * scipy's result for a window given as an empty list: it pads the empty filter with zeros, so the result is
     * zeros of the output length, plus the background value for the mean, median, minimum and maximum padtypes.
     */
    private static double[] emptyWindowResult(double[] x, int up, int down, Resampling.Mode mode, double cval,
                                              Resampling.Background bg) {
        long nOut = (long) x.length * up;
        nOut = nOut / down + (nOut % down != 0 ? 1 : 0);
        double[] y = Resampling.resamplePoly(x, up, down, new double[] {0.0}, null, mode, cval, bg);
        return java.util.Arrays.copyOf(y, (int) nOut);
    }

    // decimate

    private static Object decimate(Args a) {
        double[] x = a.array("x");
        int q = intArg(a, "q");
        Integer n = a.isNone("n") ? null : intArg(a, "n");
        Object ftype = a.raw("ftype");
        a.axis("axis");
        Object zp = a.raw("zero_phase");
        boolean zeroPhase = zp instanceof Boolean ? (Boolean) zp : zp != null && !(isNumber(zp) && number(zp) == 0);
        if ("fir".equals(ftype)) {
            if (q == 0) {
                throw SignalException.valueError("float division by zero (q = 0)");
            }
            int taps = (n != null ? n : 20 * q) + 1;
            if (1. / q <= 0 || 1. / q >= 1) {
                throw SignalException.valueError("Invalid cutoff frequency: frequencies must be greater than 0 and "
                        + "less than fs/2.");
            }
            if (taps <= 0) {
                throw SignalException.valueError("get_window(): Nx must be a positive integer, not " + taps);
            }
            return Resampling.decimateFir(x, q, taps - 1, zeroPhase, Windows.hamming(taps, true));
        }
        if ("iir".equals(ftype)) {
            return Resampling.decimateIir(x, q, n != null ? n : 8, zeroPhase);
        }
        throw SignalException.valueError("invalid ftype");
    }

}
