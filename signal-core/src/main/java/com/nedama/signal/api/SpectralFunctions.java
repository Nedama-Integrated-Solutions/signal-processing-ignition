package com.nedama.signal.api;

import com.nedama.signal.Complex;
import com.nedama.signal.SignalException;
import com.nedama.signal.Spectral;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * welch, periodogram, spectrogram and hilbert, with scipy 1.18.1's arguments and results.
 *
 * <p>Differences from scipy: nperseg, noverlap, nfft and N must be integers (scipy truncates floats in some places
 * and fails on them in others); detrend is 'constant', 'linear' or off (scipy also accepts a Python function);
 * spectrogram accepts a plain list (scipy's needs a numpy array); signals are 1-D and numbers finite.
 */
final class SpectralFunctions {

    private SpectralFunctions() {
    }

    static void register(Registry r) {
        r.define(Signature.of("welch", Param.required("x"), Param.optional("fs", 1.0),
                        Param.optional("window", "hann_periodic"), Param.optional("nperseg", null),
                        Param.optional("noverlap", null), Param.optional("nfft", null),
                        Param.optional("detrend", "constant"), Param.optional("return_onesided", Boolean.TRUE),
                        Param.optional("scaling", "density"), Param.optional("axis", -1L),
                        Param.optional("average", "mean")),
                SpectralFunctions::welch);
        r.define(Signature.of("periodogram", Param.required("x"), Param.optional("fs", 1.0),
                        Param.optional("window", "boxcar"), Param.optional("nfft", null),
                        Param.optional("detrend", "constant"), Param.optional("return_onesided", Boolean.TRUE),
                        Param.optional("scaling", "density"), Param.optional("axis", -1L)),
                SpectralFunctions::periodogram);
        r.define(Signature.of("spectrogram", Param.required("x"), Param.optional("fs", 1.0),
                        Param.optional("window", Arrays.asList("tukey_periodic", 0.25)),
                        Param.optional("nperseg", null), Param.optional("noverlap", null),
                        Param.optional("nfft", null), Param.optional("detrend", "constant"),
                        Param.optional("return_onesided", Boolean.TRUE), Param.optional("scaling", "density"),
                        Param.optional("axis", -1L), Param.optional("mode", "psd")),
                SpectralFunctions::spectrogram);
        r.define(Signature.of("hilbert", Param.required("x"), Param.optional("N", null),
                        Param.optional("axis", -1L)),
                SpectralFunctions::hilbert);
    }

    // Arguments

    private static Integer optInt(Args a, String name) {
        if (a.isNone(name)) {
            return null;
        }
        Object v = a.raw(name);
        if (!(v instanceof Long) && !(v instanceof Boolean)) {
            throw SignalException.typeError(name + " must be an integer, not " + Args.typeName(v));
        }
        long n = a.integer(name);
        if (n > Integer.MAX_VALUE / 2 || n < -(Integer.MAX_VALUE / 2)) {
            throw SignalException.valueError(name + " is out of range: " + n);
        }
        return (int) n;
    }

    private static boolean truthy(Object v) {
        if (v instanceof Boolean) {
            return (Boolean) v;
        }
        if (v instanceof Long || v instanceof Double || v instanceof BigInteger) {
            return ((Number) v).doubleValue() != 0;
        }
        if (v instanceof String) {
            return !((String) v).isEmpty();
        }
        return v != null;
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

    /** A window spec (a name, or a tuple or list starting with one); null for a window given as values. */
    private static Object spec(Object window) {
        if (window instanceof String) {
            return window;
        }
        List<?> l = list(window);
        if (l != null && !l.isEmpty() && l.get(0) instanceof String) {
            return window;
        }
        return null;
    }

    /** A window given as values: a flat list of finite numbers. */
    private static double[] values(Object window) {
        List<?> l = list(window);
        if (l == null) {
            throw SignalException.typeError("window must be a name, a (name, parameters) tuple or a list of numbers,"
                    + " not " + Args.typeName(window));
        }
        double[] w = new double[l.size()];
        for (int i = 0; i < w.length; i++) {
            Object o = l.get(i);
            if (!(o instanceof Long || o instanceof Double || o instanceof Boolean || o instanceof BigInteger)) {
                throw SignalException.valueError("window must be 1-D numbers");
            }
            w[i] = o instanceof Boolean ? ((Boolean) o ? 1 : 0) : ((Number) o).doubleValue();
            if (!Double.isFinite(w[i])) {
                throw SignalException.valueError("Parameter win must have finite entries!");
            }
        }
        return w;
    }

    /** welch's detrend: 'constant', 'linear' (or 'c', 'l'), False or None. */
    private static Spectral.Trend welchTrend(Object v) {
        if (v == null || Boolean.FALSE.equals(v)) {
            return Spectral.Trend.NONE;
        }
        return trend(v);
    }

    /** spectrogram's detrend: anything false means none. */
    private static Spectral.Trend helperTrend(Object v) {
        if (!truthy(v)) {
            return Spectral.Trend.NONE;
        }
        return trend(v);
    }

    private static Spectral.Trend trend(Object v) {
        if (v instanceof String) {
            switch ((String) v) {
                case "constant":
                case "c":
                    return Spectral.Trend.CONSTANT;
                case "linear":
                case "l":
                    return Spectral.Trend.LINEAR;
                default:
                    throw SignalException.valueError("Trend type must be 'linear' or 'constant'.");
            }
        }
        throw SignalException.valueError("detrend must be 'constant', 'linear' or False (functions are not "
                + "supported), not " + Args.typeName(v));
    }

    private static String string(Object v, String name) {
        if (v instanceof String) {
            return (String) v;
        }
        throw SignalException.valueError("Parameter " + name + "=" + Args.typeName(v) + " is not valid");
    }

    // welch and periodogram

    private static Object welchCore(double[] x, double fs, Object window, Integer npersegArg, Integer noverlapArg,
                                    Integer nfftArg, Object detrend, boolean onesided, Object scaling,
                                    Object average) {
        int n = x.length;
        if (n == 0) {
            return Tuple.of(new double[0], new double[0]);
        }
        Object spec = spec(window);
        double[] win;
        int nperseg;
        if (spec != null) {
            nperseg = npersegArg != null ? npersegArg : 256;
            if (nperseg < 1) {
                throw SignalException.valueError("Parameter nperseg=" + nperseg + " is not a positive integer!");
            }
            if (n < nperseg) {
                nperseg = n;
            }
            win = WindowFunctions.window(spec, nperseg, true);
        } else {
            win = values(window);
            nperseg = npersegArg != null ? npersegArg : win.length;
        }
        if (nperseg != win.length) {
            throw SignalException.valueError("nperseg=" + nperseg + " does not equal len(win)=" + win.length);
        }
        int nfft = nfftArg != null ? nfftArg : nperseg;
        if (nfft < nperseg) {
            throw SignalException.valueError("nfft=" + nfft + " must be greater than or equal to nperseg="
                    + nperseg + "!");
        }
        int noverlap = noverlapArg != null ? noverlapArg : nperseg / 2;
        if (noverlap >= nperseg) {
            throw SignalException.valueError("noverlap=" + noverlap + " must be less than nperseg=" + nperseg + "!");
        }
        if (win.length == 0) {
            throw SignalException.valueError("Parameter win must be 1d, but win.shape=(0,)!");
        }
        String sc = string(scaling, "scaling");
        Spectral.Trend trend = welchTrend(detrend);
        String avg = average instanceof String ? (String) average : String.valueOf(average);
        Spectral.Psd r = Spectral.welch(x, fs, win, noverlap, nfft, trend, onesided, sc, avg);
        return Tuple.of(r.f, r.p);
    }

    private static Object welch(Args a) {
        double[] x = a.array("x");
        double fs = a.real("fs");
        a.axis("axis");
        return welchCore(x, fs, a.raw("window"), optInt(a, "nperseg"), optInt(a, "noverlap"), optInt(a, "nfft"),
                a.raw("detrend"), truthy(a.raw("return_onesided")), a.raw("scaling"), a.raw("average"));
    }

    private static Object periodogram(Args a) {
        double[] x = a.array("x");
        double fs = a.real("fs");
        a.axis("axis");
        if (x.length == 0) {
            return Tuple.of(new double[0], new double[0]);
        }
        Object window = a.isNone("window") ? "boxcar" : a.raw("window");
        Integer nfft = optInt(a, "nfft");
        int nperseg;
        if (nfft == null) {
            nperseg = x.length;
        } else if (nfft == x.length) {
            nperseg = nfft;
        } else if (nfft > x.length) {
            nperseg = x.length;
        } else {
            if (nfft < 0) {
                throw SignalException.valueError("nfft must not be negative");
            }
            x = Arrays.copyOf(x, nfft);
            nperseg = nfft;
            nfft = null;
        }
        return welchCore(x, fs, window, nperseg, 0, nfft, a.raw("detrend"), truthy(a.raw("return_onesided")),
                a.raw("scaling"), "mean");
    }

    // spectrogram

    private static Object spectrogram(Args a) {
        Object mode = a.raw("mode");
        if (!"psd".equals(mode) && !"complex".equals(mode) && !"magnitude".equals(mode) && !"angle".equals(mode)
                && !"phase".equals(mode)) {
            throw SignalException.valueError("unknown value for mode " + mode
                    + ", must be one of ['psd', 'complex', 'magnitude', 'angle', 'phase']");
        }
        double[] x = a.array("x");
        double fs = a.real("fs");
        a.axis("axis");
        int n = x.length;
        Integer npersegArg = optInt(a, "nperseg");
        Object window = a.raw("window");
        Object spec = spec(window);
        double[] win;
        int nperseg;
        if (spec != null) {
            nperseg = npersegArg != null ? npersegArg : 256;
            if (nperseg > n) {
                nperseg = n;
            }
            win = WindowFunctions.window(spec, nperseg, true);
        } else {
            win = values(window);
            if (n < win.length) {
                throw SignalException.valueError("window is longer than input signal");
            }
            if (npersegArg != null && npersegArg != win.length) {
                throw SignalException.valueError("value specified for nperseg is different from length of window");
            }
            nperseg = win.length;
        }
        Integer noverlapArg = optInt(a, "noverlap");
        int noverlap = noverlapArg != null ? noverlapArg : nperseg / 8;
        if (n == 0) {
            return Tuple.of(new double[0], new double[0], new ArrayList<double[]>());
        }
        if (nperseg < 1) {
            throw SignalException.valueError("nperseg must be a positive integer");
        }
        Integer nfftArg = optInt(a, "nfft");
        int nfft = nfftArg != null ? nfftArg : nperseg;
        if (nfft < nperseg) {
            throw SignalException.valueError("nfft must be greater than or equal to nperseg.");
        }
        if (noverlap >= nperseg) {
            throw SignalException.valueError("noverlap must be less than nperseg.");
        }
        Spectral.Trend trend = helperTrend(a.raw("detrend"));
        String scaling = string(a.raw("scaling"), "scaling");
        Spectral.Spectrogram s = Spectral.spectrogram(x, fs, win, noverlap, nfft, trend,
                truthy(a.raw("return_onesided")), scaling, (String) mode);
        List<Object> rows = new ArrayList<>();
        if (s.complex != null) {
            for (Complex[] row : s.complex) {
                rows.add(row);
            }
        } else {
            for (double[] row : s.real) {
                rows.add(row);
            }
        }
        return Tuple.of(s.f, s.t, rows);
    }

    // hilbert

    private static Object hilbert(Args a) {
        double[] x = a.array("x");
        a.axis("axis");
        Integer n = optInt(a, "N");
        return Spectral.hilbert(x, n != null ? n : x.length);
    }
}
