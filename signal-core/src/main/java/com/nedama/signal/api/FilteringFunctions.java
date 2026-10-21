package com.nedama.signal.api;

import com.nedama.signal.Filtering;
import com.nedama.signal.SignalException;

import java.util.ArrayList;
import java.util.List;

/**
 * lfilter, lfilter_zi, sosfilt, sosfilt_zi, filtfilt and sosfiltfilt, with scipy 1.18.1's arguments.
 *
 * <p>Differences from scipy: signals are 1-D (axis -1 or 0); filtfilt's method='gust' (Gustafsson's method) is
 * not supported and raises a ValueError, 'pad' being the default and the method scipy recommends for most uses.
 */
final class FilteringFunctions {

    private FilteringFunctions() {
    }

    static void register(Registry r) {
        r.define(Signature.of("lfilter", Param.required("b"), Param.required("a"), Param.required("x"),
                        Param.optional("axis", -1L), Param.optional("zi", null)),
                FilteringFunctions::lfilter);
        r.define(Signature.of("lfilter_zi", Param.required("b"), Param.required("a")),
                a -> Filtering.lfilterZi(a.reals("b"), a.reals("a")));
        r.define(Signature.of("sosfilt", Param.required("sos"), Param.required("x"), Param.optional("axis", -1L),
                        Param.optional("zi", null)),
                FilteringFunctions::sosfilt);
        r.define(Signature.of("sosfilt_zi", Param.required("sos")),
                a -> rows(Filtering.sosfiltZi(a.rows("sos", 6, false))));
        r.define(Signature.of("filtfilt", Param.required("b"), Param.required("a"), Param.required("x"),
                        Param.optional("axis", -1L), Param.optional("padtype", "odd"), Param.optional("padlen", null),
                        Param.optional("method", "pad"), Param.optional("irlen", null)),
                FilteringFunctions::filtfilt);
        r.define(Signature.of("sosfiltfilt", Param.required("sos"), Param.required("x"), Param.optional("axis", -1L),
                        Param.optional("padtype", "odd"), Param.optional("padlen", null)),
                FilteringFunctions::sosfiltfilt);
    }

    private static List<double[]> rows(double[][] m) {
        List<double[]> out = new ArrayList<>();
        for (double[] row : m) {
            out.add(row);
        }
        return out;
    }

    private static Object lfilter(Args a) {
        double[] b = a.reals("b");
        double[] den = a.reals("a");
        double[] x = a.array("x");
        a.axis("axis");
        double[] zi = a.isNone("zi") ? null : a.array("zi");
        Filtering.Result r = Filtering.lfilter(b, den, x, zi);
        return zi == null ? r.y : Tuple.of(r.y, r.zf);
    }

    private static Object sosfilt(Args a) {
        double[] x = a.array("x");
        double[][] sos = a.rows("sos", 6, true);
        a.axis("axis");
        double[][] zi = a.isNone("zi") ? null : a.rows("zi", 2, false);
        Filtering.SosResult r = Filtering.sosfilt(sos, x, zi);
        return zi == null ? r.y : Tuple.of(r.y, rows(r.zf));
    }

    /** padtype: 'odd', 'even', 'constant' or None, exactly as scipy spells them. */
    private static Filtering.PadType padtype(Args a) {
        Object v = a.raw("padtype");
        if (v == null) {
            return Filtering.PadType.NONE;
        }
        if (v instanceof String) {
            switch ((String) v) {
                case "odd":
                    return Filtering.PadType.ODD;
                case "even":
                    return Filtering.PadType.EVEN;
                case "constant":
                    return Filtering.PadType.CONSTANT;
                default:
                    break;
            }
        }
        throw SignalException.valueError("Unknown value '" + v + "' given to padtype. padtype must be 'even', "
                + "'odd', 'constant', or None.");
    }

    private static Long padlen(Args a) {
        return a.isNone("padlen") ? null : a.integer("padlen");
    }

    private static Object filtfilt(Args a) {
        double[] b = a.reals("b");
        double[] den = a.reals("a");
        double[] x = a.array("x");
        a.axis("axis");
        Object method = a.raw("method");
        if (!"pad".equals(method) && !"gust".equals(method)) {
            throw SignalException.valueError("method must be 'pad' or 'gust'.");
        }
        if ("gust".equals(method)) {
            throw SignalException.valueError("filtfilt(): method='gust' is not supported; use the default 'pad'");
        }
        return Filtering.filtfilt(b, den, x, padtype(a), padlen(a));
    }

    private static Object sosfiltfilt(Args a) {
        double[][] sos = a.rows("sos", 6, true);
        double[] x = a.array("x");
        a.axis("axis");
        return Filtering.sosfiltfilt(sos, x, padtype(a), padlen(a));
    }
}
