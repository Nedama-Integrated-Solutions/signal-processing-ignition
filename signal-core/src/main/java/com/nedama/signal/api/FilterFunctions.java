package com.nedama.signal.api;

import com.nedama.signal.Freqz;
import com.nedama.signal.IirDesign;
import com.nedama.signal.SignalException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * butter, cheby1, cheby2, ellip, bessel, freqz, freqz_sos and sosfreqz, with scipy 1.18.1's arguments and checks
 * (scipy's iirfilter does the shared work for the five designs). Strings are case-insensitive as in scipy.
 *
 * <p>Differences from scipy: freqz's plot argument must be None (it calls back into Python); filter orders above
 * 500 (Bessel: 100) are refused; numbers must be finite and real. Bessel orders from about 90 to 100, where
 * scipy's root finding fails, still work here.
 */
final class FilterFunctions {

    private FilterFunctions() {
    }

    static final int MAX_ORDER = 500;

    /**
     * Bessel poles need a root finder whose cost grows steeply with the order. scipy's own fails from about order
     * 90; this one works to 100 (about 3 s), which bounds the time a mistyped order can take.
     */
    static final int MAX_BESSEL_ORDER = 100;

    private static final Map<String, String> BANDS = new HashMap<>();

    static {
        for (String s : new String[] {"band", "bandpass", "pass", "bp"}) {
            BANDS.put(s, "bandpass");
        }
        for (String s : new String[] {"bs", "bandstop", "bands", "stop"}) {
            BANDS.put(s, "bandstop");
        }
        for (String s : new String[] {"l", "low", "lowpass", "lp"}) {
            BANDS.put(s, "lowpass");
        }
        for (String s : new String[] {"high", "highpass", "h", "hp"}) {
            BANDS.put(s, "highpass");
        }
    }

    static void register(Registry r) {
        r.define(Signature.of("butter", Param.required("N"), Param.required("Wn"), Param.optional("btype", "low"),
                        Param.optional("analog", Boolean.FALSE), Param.optional("output", "ba"),
                        Param.optional("fs", null)),
                a -> design(a, "butter"));
        r.define(Signature.of("cheby1", Param.required("N"), Param.required("rp"), Param.required("Wn"),
                        Param.optional("btype", "low"), Param.optional("analog", Boolean.FALSE),
                        Param.optional("output", "ba"), Param.optional("fs", null)),
                a -> design(a, "cheby1"));
        r.define(Signature.of("cheby2", Param.required("N"), Param.required("rs"), Param.required("Wn"),
                        Param.optional("btype", "low"), Param.optional("analog", Boolean.FALSE),
                        Param.optional("output", "ba"), Param.optional("fs", null)),
                a -> design(a, "cheby2"));
        r.define(Signature.of("ellip", Param.required("N"), Param.required("rp"), Param.required("rs"),
                        Param.required("Wn"), Param.optional("btype", "low"), Param.optional("analog", Boolean.FALSE),
                        Param.optional("output", "ba"), Param.optional("fs", null)),
                a -> design(a, "ellip"));
        r.define(Signature.of("bessel", Param.required("N"), Param.required("Wn"), Param.optional("btype", "low"),
                        Param.optional("analog", Boolean.FALSE), Param.optional("output", "ba"),
                        Param.optional("norm", "phase"), Param.optional("fs", null)),
                a -> design(a, "bessel"));
        r.define(Signature.of("freqz", Param.required("b"), Param.optional("a", 1L), Param.optional("worN", 512L),
                        Param.optional("whole", Boolean.FALSE), Param.optional("plot", null),
                        Param.optional("fs", 2 * Math.PI), Param.optional("include_nyquist", Boolean.FALSE)),
                FilterFunctions::freqz);
        r.define(Signature.of("freqz_sos", Param.required("sos"), Param.optional("worN", 512L),
                        Param.optional("whole", Boolean.FALSE), Param.optional("fs", 2 * Math.PI)),
                FilterFunctions::freqzSos);
        r.define(Signature.of("sosfreqz", Param.required("sos"), Param.optional("worN", 512L),
                        Param.optional("whole", Boolean.FALSE), Param.optional("fs", 2 * Math.PI)),
                FilterFunctions::freqzSos);
    }

    /** The shared work of scipy's iirfilter, in its order of checks. */
    private static Object design(Args a, String ftype) {
        String fn = ftype;
        double[] wn = a.reals("Wn");
        Double fs = a.isNone("fs") ? null : a.real("fs");
        String norm = null;
        if (ftype.equals("bessel")) {
            norm = a.lower("norm");
            if (!norm.equals("phase") && !norm.equals("delay") && !norm.equals("mag")) {
                throw SignalException.valueError("'bessel_" + norm + "' is not a valid basic IIR filter.");
            }
        }
        String btype = a.lower("btype");
        String output = a.lower("output");
        boolean analog = a.flag("analog");
        if (fs != null) {
            if (analog) {
                throw SignalException.valueError("fs cannot be specified for an analog filter");
            }
            for (int i = 0; i < wn.length; i++) {
                wn[i] = wn[i] / (fs / 2);
            }
        }
        for (double w : wn) {
            if (w <= 0) {
                throw SignalException.valueError("filter critical frequencies must be greater than 0");
            }
        }
        if (wn.length > 1 && !(wn[0] < wn[1])) {
            throw SignalException.valueError("Wn[0] must be less than Wn[1]");
        }
        String band = BANDS.get(btype);
        if (band == null) {
            throw SignalException.valueError("'" + btype + "' is an invalid bandtype for filter.");
        }
        if (!output.equals("ba") && !output.equals("zpk") && !output.equals("sos")) {
            throw SignalException.valueError("'" + output + "' is not a valid output form.");
        }
        Double rp = null;
        Double rs = null;
        if (ftype.equals("cheby1") || ftype.equals("ellip")) {
            rp = a.isNone("rp") ? null : a.real("rp");
        }
        if (ftype.equals("cheby2") || ftype.equals("ellip")) {
            rs = a.isNone("rs") ? null : a.real("rs");
        }
        if (rp != null && rp < 0) {
            throw SignalException.valueError("passband ripple (rp) must be positive");
        }
        if (rs != null && rs < 0) {
            throw SignalException.valueError("stopband attenuation (rs) must be positive");
        }
        int n = a.order("N", ftype.equals("bessel") ? MAX_BESSEL_ORDER : MAX_ORDER);
        if (ftype.equals("cheby1") && rp == null) {
            throw SignalException.valueError("passband ripple (rp) must be provided to design a Chebyshev I filter.");
        }
        if (ftype.equals("cheby2") && rs == null) {
            throw SignalException.valueError("stopband attenuation (rs) must be provided to design a Chebyshev II "
                    + "filter.");
        }
        if (ftype.equals("ellip") && (rs == null || rp == null)) {
            throw SignalException.valueError("Both rp and rs must be provided to design an elliptic filter.");
        }
        IirDesign.Zpk f = IirDesign.design(fn, n, wn, rp, rs, band, analog, norm, fs);
        switch (output) {
            case "zpk":
                Object z;
                if (f.zReal) {
                    double[] zr = new double[f.z.length];
                    for (int i = 0; i < zr.length; i++) {
                        zr[i] = f.z[i].re;
                    }
                    z = zr;
                } else {
                    z = f.z;
                }
                return Tuple.of(z, f.p, f.k);
            case "ba":
                IirDesign.Poly[] ba = IirDesign.zpk2tf(f.z, f.zReal, f.p, f.k);
                return Tuple.of(value(ba[0]), value(ba[1]));
            default:
                List<double[]> rows = new ArrayList<>();
                for (double[] row : IirDesign.sos(f, analog)) {
                    rows.add(row);
                }
                return rows;
        }
    }

    private static Object value(IirDesign.Poly p) {
        return p.isReal() ? p.real : p.complex;
    }

    private static void noPlot(Args a) {
        if (!a.isNone("plot")) {
            throw SignalException.typeError("freqz(): plot is not supported; plot w and h from the result instead");
        }
    }

    /** worN: None (512), a whole count (int or bool), or frequencies (a float or a sequence of them). */
    private static Object worN(Args a) {
        Object v = a.raw("worN");
        if (v == null) {
            return 512L;
        }
        if (v instanceof Long || v instanceof Boolean || v instanceof java.math.BigInteger) {
            return a.integer("worN");
        }
        return a.reals("worN");
    }

    private static Object freqz(Args a) {
        noPlot(a);
        double[] b = a.reals("b");
        double[] den = a.reals("a");
        double fs = fs(a);
        Object w = worN(a);
        Freqz.Response r = w instanceof double[] ? Freqz.freqz(b, den, (double[]) w, fs)
                : Freqz.freqz(b, den, (Long) w, a.flag("whole"), fs, a.flag("include_nyquist"));
        return Tuple.of(r.w, r.h);
    }

    private static Object freqzSos(Args a) {
        double[][] sos = a.rows("sos", 6);
        double fs = fs(a);
        Freqz.Response r = Freqz.freqzSos(sos, worN(a), a.flag("whole"), fs);
        return Tuple.of(r.w, r.h);
    }

    private static double fs(Args a) {
        if (a.isNone("fs")) {
            throw SignalException.valueError("Sampling frequency can not be none.");
        }
        return a.real("fs");
    }
}
