package com.nedama.signal.api;

import com.nedama.signal.SignalException;
import com.nedama.signal.Windows;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * get_window, with scipy 1.18.1's rules: Nx is a positive integer, fftbins is True or False, and the window is
 * a name, a (name, parameters...) tuple, or a number (a Kaiser window with that beta). A name may end in
 * "_symmetric" or "_periodic" to override fftbins. Each window's parameters bind like a Python call, so too many
 * or too few fail the same way they do in scipy.
 *
 * <p>Differences from scipy: a list works like a tuple (Jython and JSON cannot always tell them apart); dpss is
 * not supported; non-finite parameters are refused.
 */
final class WindowFunctions {

    private WindowFunctions() {
    }

    /** Longest window accepted, to stop a mistyped length from exhausting gateway memory. */
    static final long MAX_LENGTH = 50_000_000L;

    private enum Needs { NONE, REQUIRED, OPTIONAL }

    @FunctionalInterface
    private interface Maker {
        double[] make(int m, boolean sym, Args params);
    }

    private static final class Spec {
        final Signature params;
        final Needs needs;
        final Maker maker;

        Spec(Signature params, Needs needs, Maker maker) {
            this.params = params;
            this.needs = needs;
            this.maker = maker;
        }
    }

    private static final Map<String, Spec> WINDOWS = buildTable();

    private static Map<String, Spec> buildTable() {
        Map<String, Spec> t = new HashMap<>();
        none(t, (m, s, a) -> Windows.barthann(m, s), "barthann", "brthan", "bth");
        none(t, (m, s, a) -> Windows.bartlett(m, s), "bartlett", "bart", "brt");
        none(t, (m, s, a) -> Windows.blackman(m, s), "blackman", "black", "blk");
        none(t, (m, s, a) -> Windows.blackmanHarris(m, s), "blackmanharris", "blackharr", "bkh");
        none(t, (m, s, a) -> Windows.bohman(m, s), "bohman", "bman", "bmn");
        none(t, (m, s, a) -> Windows.boxcar(m, s), "boxcar", "box", "ones", "rect", "rectangular");
        add(t, Signature.of("chebwin", Param.required("at")), Needs.REQUIRED,
                (m, s, a) -> Windows.chebwin(m, a.real("at"), s), "chebwin", "cheb");
        none(t, (m, s, a) -> Windows.cosine(m, s), "cosine", "halfcosine");
        add(t, Signature.of("exponential", Param.optional("center", null), Param.optional("tau", 1.0)),
                Needs.OPTIONAL,
                (m, s, a) -> Windows.exponential(m, a.isNone("center") ? null : a.real("center"), a.real("tau"), s),
                "exponential", "poisson");
        none(t, (m, s, a) -> Windows.flattop(m, s), "flattop", "flat", "flt");
        add(t, Signature.of("gaussian", Param.required("std")), Needs.REQUIRED,
                (m, s, a) -> Windows.gaussian(m, a.real("std"), s), "gaussian", "gauss", "gss");
        add(t, Signature.of("general_cosine", Param.required("a")), Needs.REQUIRED,
                (m, s, a) -> Windows.generalCosine(m, a.array("a"), s), "general cosine", "general_cosine");
        add(t, Signature.of("general_gaussian", Param.required("p"), Param.required("sig")), Needs.REQUIRED,
                (m, s, a) -> Windows.generalGaussian(m, a.real("p"), a.real("sig"), s),
                "general gaussian", "general_gaussian", "general gauss", "general_gauss", "ggs");
        add(t, Signature.of("general_hamming", Param.required("alpha")), Needs.REQUIRED,
                (m, s, a) -> Windows.generalHamming(m, a.real("alpha"), s), "general hamming", "general_hamming");
        none(t, (m, s, a) -> Windows.hamming(m, s), "hamming", "hamm", "ham");
        none(t, (m, s, a) -> Windows.hann(m, s), "hann", "han");
        add(t, Signature.of("kaiser", Param.required("beta")), Needs.REQUIRED,
                (m, s, a) -> Windows.kaiser(m, a.real("beta"), s), "kaiser", "ksr");
        add(t, Signature.of("kaiser_bessel_derived", Param.required("beta")), Needs.REQUIRED,
                (m, s, a) -> Windows.kaiserBesselDerived(m, a.real("beta"), s),
                "kaiser bessel derived", "kaiser_bessel_derived", "kbd");
        none(t, (m, s, a) -> Windows.lanczos(m, s), "lanczos", "sinc");
        none(t, (m, s, a) -> Windows.nuttall(m, s), "nuttall", "nutl", "nut");
        none(t, (m, s, a) -> Windows.parzen(m, s), "parzen", "parz", "par");
        add(t, Signature.of("taylor", Param.optional("nbar", 4L), Param.optional("sll", 30L),
                        Param.optional("norm", Boolean.TRUE)), Needs.OPTIONAL,
                (m, s, a) -> Windows.taylor(m, taylorNbar(a), a.real("sll"), a.flag("norm"), s),
                "taylor", "taylorwin");
        none(t, (m, s, a) -> Windows.triang(m, s), "triangle", "triang", "tri");
        add(t, Signature.of("tukey", Param.optional("alpha", 0.5)), Needs.OPTIONAL,
                (m, s, a) -> Windows.tukey(m, a.real("alpha"), s), "tukey", "tuk");
        return Collections.unmodifiableMap(t);
    }

    private static void none(Map<String, Spec> t, Maker maker, String... names) {
        add(t, Signature.of(names[0]), Needs.NONE, maker, names);
    }

    private static void add(Map<String, Spec> t, Signature params, Needs needs, Maker maker, String... names) {
        Spec spec = new Spec(params, needs, maker);
        for (String name : names) {
            t.put(name, spec);
        }
    }

    private static int taylorNbar(Args a) {
        long nbar = a.integer("nbar");
        if (nbar < 1 || nbar > 100_000) {
            throw SignalException.valueError("taylor(): nbar must be between 1 and 100000, not " + nbar);
        }
        return (int) nbar;
    }

    static void register(Registry r) {
        r.define(Signature.of("get_window",
                        Param.required("window"),
                        Param.required("Nx"),
                        Param.optional("fftbins", Boolean.TRUE)),
                WindowFunctions::getWindow);
    }

    private static Object getWindow(Args a) {
        int nx = length(a.raw("Nx"));
        Object fftbins = a.raw("fftbins");
        if (!(fftbins instanceof Boolean)) {
            throw SignalException.valueError("get_window(): fftbins must be True or False, not "
                    + Args.typeName(fftbins));
        }
        Object window = a.raw("window");
        if (!(window instanceof String) && !(window instanceof List) && !(window instanceof double[])) {
            if (window == null) {
                throw SignalException.valueError("get_window(): window must be a string, a tuple or a number, not None");
            }
            return Windows.kaiser(nx, a.real("window"), !(Boolean) fftbins);
        }
        return window(window, nx, (Boolean) fftbins);
    }

    /**
     * get_window for a window given by name or (name, parameters...), as the spectral functions use it: a string
     * or a list or tuple whose first entry is the name.
     */
    static double[] window(Object window, int nx, boolean fftbins) {
        if (nx <= 0) {
            throw SignalException.valueError("get_window(): Nx must be a positive integer, not " + nx);
        }
        boolean sym = !fftbins;

        String name;
        Object[] params;
        if (window instanceof String) {
            name = (String) window;
            params = new Object[0];
        } else {
            List<Object> items = sequence(window);
            if (items.isEmpty() || !(items.get(0) instanceof String)) {
                throw SignalException.valueError("get_window(): the first entry of a window tuple must be the "
                        + "window's name, such as ('kaiser', 8.6)");
            }
            name = (String) items.get(0);
            params = items.subList(1, items.size()).toArray();
        }
        if (name.endsWith("_symmetric")) {
            sym = true;
            name = name.substring(0, name.length() - "_symmetric".length());
        } else if (name.endsWith("_periodic")) {
            sym = false;
            name = name.substring(0, name.length() - "_periodic".length());
        }

        Spec spec = WINDOWS.get(name);
        if (spec == null) {
            if (name.equals("dpss")) {
                throw SignalException.valueError("get_window(): dpss windows are not supported");
            }
            throw SignalException.valueError("get_window(): invalid window name '" + name + "'");
        }
        if (params.length > 0 && spec.needs == Needs.NONE) {
            throw SignalException.valueError("get_window(): '" + name + "' does not take parameters");
        }
        if (params.length == 0 && spec.needs == Needs.REQUIRED) {
            throw SignalException.valueError("get_window(): '" + name + "' needs parameters, for example ('"
                    + name + "', ...)");
        }
        return spec.maker.make(nx, sym, spec.params.bind(params, Collections.emptyMap()));
    }

    /** Nx: a positive integer. Booleans are refused, as in scipy. */
    private static int length(Object v) {
        if (!(v instanceof Long)) {
            throw SignalException.valueError("get_window(): Nx must be a positive integer, not " + Args.typeName(v));
        }
        long n = (Long) v;
        if (n <= 0) {
            throw SignalException.valueError("get_window(): Nx must be a positive integer, not " + n);
        }
        if (n > MAX_LENGTH) {
            throw SignalException.valueError("get_window(): Nx is too large (at most " + MAX_LENGTH + ")");
        }
        return (int) n;
    }

    private static List<Object> sequence(Object window) {
        if (window instanceof double[]) {
            List<Object> out = new ArrayList<>();
            for (double d : (double[]) window) {
                out.add(d);
            }
            return out;
        }
        List<Object> out = new ArrayList<>();
        for (Object o : (List<?>) window) {
            out.add(o);
        }
        return out;
    }
}
