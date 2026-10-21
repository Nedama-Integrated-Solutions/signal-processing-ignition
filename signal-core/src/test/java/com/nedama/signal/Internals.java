package com.nedama.signal;

import com.nedama.signal.api.Args;
import com.nedama.signal.api.Handler;
import com.nedama.signal.api.Param;
import com.nedama.signal.api.Signature;
import com.nedama.signal.api.Tuple;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.DoubleUnaryOperator;
import java.util.function.UnaryOperator;

/**
 * Test-only entry points to the core's internal building blocks (FFT, special functions, root finder,
 * minimiser), so random_check.py can compare each one with its scipy or numpy counterpart directly. Names
 * start with an underscore. They live in test sources and never reach system.signal.
 */
final class Internals {

    private Internals() {
    }

    private static final class Entry {
        final Signature signature;
        final Handler handler;

        Entry(Signature signature, Handler handler) {
            this.signature = signature;
            this.handler = handler;
        }
    }

    private static final Map<String, Entry> ENTRIES = new LinkedHashMap<>();

    private static void define(Signature s, Handler h) {
        ENTRIES.put(s.function(), new Entry(s, h));
    }

    static {
        define(Signature.of("_fft", Param.required("re"), Param.required("im")),
                a -> pair(Fft.fft(a.array("re"), a.array("im"))));
        define(Signature.of("_ifft", Param.required("re"), Param.required("im")),
                a -> pair(Fft.ifft(a.array("re"), a.array("im"))));
        define(Signature.of("_rfft", Param.required("x")),
                a -> pair(Fft.rfft(a.array("x"))));
        define(Signature.of("_irfft", Param.required("re"), Param.required("im"), Param.optional("n", null)),
                a -> {
                    double[] re = a.array("re");
                    int n = a.isNone("n") ? 2 * (re.length - 1) : (int) a.integer("n");
                    return Fft.irfft(re, a.array("im"), n);
                });
        define(Signature.of("_i0", Param.required("x")), a -> map(a.array("x"), Special::i0));
        define(Signature.of("_ellipk", Param.required("m")), a -> map(a.array("m"), Special::ellipk));
        define(Signature.of("_ellipkm1", Param.required("p")), a -> map(a.array("p"), Special::ellipkm1));
        define(Signature.of("_ellipj", Param.required("u"), Param.required("m")), Internals::ellipj);
        define(Signature.of("_fminbound", Param.required("func"), Param.required("x1"), Param.required("x2"),
                        Param.optional("params", new java.util.ArrayList<>()), Param.optional("xtol", 1e-5),
                        Param.optional("maxfun", 500L)),
                Internals::fminbound);
        define(Signature.of("_bessel_zeros", Param.required("N")), a -> {
            Complex[] z = IirDesign.besselZeros((int) a.integer("N"));
            double[] re = new double[z.length];
            double[] im = new double[z.length];
            for (int i = 0; i < z.length; i++) {
                re[i] = z[i].re;
                im[i] = z[i].im;
            }
            return Tuple.of(re, im);
        });
        define(Signature.of("_aberth", Param.required("coeffs"), Param.required("x0re"), Param.required("x0im"),
                        Param.optional("tol", 1e-15), Param.optional("maxiter", 50L)),
                Internals::aberth);
    }

    static boolean has(String name) {
        return ENTRIES.containsKey(name);
    }

    static Object call(String name, Object[] positional, Map<String, Object> keywords) {
        Entry e = ENTRIES.get(name);
        if (e == null) {
            throw new IllegalArgumentException("Unknown internal function: " + name);
        }
        return e.handler.call(e.signature.bind(positional, keywords));
    }

    private static Tuple pair(double[][] reIm) {
        return Tuple.of(reIm[0], reIm[1]);
    }

    private static double[] map(double[] x, DoubleUnaryOperator f) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = f.applyAsDouble(x[i]);
        }
        return out;
    }

    private static Object ellipj(Args a) {
        double[] u = a.array("u");
        double[] m = a.array("m");
        if (u.length != m.length) {
            throw SignalException.valueError("u and m must have the same length");
        }
        double[][] out = new double[4][u.length];
        for (int i = 0; i < u.length; i++) {
            double[] r = Special.ellipj(u[i], m[i]);
            for (int k = 0; k < 4; k++) {
                out[k][i] = r[k];
            }
        }
        return Tuple.of(out[0], out[1], out[2], out[3]);
    }

    /** Test functions for the minimiser. Only +, -, *, / and abs, so Python evaluates them bit for bit the same. */
    private static DoubleUnaryOperator testFunction(String name, double[] p) {
        switch (name) {
            case "quadratic":
                return x -> p[1] * (x - p[0]) * (x - p[0]) + p[2];
            case "quartic":
                return x -> (x * x - p[0]) * (x * x - p[0]) + p[1] * x;
            case "abs":
                return x -> Math.abs(x - p[0]);
            case "rational":
                return x -> (x - p[0]) * (x - p[0]) / (1 + x * x);
            case "piecewise":
                return x -> x < p[0] ? p[1] * (p[0] - x) : x - p[0];
            case "cubic":
                return x -> ((p[0] * x + p[1]) * x + p[2]) * x;
            case "flat":
                return x -> 1.0;
            case "nan_above":
                return x -> x > p[0] ? Double.NaN : (x - p[0]) * (x - p[0]);
            default:
                throw SignalException.valueError("unknown test function " + name);
        }
    }

    private static Object fminbound(Args a) {
        Object name = a.raw("func");
        DoubleUnaryOperator f = testFunction(String.valueOf(name), a.array("params"));
        Fminbound.Result r = Fminbound.minimize(f, a.real("x1"), a.real("x2"), a.real("xtol"),
                (int) a.integer("maxfun"));
        return Tuple.of(r.x, r.fun, (long) r.status, (long) r.nfev);
    }

    private static Object aberth(Args a) {
        double[] c = a.array("coeffs");
        double[] re = a.array("x0re");
        double[] im = a.array("x0im");
        int deg = c.length - 1;
        double[] d = new double[Math.max(deg, 0)];
        for (int i = 0; i < deg; i++) {
            d[i] = c[i] * (deg - i);
        }
        UnaryOperator<Complex> f = z -> horner(c, z);
        UnaryOperator<Complex> fp = z -> horner(d, z);
        Complex[] x0 = new Complex[re.length];
        for (int i = 0; i < re.length; i++) {
            x0[i] = Complex.of(re[i], im[i]);
        }
        Complex[] x = Aberth.solve(f, fp, x0, a.real("tol"), (int) a.integer("maxiter"));
        double[] outRe = new double[x.length];
        double[] outIm = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            outRe[i] = x[i].re;
            outIm[i] = x[i].im;
        }
        return Tuple.of(outRe, outIm);
    }

    /** numpy.polyval: Horner's rule from the highest power. */
    private static Complex horner(double[] c, Complex z) {
        Complex y = Complex.ZERO;
        for (double v : c) {
            y = y.times(z).plus(v);
        }
        return y;
    }
}
