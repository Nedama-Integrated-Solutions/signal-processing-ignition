package com.nedama.signal;

/**
 * Minimum filter orders, ported from scipy.signal's buttord, cheb1ord, cheb2ord and ellipord (scipy 1.18.1). Given
 * passband and stopband edges and the loss allowed in each band, each returns the lowest order N that meets the
 * specification and the natural frequency Wn to design with.
 *
 * <p>scipy computes several steps with Python's math module, which raises for a domain error (the square root of
 * a negative number, the log of zero, division by zero) where Java would return NaN or infinity. Those steps raise
 * a ValueError here too, through the py* helpers.
 */
public final class OrderSelection {

    private OrderSelection() {
    }

    /** The order and natural frequency; {@code wn} has one value for low-pass and high-pass, two for band filters. */
    public static final class Result {
        public final long n;
        public final double[] wn;

        Result(long n, double[] wn) {
            this.n = n;
            this.wn = wn;
        }
    }

    private enum Kind { BUTTER, CHEBY, ELLIP }

    // Python math semantics

    private static double pyDiv(double a, double b) {
        if (b == 0.0) {
            throw SignalException.valueError("float division by zero");
        }
        return a / b;
    }

    private static double pySqrt(double x) {
        if (x < 0) {
            throw SignalException.valueError("math domain error");
        }
        return Math.sqrt(x);
    }

    private static double pyLog10(double x) {
        if (x <= 0) {
            throw SignalException.valueError("math domain error");
        }
        return Math.log10(x);
    }

    private static double pyAcosh(double x) {
        if (x < 1) {
            throw SignalException.valueError("math domain error");
        }
        return Special.acosh(x);
    }

    private static double pyCosh(double x) {
        double v = Math.cosh(x);
        if (Double.isInfinite(v) && !Double.isInfinite(x)) {
            throw SignalException.valueError("math range error");
        }
        return v;
    }

    /** int(math.ceil(x)) / int(numpy.ceil(x)): NaN and infinity cannot become an integer. */
    private static long pyCeilInt(double x) {
        if (Double.isNaN(x)) {
            throw SignalException.valueError("cannot convert float NaN to integer");
        }
        if (Double.isInfinite(x)) {
            throw SignalException.valueError("cannot convert float infinity to integer");
        }
        double c = Math.ceil(x);
        if (Math.abs(c) >= 9.2e18) {
            throw SignalException.valueError("filter order is too large");
        }
        return (long) c;
    }

    /** numpy.min: NaN if any value is NaN. */
    private static double npMin(double[] v) {
        double m = Double.POSITIVE_INFINITY;
        for (double x : v) {
            if (Double.isNaN(x)) {
                return Double.NaN;
            }
            m = Math.min(m, x);
        }
        return m;
    }

    /** Python's builtin min over an array: replaces only on a strict less-than, so NaN after the first is ignored. */
    private static double pyMin(double[] v) {
        double m = v[0];
        for (int i = 1; i < v.length; i++) {
            if (v[i] < m) {
                m = v[i];
            }
        }
        return m;
    }

    // Shared steps

    static void validateGpassGstop(double gpass, double gstop) {
        if (gpass <= 0.0) {
            throw SignalException.valueError("gpass should be larger than 0.0");
        } else if (gstop <= 0.0) {
            throw SignalException.valueError("gstop should be larger than 0.0");
        } else if (gpass > gstop) {
            throw SignalException.valueError("gpass should be smaller than gstop");
        }
    }

    /** Band type: 1 low-pass, 2 high-pass, 3 band-stop, 4 band-pass. Divides wp and ws by fs/2 when fs is given. */
    private static int filterType(double[] wp, double[] ws, boolean analog, Double fs) {
        if (fs != null) {
            if (analog) {
                throw SignalException.valueError("fs cannot be specified for an analog filter");
            }
            for (int i = 0; i < wp.length; i++) {
                wp[i] = 2 * wp[i] / fs;
            }
            for (int i = 0; i < ws.length; i++) {
                ws[i] = 2 * ws[i] / fs;
            }
        }
        if (wp.length == 0 || ws.length == 0) {
            throw SignalException.valueError("wp and ws must not be empty");
        }
        int type = 2 * (wp.length - 1) + 1;
        if (wp[0] >= ws[0]) {
            type += 1;
        }
        return type;
    }

    private static double[] preWarp(double[] w, boolean analog) {
        if (analog) {
            return w.clone();
        }
        double[] out = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            out[i] = Math.tan(Math.PI * w[i] / 2.0);
        }
        return out;
    }

    /** a op b with numpy broadcasting for 1-D arrays (equal lengths, or one of length 1). */
    private static double[] broadcast(double[] a, double[] b, java.util.function.DoubleBinaryOperator op) {
        if (a.length != b.length && a.length != 1 && b.length != 1) {
            throw SignalException.valueError("operands could not be broadcast together with shapes (" + a.length
                    + ",) (" + b.length + ",)");
        }
        int n = Math.max(a.length, b.length);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = op.applyAsDouble(a[a.length == 1 ? 0 : i], b[b.length == 1 ? 0 : i]);
        }
        return out;
    }

    /** scipy's band_stop_obj: the order needed, as a function of one passband edge of a band-stop filter. */
    private static double bandStopObj(double wp, int ind, double[] passb, double[] stopb, double gpass, double gstop,
                                      Kind kind) {
        double[] pc = passb.clone();
        pc[ind] = wp;
        double[] nat = new double[stopb.length];
        for (int i = 0; i < stopb.length; i++) {
            nat[i] = Math.abs(stopb[i] * (pc[0] - pc[1]) / (stopb[i] * stopb[i] - pc[0] * pc[1]));
        }
        double natMin = pyMin(nat);
        switch (kind) {
            case BUTTER: {
                double gStop = Math.pow(10, 0.1 * Math.abs(gstop));
                double gPass = Math.pow(10, 0.1 * Math.abs(gpass));
                return Math.log10(pyDiv(gStop - 1.0, gPass - 1.0)) / (2 * Math.log10(natMin));
            }
            case CHEBY: {
                double gStop = Math.pow(10, 0.1 * Math.abs(gstop));
                double gPass = Math.pow(10, 0.1 * Math.abs(gpass));
                return Special.acosh(Math.sqrt(pyDiv(gStop - 1.0, gPass - 1.0))) / Special.acosh(natMin);
            }
            default: {
                double gStop = Math.pow(10, 0.1 * gstop);
                double gPass = Math.pow(10, 0.1 * gpass);
                double arg1 = Math.sqrt(pyDiv(gPass - 1.0, gStop - 1.0));
                double arg0 = 1.0 / natMin;
                double d00 = Special.ellipk(arg0 * arg0);
                double d01 = Special.ellipk(1 - arg0 * arg0);
                double d10 = Special.ellipk(arg1 * arg1);
                double d11 = Special.ellipk(1 - arg1 * arg1);
                return d00 * d11 / (d01 * d10);
            }
        }
    }

    /** scipy's _find_nat_freq: returns {nat} and, for band-stop, replaces passb with the optimised edges. */
    private static double findNatFreq(double[] stopb, double[][] passbRef, double gpass, double gstop, int type,
                                      Kind kind) {
        double[] passb = passbRef[0];
        double[] nat;
        if (type == 1) {
            nat = broadcast(stopb, passb, (s, p) -> s / p);
        } else if (type == 2) {
            nat = broadcast(passb, stopb, (p, s) -> p / s);
        } else if (type == 3) {
            if (stopb.length < 2) {
                throw SignalException.valueError("index 1 is out of bounds for axis 0 with size " + stopb.length);
            }
            final double[] pb = passb.clone();
            double wp0 = Fminbound.minimize(x -> bandStopObj(x, 0, pb, stopb, gpass, gstop, kind),
                    pb[0], stopb[0] - 1e-12, 1e-5, 500).x;
            double wp1 = Fminbound.minimize(x -> bandStopObj(x, 1, pb, stopb, gpass, gstop, kind),
                    stopb[1] + 1e-12, pb[1], 1e-5, 500).x;
            passb = new double[] {wp0, wp1};
            passbRef[0] = passb;
            nat = new double[stopb.length];
            for (int i = 0; i < stopb.length; i++) {
                nat[i] = (stopb[i] * (passb[0] - passb[1])) / (stopb[i] * stopb[i] - passb[0] * passb[1]);
            }
        } else if (type == 4) {
            nat = new double[stopb.length];
            for (int i = 0; i < stopb.length; i++) {
                nat[i] = (stopb[i] * stopb[i] - passb[0] * passb[1]) / (stopb[i] * (passb[0] - passb[1]));
            }
        } else {
            throw SignalException.valueError("should not happen: filter_type =" + type + ".");
        }
        double[] abs = new double[nat.length];
        for (int i = 0; i < nat.length; i++) {
            abs[i] = Math.abs(nat[i]);
        }
        return npMin(abs);
    }

    /** scipy's _postprocess_wn: back from pre-warped frequencies, then to fs units. */
    private static double[] postprocess(double[] w, boolean analog, Double fs) {
        double[] out = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            out[i] = analog ? w[i] : Math.atan(w[i]) * 2.0 / Math.PI;
            if (fs != null) {
                out[i] = out[i] * fs / 2;
            }
        }
        return out;
    }

    /** numpy.sort(numpy.abs([a, b])): ascending, NaN last. */
    private static double[] sortAbs(double a, double b) {
        double x = Math.abs(a);
        double y = Math.abs(b);
        if (Double.isNaN(x)) {
            return new double[] {y, x};
        }
        return Double.isNaN(y) || x <= y ? new double[] {x, y} : new double[] {y, x};
    }

    // The four functions

    public static Result buttord(double[] wpIn, double[] wsIn, double gpass, double gstop, boolean analog, Double fs) {
        double[] wp = wpIn.clone();
        double[] ws = wsIn.clone();
        validateGpassGstop(gpass, gstop);
        int type = filterType(wp, ws, analog, fs);
        double[][] passb = {preWarp(wp, analog)};
        double[] stopb = preWarp(ws, analog);
        double nat = findNatFreq(stopb, passb, gpass, gstop, type, Kind.BUTTER);
        double gStop = Math.pow(10, 0.1 * Math.abs(gstop));
        double gPass = Math.pow(10, 0.1 * Math.abs(gpass));
        long ord = pyCeilInt(pyDiv(pyLog10(pyDiv(gStop - 1.0, gPass - 1.0)), 2 * pyLog10(nat)));
        double w0 = ord == 0 ? 1.0 : Math.pow(gPass - 1.0, -1.0 / (2.0 * ord));
        double[] p = passb[0];
        double[] wn;
        if (type == 1) {
            wn = new double[p.length];
            for (int i = 0; i < p.length; i++) {
                wn[i] = w0 * p[i];
            }
        } else if (type == 2) {
            wn = new double[p.length];
            for (int i = 0; i < p.length; i++) {
                wn[i] = p[i] / w0;
            }
        } else if (type == 3) {
            double d = p[1] - p[0];
            double discr = Math.sqrt(d * d + 4 * (w0 * w0) * p[0] * p[1]);
            double wn0 = (d + discr) / (2 * w0);
            double wn1 = (d - discr) / (2 * w0);
            wn = sortAbs(wn0, wn1);
        } else {
            double d = p[1] - p[0];
            double[] vals = new double[2];
            double[] signs = {-w0, w0};
            for (int i = 0; i < 2; i++) {
                double w = signs[i];
                vals[i] = -w * d / 2.0 + Math.sqrt((w * w) / 4.0 * (d * d) + p[0] * p[1]);
            }
            wn = sortAbs(vals[0], vals[1]);
        }
        return new Result(ord, postprocess(wn, analog, fs));
    }

    public static Result cheb1ord(double[] wpIn, double[] wsIn, double gpass, double gstop, boolean analog,
                                  Double fs) {
        double[] wp = wpIn.clone();
        double[] ws = wsIn.clone();
        validateGpassGstop(gpass, gstop);
        int type = filterType(wp, ws, analog, fs);
        double[][] passb = {preWarp(wp, analog)};
        double[] stopb = preWarp(ws, analog);
        double nat = findNatFreq(stopb, passb, gpass, gstop, type, Kind.CHEBY);
        long ord = chebOrder(gpass, gstop, nat);
        return new Result(ord, postprocess(passb[0], analog, fs));
    }

    private static double vPassStop(double gpass, double gstop) {
        double gStop = Math.pow(10, 0.1 * Math.abs(gstop));
        double gPass = Math.pow(10, 0.1 * Math.abs(gpass));
        return pyAcosh(pySqrt(pyDiv(gStop - 1.0, gPass - 1.0)));
    }

    private static long chebOrder(double gpass, double gstop, double nat) {
        return pyCeilInt(vPassStop(gpass, gstop) / Special.acosh(nat));
    }

    public static Result cheb2ord(double[] wpIn, double[] wsIn, double gpass, double gstop, boolean analog,
                                  Double fs) {
        double[] wp = wpIn.clone();
        double[] ws = wsIn.clone();
        validateGpassGstop(gpass, gstop);
        int type = filterType(wp, ws, analog, fs);
        double[][] passb = {preWarp(wp, analog)};
        double[] stopb = preWarp(ws, analog);
        double nat = findNatFreq(stopb, passb, gpass, gstop, type, Kind.CHEBY);
        double v = vPassStop(gpass, gstop);
        long ord = pyCeilInt(v / Special.acosh(nat));
        double newFreq = pyCosh(pyDiv(1.0, ord) * v);
        newFreq = 1.0 / newFreq;
        double[] p = passb[0];
        double[] out;
        if (type == 1) {
            out = new double[p.length];
            for (int i = 0; i < p.length; i++) {
                out[i] = p[i] / newFreq;
            }
        } else if (type == 2) {
            out = new double[p.length];
            for (int i = 0; i < p.length; i++) {
                out[i] = p[i] * newFreq;
            }
        } else if (type == 3) {
            double d = p[1] - p[0];
            // passb holds numpy values: their divisions give inf rather than raising; math.sqrt still raises.
            double nat0 = newFreq / 2.0 * (p[0] - p[1])
                    + pySqrt((newFreq * newFreq) * (d * d) / 4.0 + p[1] * p[0]);
            double nat1 = p[1] * p[0] / nat0;
            out = new double[] {nat0, nat1};
        } else {
            double d = p[1] - p[0];
            double nat0 = pyDiv(1.0, 2.0 * newFreq) * (p[0] - p[1])
                    + pySqrt(d * d / (4.0 * (newFreq * newFreq)) + p[1] * p[0]);
            double nat1 = p[0] * p[1] / nat0;
            out = new double[] {nat0, nat1};
        }
        return new Result(ord, postprocess(out, analog, fs));
    }

    public static Result ellipord(double[] wpIn, double[] wsIn, double gpass, double gstop, boolean analog,
                                  Double fs) {
        double[] wp = wpIn.clone();
        double[] ws = wsIn.clone();
        validateGpassGstop(gpass, gstop);
        int type = filterType(wp, ws, analog, fs);
        double[][] passb = {preWarp(wp, analog)};
        double[] stopb = preWarp(ws, analog);
        double nat = findNatFreq(stopb, passb, gpass, gstop, type, Kind.ELLIP);
        double arg1Sq = pyDiv(IirDesign.pow10m1(0.1 * gpass), IirDesign.pow10m1(0.1 * gstop));
        double arg0 = 1.0 / nat;
        double a0 = arg0 * arg0;
        double d0k = Special.ellipk(a0);
        double d0km1 = Special.ellipkm1(a0);
        double d1k = Special.ellipk(arg1Sq);
        double d1km1 = Special.ellipkm1(arg1Sq);
        long ord = pyCeilInt(d0k * d1km1 / (d0km1 * d1k));
        return new Result(ord, postprocess(passb[0], analog, fs));
    }
}
