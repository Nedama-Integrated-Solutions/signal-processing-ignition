package com.nedama.signal.api;

import com.nedama.signal.OrderSelection;

/**
 * buttord, cheb1ord, cheb2ord and ellipord, with scipy 1.18.1's arguments. Each returns (N, Wn): Wn is a number
 * for low-pass and high-pass, a list of two for band filters, ready to pass to the matching design function.
 */
final class OrderFunctions {

    private OrderFunctions() {
    }

    @FunctionalInterface
    private interface Method {
        OrderSelection.Result apply(double[] wp, double[] ws, double gpass, double gstop, boolean analog, Double fs);
    }

    static void register(Registry r) {
        define(r, "buttord", OrderSelection::buttord);
        define(r, "cheb1ord", OrderSelection::cheb1ord);
        define(r, "cheb2ord", OrderSelection::cheb2ord);
        define(r, "ellipord", OrderSelection::ellipord);
    }

    private static void define(Registry r, String name, Method m) {
        r.define(Signature.of(name, Param.required("wp"), Param.required("ws"), Param.required("gpass"),
                        Param.required("gstop"), Param.optional("analog", Boolean.FALSE), Param.optional("fs", null)),
                a -> {
                    double[] wp = a.reals("wp");
                    double[] ws = a.reals("ws");
                    double gpass = a.real("gpass");
                    double gstop = a.real("gstop");
                    Double fs = a.isNone("fs") ? null : a.real("fs");
                    OrderSelection.Result res = m.apply(wp, ws, gpass, gstop, a.flag("analog"), fs);
                    Object wn = res.wn.length == 1 ? (Object) res.wn[0] : res.wn;
                    return Tuple.of(res.n, wn);
                });
    }
}
