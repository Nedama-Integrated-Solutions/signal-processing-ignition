package com.nedama.signal.api;

import com.nedama.signal.Detrend;

/** detrend. */
final class DetrendFunctions {

    private DetrendFunctions() {
    }

    static void register(Registry r) {
        r.define(Signature.of("detrend",
                        Param.required("data"),
                        Param.optional("axis", -1L),
                        Param.optional("type", "linear"),
                        Param.optional("bp", 0L),
                        Param.optional("overwrite_data", Boolean.FALSE)),
                DetrendFunctions::detrend);
    }

    private static Object detrend(Args a) {
        String type = a.choice("type", "Trend type must be 'linear' or 'constant'.", "linear", "l", "constant", "c");
        double[] data = a.array("data");
        a.axis("axis");
        // overwrite_data is accepted for compatibility and has no effect: the input is never modified.
        if (type.startsWith("c")) {
            return Detrend.constant(data);
        }
        long[] bp = a.integers("bp");
        // An empty breakpoint list means no breakpoints. scipy raises a TypeError for it; see the test reference.
        return Detrend.linear(data, bp);
    }
}
