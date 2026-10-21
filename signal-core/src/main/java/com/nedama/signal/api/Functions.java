package com.nedama.signal.api;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The single entry point for every function, used by the Ignition script module and by the test driver.
 *
 * <p>Argument values: null (None), Boolean, Long or BigInteger (int), Double (float), String, and List for a
 * sequence (Python list or tuple). A sequence whose elements are all floats may also arrive as a double[].
 *
 * <p>Result values: the same types, plus double[] and long[] for arrays, {@link com.nedama.signal.Complex} and
 * Complex[] for complex numbers and arrays, {@link Tuple} for tuples, List of double[] for 2-D arrays (rows), and
 * Map&lt;String, Object&gt; for dicts. Invalid arguments raise {@link com.nedama.signal.SignalException}.
 */
public final class Functions {

    private static final Map<String, Registry.Definition> DEFINITIONS = build();

    private Functions() {
    }

    private static Map<String, Registry.Definition> build() {
        Registry registry = new Registry();
        DetrendFunctions.register(registry);
        WindowFunctions.register(registry);
        FilterFunctions.register(registry);
        OrderFunctions.register(registry);
        FilteringFunctions.register(registry);
        SmoothingFunctions.register(registry);
        PeakFunctions.register(registry);
        SpectralFunctions.register(registry);
        ResamplingFunctions.register(registry);
        return registry.definitions();
    }

    /** Calls a function by its scipy name. */
    public static Object call(String name, Object[] positional, Map<String, Object> keywords) {
        Registry.Definition d = DEFINITIONS.get(name);
        if (d == null) {
            throw new IllegalArgumentException("Unknown function: " + name);
        }
        return d.handler.call(d.signature.bind(positional, keywords));
    }

    /** Every function's signature, by name, in registration order. */
    public static Map<String, Signature> signatures() {
        Map<String, Signature> out = new LinkedHashMap<>();
        for (Map.Entry<String, Registry.Definition> e : DEFINITIONS.entrySet()) {
            out.put(e.getKey(), e.getValue().signature);
        }
        return Collections.unmodifiableMap(out);
    }
}
