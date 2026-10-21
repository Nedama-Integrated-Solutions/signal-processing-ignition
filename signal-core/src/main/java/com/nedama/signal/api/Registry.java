package com.nedama.signal.api;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/** Collects the function definitions. Each group class adds its functions in a {@code register} method. */
final class Registry {

    /** A signature and the code that runs it. */
    static final class Definition {
        final Signature signature;
        final Handler handler;

        Definition(Signature signature, Handler handler) {
            this.signature = signature;
            this.handler = handler;
        }
    }

    private final Map<String, Definition> definitions = new LinkedHashMap<>();

    void define(Signature signature, Handler handler) {
        if (definitions.containsKey(signature.function())) {
            throw new IllegalStateException("Function defined twice: " + signature.function());
        }
        definitions.put(signature.function(), new Definition(signature, handler));
    }

    Map<String, Definition> definitions() {
        return Collections.unmodifiableMap(definitions);
    }
}
