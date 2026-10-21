package com.nedama.signal.api;

import com.nedama.signal.SignalException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A scipy function signature, with Python's rules for binding positional and keyword arguments. Binding errors
 * are TypeErrors with Python's wording, so a mistyped keyword fails the same way it would with scipy.
 */
public final class Signature {

    private final String function;
    private final Param[] params;
    private final int positionalCount;
    private final int requiredPositionalCount;

    private Signature(String function, Param[] params) {
        this.function = function;
        this.params = params.clone();
        int positional = 0;
        int required = 0;
        boolean seenKeywordOnly = false;
        boolean seenOptional = false;
        for (Param p : params) {
            if (p.isKeywordOnly()) {
                seenKeywordOnly = true;
                continue;
            }
            if (seenKeywordOnly) {
                throw new IllegalStateException(function + ": positional parameter after a keyword-only one");
            }
            if (p.isRequired()) {
                if (seenOptional) {
                    throw new IllegalStateException(function + ": required parameter after an optional one");
                }
                required++;
            } else {
                seenOptional = true;
            }
            positional++;
        }
        this.positionalCount = positional;
        this.requiredPositionalCount = required;
    }

    public static Signature of(String function, Param... params) {
        return new Signature(function, params);
    }

    public String function() {
        return function;
    }

    /** Parameter names in signature order. */
    public List<String> parameterNames() {
        List<String> names = new ArrayList<>(params.length);
        for (Param p : params) {
            names.add(p.name());
        }
        return Collections.unmodifiableList(names);
    }

    /** Names of the keyword-only parameters, in signature order. */
    public List<String> keywordOnlyNames() {
        List<String> names = new ArrayList<>();
        for (Param p : params) {
            if (p.isKeywordOnly()) {
                names.add(p.name());
            }
        }
        return Collections.unmodifiableList(names);
    }

    /** Binds a call's arguments to this signature, filling in defaults. */
    public Args bind(Object[] positional, Map<String, Object> keywords) {
        if (positional.length > positionalCount) {
            throw SignalException.typeError(tooManyPositional(positional.length));
        }
        Map<String, Object> values = new LinkedHashMap<>();
        for (int i = 0; i < positional.length; i++) {
            values.put(params[i].name(), positional[i]);
        }
        for (Map.Entry<String, Object> e : keywords.entrySet()) {
            String name = e.getKey();
            if (find(name) == null) {
                throw SignalException.typeError(function + "() got an unexpected keyword argument '" + name + "'");
            }
            if (values.containsKey(name)) {
                throw SignalException.typeError(function + "() got multiple values for argument '" + name + "'");
            }
            values.put(name, e.getValue());
        }
        List<String> missingPositional = new ArrayList<>();
        List<String> missingKeywordOnly = new ArrayList<>();
        Map<String, Object> bound = new LinkedHashMap<>();
        for (Param p : params) {
            if (values.containsKey(p.name())) {
                bound.put(p.name(), values.get(p.name()));
            } else if (p.isRequired()) {
                (p.isKeywordOnly() ? missingKeywordOnly : missingPositional).add(p.name());
            } else {
                bound.put(p.name(), p.defaultValue());
            }
        }
        if (!missingPositional.isEmpty()) {
            throw SignalException.typeError(missing(missingPositional, "positional"));
        }
        if (!missingKeywordOnly.isEmpty()) {
            throw SignalException.typeError(missing(missingKeywordOnly, "keyword-only"));
        }
        return new Args(function, bound);
    }

    private Param find(String name) {
        for (Param p : params) {
            if (p.name().equals(name)) {
                return p;
            }
        }
        return null;
    }

    private String tooManyPositional(int given) {
        String takes;
        if (requiredPositionalCount == positionalCount) {
            takes = positionalCount + (positionalCount == 1 ? " positional argument" : " positional arguments");
        } else {
            takes = "from " + requiredPositionalCount + " to " + positionalCount + " positional arguments";
        }
        return function + "() takes " + takes + " but " + given + (given == 1 ? " was" : " were") + " given";
    }

    private String missing(List<String> names, String kind) {
        StringBuilder list = new StringBuilder();
        for (int i = 0; i < names.size(); i++) {
            if (i > 0) {
                list.append(i == names.size() - 1 ? (names.size() > 2 ? ", and " : " and ") : ", ");
            }
            list.append('\'').append(names.get(i)).append('\'');
        }
        return function + "() missing " + names.size() + " required " + kind + " argument"
                + (names.size() == 1 ? "" : "s") + ": " + list;
    }
}
