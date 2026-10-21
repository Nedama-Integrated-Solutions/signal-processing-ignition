package com.nedama.signal.api;

/** One parameter of a scipy function signature: its name, its default, and whether it is keyword-only. */
public final class Param {

    private final String name;
    private final boolean required;
    private final Object defaultValue;
    private final boolean keywordOnly;

    private Param(String name, boolean required, Object defaultValue, boolean keywordOnly) {
        this.name = name;
        this.required = required;
        this.defaultValue = defaultValue;
        this.keywordOnly = keywordOnly;
    }

    /** A parameter with no default. */
    public static Param required(String name) {
        return new Param(name, true, null, false);
    }

    /**
     * A parameter with a default. The default uses the same value types as arguments (see {@link Functions}):
     * null for None, Long for integers, Double for floats, String, Boolean, List for tuples.
     */
    public static Param optional(String name, Object defaultValue) {
        return new Param(name, false, defaultValue, false);
    }

    /** The same parameter, but only accepted by keyword (it comes after {@code *} in scipy's signature). */
    public Param keywordOnly() {
        return new Param(name, required, defaultValue, true);
    }

    public String name() {
        return name;
    }

    public boolean isRequired() {
        return required;
    }

    public Object defaultValue() {
        return defaultValue;
    }

    public boolean isKeywordOnly() {
        return keywordOnly;
    }
}
