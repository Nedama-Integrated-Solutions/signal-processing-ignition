package com.nedama.signal;

/**
 * Raised for invalid arguments. The kind says which Python exception scipy would raise for the same
 * mistake, and the Ignition layer rethrows it as that exception, so scripts can catch
 * {@code ValueError} and {@code TypeError} exactly as they would with scipy.
 */
public class SignalException extends IllegalArgumentException {

    private static final long serialVersionUID = 1L;

    /** The Python exception this maps to. */
    public enum Kind {
        /** A value of the right type that is not allowed, for example a negative order. */
        VALUE,
        /** An argument of the wrong type, or a call that does not match the function's signature. */
        TYPE
    }

    private final Kind kind;

    public SignalException(Kind kind, String message) {
        super(message);
        this.kind = kind;
    }

    public Kind kind() {
        return kind;
    }

    public static SignalException valueError(String message) {
        return new SignalException(Kind.VALUE, message);
    }

    public static SignalException typeError(String message) {
        return new SignalException(Kind.TYPE, message);
    }
}
