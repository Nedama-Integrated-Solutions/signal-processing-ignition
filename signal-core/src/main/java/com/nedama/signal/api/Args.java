package com.nedama.signal.api;

import com.nedama.signal.SignalException;

import java.math.BigInteger;
import java.util.List;
import java.util.Map;

/**
 * Arguments bound to a signature. Values are converted only when a getter asks for them, so an argument the
 * call never uses (like {@code bp} for a constant detrend) is never checked, just as in scipy.
 *
 * <p>Rules shared by every function: signals are 1-D sequences of real numbers; None, NaN and infinity are
 * rejected with a ValueError; booleans count as the integers 1 and 0, as in Python.
 */
public final class Args {

    private final String function;
    private final Map<String, Object> values;

    Args(String function, Map<String, Object> values) {
        this.function = function;
        this.values = values;
    }

    /** The value as passed, or the default. */
    public Object raw(String name) {
        if (!values.containsKey(name)) {
            throw new IllegalStateException(function + "() has no parameter '" + name + "'");
        }
        return values.get(name);
    }

    public boolean isNone(String name) {
        return raw(name) == null;
    }

    /** A 1-D sequence of finite real numbers, as a new array the caller may modify. */
    public double[] array(String name) {
        Object v = raw(name);
        double[] out;
        if (v instanceof double[]) {
            out = ((double[]) v).clone();
        } else if (v instanceof List) {
            List<?> list = (List<?>) v;
            out = new double[list.size()];
            for (int i = 0; i < out.length; i++) {
                out[i] = element(name, list.get(i), i);
            }
        } else {
            throw typeError(name + " must be a 1-D sequence of numbers, not " + typeName(v));
        }
        for (int i = 0; i < out.length; i++) {
            if (!Double.isFinite(out[i])) {
                throw nonFinite(name, i);
            }
        }
        return out;
    }

    /** A real number or a 1-D sequence of them, as an array (a number gives one element). */
    public double[] reals(String name) {
        Object v = raw(name);
        if (v instanceof Double || v instanceof Long || v instanceof Boolean || v instanceof BigInteger) {
            return new double[] {real(name)};
        }
        return array(name);
    }

    /**
     * A filter order: a whole number, given as an int or as a float with no fractional part (scipy accepts 4.0).
     * Negative or fractional values are a ValueError, as scipy's abs(int(N)) != N test makes them.
     */
    public int order(String name, int max) {
        Object v = raw(name);
        double d;
        if (v instanceof Long || v instanceof Boolean || v instanceof BigInteger) {
            d = number(v);
        } else if (v instanceof Double) {
            d = (Double) v;
        } else {
            throw typeError(name + " must be an integer, not " + typeName(v));
        }
        if (!Double.isFinite(d) || d != Math.rint(d) || d < 0) {
            throw SignalException.valueError("Filter order must be a nonnegative integer");
        }
        if (d > max) {
            throw valueError(name + " is too large (at most " + max + ")");
        }
        return (int) d;
    }

    /** A string, lower-cased, as scipy's x.lower(); anything else is a TypeError. */
    public String lower(String name) {
        Object v = raw(name);
        if (!(v instanceof String)) {
            throw typeError(name + " must be a string, not " + typeName(v));
        }
        return ((String) v).toLowerCase(java.util.Locale.ROOT);
    }

    /**
     * Rows of a fixed width, such as second-order sections (6 numbers each). A single flat row counts as one row,
     * as numpy's atleast_2d makes it. Every value must be finite.
     */
    public double[][] rows(String name, int width) {
        return rows(name, width, true);
    }

    /**
     * Rows of a fixed width. With {@code flatIsRow}, a single flat row counts as one row (numpy's atleast_2d, as
     * sosfilt applies to sos); without it, only a list of rows is accepted (sosfilt_zi and zi need a 2-D array).
     */
    public double[][] rows(String name, int width, boolean flatIsRow) {
        Object v = raw(name);
        if (!(v instanceof List) && !(v instanceof double[])) {
            throw typeError(name + " must be a list of rows, not " + typeName(v));
        }
        List<?> list = v instanceof double[] ? null : (List<?>) v;
        boolean flat = v instanceof double[] || list.isEmpty()
                || !(list.get(0) instanceof List || list.get(0) instanceof double[]);
        if (flat && !flatIsRow) {
            throw SignalException.valueError(name + " must be a 2-D array of shape (n, " + width + ")");
        }
        Object[] items = flat ? new Object[] {v} : list.toArray();
        double[][] out = new double[items.length][];
        for (int r = 0; r < items.length; r++) {
            Object row = items[r];
            if (!(row instanceof List) && !(row instanceof double[])) {
                throw SignalException.valueError(name + " array must be shape (n_sections, " + width + ")");
            }
            out[r] = rowValues(name, row);
        }
        for (double[] row : out) {
            if (row.length != width) {
                throw SignalException.valueError(name + " array must be shape (n_sections, " + width + ")");
            }
        }
        return out;
    }

    private double[] rowValues(String name, Object row) {
        double[] out;
        if (row instanceof double[]) {
            out = ((double[]) row).clone();
        } else {
            List<?> list = (List<?>) row;
            out = new double[list.size()];
            for (int i = 0; i < out.length; i++) {
                out[i] = element(name, list.get(i), i);
            }
        }
        for (int i = 0; i < out.length; i++) {
            if (!Double.isFinite(out[i])) {
                throw nonFinite(name, i);
            }
        }
        return out;
    }

    /** A single finite real number. */
    public double real(String name) {
        Object v = raw(name);
        if (v instanceof Double || v instanceof Long || v instanceof Boolean || v instanceof BigInteger) {
            double d = number(v);
            if (!Double.isFinite(d)) {
                throw valueError(name + " must be a finite number, not " + d);
            }
            return d;
        }
        throw typeError(name + " must be a number, not " + typeName(v));
    }

    /** Python truthiness: None, False, 0, 0.0, "" and empty sequences are false; everything else is true. */
    public boolean flag(String name) {
        return truthy(raw(name));
    }

    static boolean truthy(Object v) {
        if (v == null) {
            return false;
        }
        if (v instanceof Boolean) {
            return (Boolean) v;
        }
        if (v instanceof Long) {
            return (Long) v != 0;
        }
        if (v instanceof BigInteger) {
            return ((BigInteger) v).signum() != 0;
        }
        if (v instanceof Double) {
            return (Double) v != 0.0;
        }
        if (v instanceof String) {
            return !((String) v).isEmpty();
        }
        if (v instanceof List) {
            return !((List<?>) v).isEmpty();
        }
        if (v instanceof double[]) {
            return ((double[]) v).length > 0;
        }
        if (v instanceof Map) {
            return !((Map<?, ?>) v).isEmpty();
        }
        return true;
    }

    /** A single integer. Floats are refused, as in scipy where an index or count must be an integer. */
    public long integer(String name) {
        Object v = raw(name);
        if (!isIntegerLike(v)) {
            throw typeError(name + " must be an integer, not " + typeName(v));
        }
        return toLong(name, v);
    }

    /** An integer, or a 1-D sequence of integers (an empty sequence gives an empty array). */
    public long[] integers(String name) {
        Object v = raw(name);
        if (isIntegerLike(v)) {
            return new long[] {toLong(name, v)};
        }
        if (v instanceof List) {
            List<?> list = (List<?>) v;
            long[] out = new long[list.size()];
            for (int i = 0; i < out.length; i++) {
                Object e = list.get(i);
                if (!isIntegerLike(e)) {
                    throw typeError(name + " must contain integers, found " + typeName(e) + " at index " + i);
                }
                out[i] = toLong(name, e);
            }
            return out;
        }
        throw typeError(name + " must be an integer or a sequence of integers, not " + typeName(v));
    }

    /** The axis of a 1-D signal: -1 or 0. */
    public int axis(String name) {
        long axis = integer(name);
        if (axis != -1 && axis != 0) {
            throw valueError(name + " " + axis + " is out of bounds for 1-D data (use -1 or 0)");
        }
        return (int) axis;
    }

    /** A string that must be one of {@code allowed}; anything else raises a ValueError with {@code message}. */
    public String choice(String name, String message, String... allowed) {
        Object v = raw(name);
        if (v instanceof String) {
            for (String a : allowed) {
                if (a.equals(v)) {
                    return a;
                }
            }
        }
        throw SignalException.valueError(message);
    }

    private double element(String name, Object e, int index) {
        if (e == null) {
            throw nonFinite(name, index);
        }
        if (e instanceof Double || e instanceof Long || e instanceof Boolean || e instanceof BigInteger) {
            return number(e);
        }
        if (e instanceof List || e instanceof double[]) {
            throw typeError(name + " must be 1-D: found a nested sequence at index " + index);
        }
        throw typeError(name + " must contain only numbers, found " + typeName(e) + " at index " + index);
    }

    private SignalException nonFinite(String name, int index) {
        return valueError(name + " must not contain None, NaN or infinite values (first one at index " + index
                + "). Fill or remove gaps before calling.");
    }

    private static boolean isIntegerLike(Object v) {
        return v instanceof Long || v instanceof Boolean || v instanceof BigInteger;
    }

    private long toLong(String name, Object v) {
        if (v instanceof Long) {
            return (Long) v;
        }
        if (v instanceof Boolean) {
            return (Boolean) v ? 1L : 0L;
        }
        BigInteger b = (BigInteger) v;
        if (b.bitLength() >= 64) {
            throw valueError(name + " is out of range: " + b);
        }
        return b.longValue();
    }

    private static double number(Object v) {
        if (v instanceof Boolean) {
            return (Boolean) v ? 1.0 : 0.0;
        }
        return ((Number) v).doubleValue();
    }

    /** Python's name for the type of a value, for error messages. */
    static String typeName(Object v) {
        if (v == null) {
            return "None";
        }
        if (v instanceof Boolean) {
            return "bool";
        }
        if (v instanceof Long || v instanceof BigInteger) {
            return "int";
        }
        if (v instanceof Double) {
            return "float";
        }
        if (v instanceof String) {
            return "str";
        }
        if (v instanceof Tuple) {
            return "tuple";
        }
        if (v instanceof List || v instanceof double[]) {
            return "list";
        }
        if (v instanceof Map) {
            return "dict";
        }
        return v.getClass().getSimpleName();
    }

    private SignalException typeError(String message) {
        return SignalException.typeError(function + "(): " + message);
    }

    private SignalException valueError(String message) {
        return SignalException.valueError(function + "(): " + message);
    }
}
