package com.nedama.signal;

import com.nedama.signal.api.Tuple;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Just enough JSON for the test driver, so the core keeps zero dependencies even in tests. Reads what Python's
 * json module writes, including NaN, Infinity and -Infinity. Integers become Long (BigInteger when too large),
 * other numbers Double. Doubles are written with Double.toString, which Python reads back to the same value.
 * Complex numbers are written as {"__complex__": [re, im]}, which harness.py turns back into Python complex.
 */
final class Json {

    private final String s;
    private int i;

    private Json(String s) {
        this.s = s;
    }

    static Object parse(String text) {
        Json p = new Json(text);
        Object v = p.value();
        p.space();
        if (p.i != p.s.length()) {
            throw p.error("trailing characters");
        }
        return v;
    }

    private Object value() {
        space();
        if (i >= s.length()) {
            throw error("unexpected end");
        }
        char c = s.charAt(i);
        switch (c) {
            case '{':
                return object();
            case '[':
                return array();
            case '"':
                return string();
            case 't':
                return literal("true", Boolean.TRUE);
            case 'f':
                return literal("false", Boolean.FALSE);
            case 'n':
                return literal("null", null);
            case 'N':
                return literal("NaN", Double.NaN);
            case 'I':
                return literal("Infinity", Double.POSITIVE_INFINITY);
            default:
                if (c == '-' && s.startsWith("-Infinity", i)) {
                    return literal("-Infinity", Double.NEGATIVE_INFINITY);
                }
                return number();
        }
    }

    private Map<String, Object> object() {
        Map<String, Object> m = new LinkedHashMap<>();
        i++;
        space();
        if (peek() == '}') {
            i++;
            return m;
        }
        while (true) {
            space();
            String key = string();
            space();
            expect(':');
            m.put(key, value());
            space();
            if (peek() == ',') {
                i++;
            } else {
                expect('}');
                return m;
            }
        }
    }

    private List<Object> array() {
        List<Object> a = new ArrayList<>();
        i++;
        space();
        if (peek() == ']') {
            i++;
            return a;
        }
        while (true) {
            a.add(value());
            space();
            if (peek() == ',') {
                i++;
            } else {
                expect(']');
                return a;
            }
        }
    }

    private String string() {
        expect('"');
        StringBuilder b = new StringBuilder();
        while (true) {
            if (i >= s.length()) {
                throw error("unterminated string");
            }
            char c = s.charAt(i++);
            if (c == '"') {
                return b.toString();
            }
            if (c != '\\') {
                b.append(c);
                continue;
            }
            char e = s.charAt(i++);
            switch (e) {
                case 'n':
                    b.append('\n');
                    break;
                case 't':
                    b.append('\t');
                    break;
                case 'r':
                    b.append('\r');
                    break;
                case 'b':
                    b.append('\b');
                    break;
                case 'f':
                    b.append('\f');
                    break;
                case 'u':
                    b.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                    i += 4;
                    break;
                default:
                    b.append(e);
            }
        }
    }

    private Object number() {
        int start = i;
        boolean integer = true;
        while (i < s.length()) {
            char c = s.charAt(i);
            if (c >= '0' && c <= '9' || c == '-' || c == '+') {
                i++;
            } else if (c == '.' || c == 'e' || c == 'E') {
                integer = false;
                i++;
            } else {
                break;
            }
        }
        String t = s.substring(start, i);
        if (t.isEmpty()) {
            throw error("unexpected character '" + s.charAt(i) + "'");
        }
        if (!integer) {
            return Double.parseDouble(t);
        }
        BigInteger b = new BigInteger(t);
        return b.bitLength() < 64 ? (Object) b.longValue() : b;
    }

    private Object literal(String word, Object v) {
        if (!s.startsWith(word, i)) {
            throw error("bad literal");
        }
        i += word.length();
        return v;
    }

    private void space() {
        while (i < s.length() && Character.isWhitespace(s.charAt(i))) {
            i++;
        }
    }

    private char peek() {
        return i < s.length() ? s.charAt(i) : '\0';
    }

    private void expect(char c) {
        if (peek() != c) {
            throw error("expected '" + c + "'");
        }
        i++;
    }

    private IllegalArgumentException error(String what) {
        return new IllegalArgumentException("JSON " + what + " at position " + i);
    }

    // Writing

    static String write(Object v) {
        StringBuilder b = new StringBuilder();
        write(b, v);
        return b.toString();
    }

    private static void write(StringBuilder b, Object v) {
        if (v == null) {
            b.append("null");
        } else if (v instanceof Double) {
            number(b, (Double) v);
        } else if (v instanceof Long || v instanceof Integer || v instanceof BigInteger) {
            b.append(v);
        } else if (v instanceof Boolean) {
            b.append(v);
        } else if (v instanceof String) {
            quote(b, (String) v);
        } else if (v instanceof double[]) {
            double[] a = (double[]) v;
            b.append('[');
            for (int k = 0; k < a.length; k++) {
                if (k > 0) {
                    b.append(',');
                }
                number(b, a[k]);
            }
            b.append(']');
        } else if (v instanceof long[]) {
            long[] a = (long[]) v;
            b.append('[');
            for (int k = 0; k < a.length; k++) {
                if (k > 0) {
                    b.append(',');
                }
                b.append(a[k]);
            }
            b.append(']');
        } else if (v instanceof Complex) {
            Complex c = (Complex) v;
            b.append("{\"__complex__\":[");
            number(b, c.re);
            b.append(',');
            number(b, c.im);
            b.append("]}");
        } else if (v instanceof Complex[]) {
            Complex[] a = (Complex[]) v;
            b.append('[');
            for (int k = 0; k < a.length; k++) {
                if (k > 0) {
                    b.append(',');
                }
                write(b, a[k]);
            }
            b.append(']');
        } else if (v instanceof Tuple || v instanceof List) {
            b.append('[');
            boolean first = true;
            for (Object o : (List<?>) v) {
                if (!first) {
                    b.append(',');
                }
                first = false;
                write(b, o);
            }
            b.append(']');
        } else if (v instanceof Map) {
            b.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : ((Map<?, ?>) v).entrySet()) {
                if (!first) {
                    b.append(',');
                }
                first = false;
                quote(b, String.valueOf(e.getKey()));
                b.append(':');
                write(b, e.getValue());
            }
            b.append('}');
        } else {
            throw new IllegalArgumentException("Cannot write " + v.getClass().getName() + " as JSON");
        }
    }

    private static void number(StringBuilder b, double d) {
        if (Double.isNaN(d)) {
            b.append("NaN");
        } else if (Double.isInfinite(d)) {
            b.append(d > 0 ? "Infinity" : "-Infinity");
        } else {
            b.append(Double.toString(d));
        }
    }

    private static void quote(StringBuilder b, String str) {
        b.append('"');
        for (int k = 0; k < str.length(); k++) {
            char c = str.charAt(k);
            if (c == '"' || c == '\\') {
                b.append('\\').append(c);
            } else if (c < 0x20) {
                b.append(String.format("\\u%04x", (int) c));
            } else {
                b.append(c);
            }
        }
        b.append('"');
    }
}
