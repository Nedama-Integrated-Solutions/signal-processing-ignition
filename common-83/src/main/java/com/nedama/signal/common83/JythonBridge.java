package com.nedama.signal.common83;

import com.nedama.signal.Complex;
import com.nedama.signal.SignalException;
import com.nedama.signal.api.Functions;
import com.nedama.signal.api.Tuple;
import org.python.core.Py;
import org.python.core.PyArray;
import org.python.core.PyBoolean;
import org.python.core.PyComplex;
import org.python.core.PyDictionary;
import org.python.core.PyFloat;
import org.python.core.PyInteger;
import org.python.core.PyList;
import org.python.core.PyLong;
import org.python.core.PyObject;
import org.python.core.PySequence;
import org.python.core.PyString;
import org.python.core.PyTuple;

import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Connects Jython calls to the core. Converts arguments to the core's plain Java values, converts results back
 * to Python values, and rethrows the core's errors as the ValueError or TypeError scipy would raise.
 *
 * <p>A non-empty sequence whose elements are all floats becomes a double[] (the fast path for signals); the
 * test driver applies the same rule, so both reach the core with the same value types.
 *
 * <p>Identical in common-81 and common-83 apart from the package line.
 */
final class JythonBridge {

    private JythonBridge() {
    }

    /** Runs a core function. Jython passes keyword values at the end of {@code args}, named by {@code keywords}. */
    static PyObject call(String function, PyObject[] args, String[] keywords) {
        int nKeywords = keywords == null ? 0 : keywords.length;
        int nPositional = args.length - nKeywords;
        try {
            Object[] positional = new Object[nPositional];
            for (int i = 0; i < nPositional; i++) {
                positional[i] = toJava(args[i]);
            }
            Map<String, Object> named = new LinkedHashMap<>();
            for (int i = 0; i < nKeywords; i++) {
                named.put(keywords[i], toJava(args[nPositional + i]));
            }
            return toPy(Functions.call(function, positional, named));
        } catch (SignalException e) {
            throw e.kind() == SignalException.Kind.TYPE ? Py.TypeError(e.getMessage()) : Py.ValueError(e.getMessage());
        }
    }

    static Object toJava(PyObject o) {
        if (o == null || o == Py.None) {
            return null;
        }
        if (o instanceof PyBoolean) {
            return o.__nonzero__();
        }
        if (o instanceof PyInteger) {
            return (long) ((PyInteger) o).getValue();
        }
        if (o instanceof PyLong) {
            return integer(((PyLong) o).getValue());
        }
        if (o instanceof PyFloat) {
            return ((PyFloat) o).getValue();
        }
        if (o instanceof PyString) {
            return ((PyString) o).getString();
        }
        if (o instanceof PyComplex) {
            throw SignalException.typeError("complex numbers are not supported; pass real values");
        }
        if (o instanceof PyDictionary) {
            throw SignalException.typeError("a dict is not a valid argument here");
        }
        if (o instanceof PyArray) {
            return fromJava(((PyArray) o).getArray());
        }
        if (o instanceof PyList || o instanceof PyTuple || o instanceof PySequence) {
            return sequence(o);
        }
        Object j = o.__tojava__(Object.class);
        if (j != Py.NoConversion && !(j instanceof PyObject)) {
            return fromJava(j);
        }
        throw SignalException.typeError("unsupported argument type: " + o.getType().getName());
    }

    private static Object sequence(PyObject seq) {
        int n = seq.__len__();
        PyObject[] items = new PyObject[n];
        boolean allFloats = n > 0;
        for (int i = 0; i < n; i++) {
            items[i] = seq.__getitem__(i);
            if (!(items[i] instanceof PyFloat)) {
                allFloats = false;
            }
        }
        if (allFloats) {
            double[] d = new double[n];
            for (int i = 0; i < n; i++) {
                d[i] = ((PyFloat) items[i]).getValue();
            }
            return d;
        }
        List<Object> out = new ArrayList<>(n);
        for (PyObject item : items) {
            out.add(toJava(item));
        }
        return out;
    }

    /** Java values handed in from scripts: boxed numbers, strings, arrays such as double[], and collections. */
    private static Object fromJava(Object j) {
        if (j == null) {
            return null;
        }
        if (j instanceof PyObject) {
            return toJava((PyObject) j);
        }
        if (j instanceof Boolean) {
            return j;
        }
        if (j instanceof Long || j instanceof Integer || j instanceof Short || j instanceof Byte) {
            return ((Number) j).longValue();
        }
        if (j instanceof BigInteger) {
            return integer((BigInteger) j);
        }
        if (j instanceof Double || j instanceof Float || j instanceof BigDecimal) {
            return ((Number) j).doubleValue();
        }
        if (j instanceof CharSequence) {
            return j.toString();
        }
        if (j.getClass().isArray()) {
            int n = Array.getLength(j);
            List<Object> out = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                out.add(fromJava(Array.get(j, i)));
            }
            return floatsToArray(out);
        }
        if (j instanceof Collection) {
            List<Object> out = new ArrayList<>(((Collection<?>) j).size());
            for (Object e : (Collection<?>) j) {
                out.add(fromJava(e));
            }
            return floatsToArray(out);
        }
        throw SignalException.typeError("unsupported argument type: " + j.getClass().getName());
    }

    private static Object floatsToArray(List<Object> list) {
        if (list.isEmpty()) {
            return list;
        }
        double[] d = new double[list.size()];
        for (int i = 0; i < d.length; i++) {
            Object e = list.get(i);
            if (!(e instanceof Double)) {
                return list;
            }
            d[i] = (Double) e;
        }
        return d;
    }

    private static Object integer(BigInteger b) {
        return b.bitLength() < 64 ? (Object) b.longValue() : b;
    }

    static PyObject toPy(Object v) {
        if (v == null) {
            return Py.None;
        }
        if (v instanceof double[]) {
            double[] a = (double[]) v;
            PyObject[] items = new PyObject[a.length];
            for (int i = 0; i < a.length; i++) {
                items[i] = new PyFloat(a[i]);
            }
            return new PyList(items);
        }
        if (v instanceof long[]) {
            long[] a = (long[]) v;
            PyObject[] items = new PyObject[a.length];
            for (int i = 0; i < a.length; i++) {
                items[i] = Py.newInteger(a[i]);
            }
            return new PyList(items);
        }
        if (v instanceof Complex) {
            return new PyComplex(((Complex) v).re, ((Complex) v).im);
        }
        if (v instanceof Complex[]) {
            Complex[] a = (Complex[]) v;
            PyObject[] items = new PyObject[a.length];
            for (int i = 0; i < a.length; i++) {
                items[i] = new PyComplex(a[i].re, a[i].im);
            }
            return new PyList(items);
        }
        if (v instanceof Double) {
            return new PyFloat((Double) v);
        }
        if (v instanceof Long) {
            return Py.newInteger((Long) v);
        }
        if (v instanceof Boolean) {
            return Py.newBoolean((Boolean) v);
        }
        if (v instanceof String) {
            return Py.newUnicode((String) v);
        }
        if (v instanceof Tuple) {
            return new PyTuple(items((Tuple) v));
        }
        if (v instanceof List) {
            return new PyList(items((List<?>) v));
        }
        if (v instanceof Map) {
            PyDictionary d = new PyDictionary();
            for (Map.Entry<?, ?> e : ((Map<?, ?>) v).entrySet()) {
                d.__setitem__(Py.newString(String.valueOf(e.getKey())), toPy(e.getValue()));
            }
            return d;
        }
        throw new IllegalStateException("Unexpected result type " + v.getClass().getName());
    }

    private static PyObject[] items(List<?> list) {
        PyObject[] items = new PyObject[list.size()];
        for (int i = 0; i < items.length; i++) {
            items[i] = toPy(list.get(i));
        }
        return items;
    }
}
