package com.nedama.signal;

import com.nedama.signal.api.Functions;
import com.nedama.signal.api.Signature;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Line-based test driver for src/test/python/random_check.py. Lives in test sources, so it is never packaged
 * into the module.
 *
 * <p>Input, one JSON object per line: {@code {"id": 1, "fn": "detrend", "args": [...], "kwargs": {...}}}.
 * <br>Output, one line per call: {@code {"id": 1, "ok": result, "ns": time}}, {@code {"id": 1, "err": "..."}}
 * for a SignalException, or {@code {"id": 1, "crash": "..."}} for anything else (always a bug).
 *
 * <p>The special call {@code {"id": 0, "fn": "__signatures__"}} returns every function's parameter names. Names
 * starting with one underscore, such as {@code _fft}, call the test-only entry points in {@link Internals}.
 *
 * <p>Arguments go through the same normalisation as the Ignition bridge: a non-empty list whose elements are
 * all floats becomes a double[], so both paths reach the core with the same value types.
 */
public final class SignalCli {

    private SignalCli() {
    }

    public static void main(String[] args) throws IOException {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        PrintStream out = new PrintStream(new BufferedOutputStream(new FileOutputStream(FileDescriptor.out), 1 << 16),
                false, "UTF-8");
        String line;
        while ((line = in.readLine()) != null) {
            line = line.trim();
            if (!line.isEmpty()) {
                out.println(Json.write(handle(line)));
            }
        }
        out.flush();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> handle(String line) {
        Map<String, Object> request = (Map<String, Object>) Json.parse(line);
        Map<String, Object> reply = new LinkedHashMap<>();
        reply.put("id", request.get("id"));
        String fn = (String) request.get("fn");
        try {
            if ("__signatures__".equals(fn)) {
                reply.put("ok", signatures());
                return reply;
            }
            List<Object> positional = (List<Object>) request.getOrDefault("args", new ArrayList<>());
            Map<String, Object> keywords = (Map<String, Object>) request.getOrDefault("kwargs", new LinkedHashMap<>());
            Object[] pos = new Object[positional.size()];
            for (int k = 0; k < pos.length; k++) {
                pos[k] = normalise(positional.get(k));
            }
            Map<String, Object> kw = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : keywords.entrySet()) {
                kw.put(e.getKey(), normalise(e.getValue()));
            }
            long t0 = System.nanoTime();
            Object result = fn.startsWith("_") ? Internals.call(fn, pos, kw) : Functions.call(fn, pos, kw);
            long elapsed = System.nanoTime() - t0;
            reply.put("ok", result);
            reply.put("ns", elapsed);
        } catch (SignalException e) {
            reply.put("err", (e.kind() == SignalException.Kind.TYPE ? "TypeError: " : "ValueError: ") + e.getMessage());
        } catch (Throwable t) {
            StringWriter w = new StringWriter();
            t.printStackTrace(new PrintWriter(w));
            reply.put("crash", w.toString());
        }
        return reply;
    }

    private static Object normalise(Object v) {
        if (!(v instanceof List)) {
            return v;
        }
        List<?> list = (List<?>) v;
        boolean allFloats = !list.isEmpty();
        for (Object o : list) {
            if (!(o instanceof Double)) {
                allFloats = false;
                break;
            }
        }
        if (allFloats) {
            double[] a = new double[list.size()];
            for (int k = 0; k < a.length; k++) {
                a[k] = (Double) list.get(k);
            }
            return a;
        }
        List<Object> out = new ArrayList<>(list.size());
        for (Object o : list) {
            out.add(normalise(o));
        }
        return out;
    }

    private static Map<String, Object> signatures() {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<String, Signature> e : Functions.signatures().entrySet()) {
            Map<String, Object> s = new LinkedHashMap<>();
            s.put("params", new ArrayList<Object>(e.getValue().parameterNames()));
            s.put("kwonly", new ArrayList<Object>(e.getValue().keywordOnlyNames()));
            out.put(e.getKey(), s);
        }
        return out;
    }
}
