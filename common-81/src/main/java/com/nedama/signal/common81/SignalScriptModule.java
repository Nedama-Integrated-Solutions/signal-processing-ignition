package com.nedama.signal.common81;

import com.inductiveautomation.ignition.common.BundleUtil;
import com.inductiveautomation.ignition.common.script.builtin.KeywordArgs;
import com.inductiveautomation.ignition.common.script.hints.ScriptFunction;
import org.python.core.PyObject;

/**
 * The {@code system.signal} scripting namespace. Every function has scipy.signal's name, parameters and defaults,
 * and accepts positional and keyword arguments as scipy does. The functions are pure calculations, so the same
 * object serves the gateway, Vision client and designer scopes (Perspective scripts run on the gateway).
 *
 * <p>Signals are 1-D lists of numbers, evenly sampled. None, NaN and infinity raise a ValueError. Results are
 * Python lists, tuples and dicts. Invalid arguments raise ValueError or TypeError, as scipy would.
 *
 * <p>The {@code @KeywordArgs} names must match the core signature exactly; random_check.py verifies this.
 */
public class SignalScriptModule {

    private static final String DOCS = "SignalScriptModule";

    static {
        BundleUtil.get().addBundle(
                SignalScriptModule.class.getSimpleName(),
                SignalScriptModule.class.getClassLoader(),
                SignalScriptModule.class.getName().replace('.', '/'));
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"data", "axis", "type", "bp", "overwrite_data"},
            types = {PyObject.class, Integer.class, String.class, PyObject.class, Boolean.class})
    public PyObject detrend(PyObject[] args, String[] keywords) {
        return JythonBridge.call("detrend", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"window", "Nx", "fftbins"},
            types = {PyObject.class, Integer.class, Boolean.class})
    public PyObject get_window(PyObject[] args, String[] keywords) {
        return JythonBridge.call("get_window", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"N", "Wn", "btype", "analog", "output", "fs"},
            types = {Integer.class, PyObject.class, String.class, Boolean.class, String.class, Double.class})
    public PyObject butter(PyObject[] args, String[] keywords) {
        return JythonBridge.call("butter", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"N", "rp", "Wn", "btype", "analog", "output", "fs"},
            types = {Integer.class, Double.class, PyObject.class, String.class, Boolean.class, String.class, Double.class})
    public PyObject cheby1(PyObject[] args, String[] keywords) {
        return JythonBridge.call("cheby1", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"N", "rs", "Wn", "btype", "analog", "output", "fs"},
            types = {Integer.class, Double.class, PyObject.class, String.class, Boolean.class, String.class, Double.class})
    public PyObject cheby2(PyObject[] args, String[] keywords) {
        return JythonBridge.call("cheby2", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"N", "rp", "rs", "Wn", "btype", "analog", "output", "fs"},
            types = {Integer.class, Double.class, Double.class, PyObject.class, String.class, Boolean.class, String.class, Double.class})
    public PyObject ellip(PyObject[] args, String[] keywords) {
        return JythonBridge.call("ellip", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"N", "Wn", "btype", "analog", "output", "norm", "fs"},
            types = {Integer.class, PyObject.class, String.class, Boolean.class, String.class, String.class, Double.class})
    public PyObject bessel(PyObject[] args, String[] keywords) {
        return JythonBridge.call("bessel", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"b", "a", "worN", "whole", "plot", "fs", "include_nyquist"},
            types = {PyObject.class, PyObject.class, PyObject.class, Boolean.class, PyObject.class, Double.class, Boolean.class})
    public PyObject freqz(PyObject[] args, String[] keywords) {
        return JythonBridge.call("freqz", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"sos", "worN", "whole", "fs"},
            types = {PyObject.class, PyObject.class, Boolean.class, Double.class})
    public PyObject freqz_sos(PyObject[] args, String[] keywords) {
        return JythonBridge.call("freqz_sos", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"sos", "worN", "whole", "fs"},
            types = {PyObject.class, PyObject.class, Boolean.class, Double.class})
    public PyObject sosfreqz(PyObject[] args, String[] keywords) {
        return JythonBridge.call("sosfreqz", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"wp", "ws", "gpass", "gstop", "analog", "fs"},
            types = {PyObject.class, PyObject.class, Double.class, Double.class, Boolean.class, Double.class})
    public PyObject buttord(PyObject[] args, String[] keywords) {
        return JythonBridge.call("buttord", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"wp", "ws", "gpass", "gstop", "analog", "fs"},
            types = {PyObject.class, PyObject.class, Double.class, Double.class, Boolean.class, Double.class})
    public PyObject cheb1ord(PyObject[] args, String[] keywords) {
        return JythonBridge.call("cheb1ord", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"wp", "ws", "gpass", "gstop", "analog", "fs"},
            types = {PyObject.class, PyObject.class, Double.class, Double.class, Boolean.class, Double.class})
    public PyObject cheb2ord(PyObject[] args, String[] keywords) {
        return JythonBridge.call("cheb2ord", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"wp", "ws", "gpass", "gstop", "analog", "fs"},
            types = {PyObject.class, PyObject.class, Double.class, Double.class, Boolean.class, Double.class})
    public PyObject ellipord(PyObject[] args, String[] keywords) {
        return JythonBridge.call("ellipord", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"b", "a", "x", "axis", "zi"},
            types = {PyObject.class, PyObject.class, PyObject.class, Integer.class, PyObject.class})
    public PyObject lfilter(PyObject[] args, String[] keywords) {
        return JythonBridge.call("lfilter", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"b", "a"},
            types = {PyObject.class, PyObject.class})
    public PyObject lfilter_zi(PyObject[] args, String[] keywords) {
        return JythonBridge.call("lfilter_zi", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"sos", "x", "axis", "zi"},
            types = {PyObject.class, PyObject.class, Integer.class, PyObject.class})
    public PyObject sosfilt(PyObject[] args, String[] keywords) {
        return JythonBridge.call("sosfilt", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"sos"},
            types = {PyObject.class})
    public PyObject sosfilt_zi(PyObject[] args, String[] keywords) {
        return JythonBridge.call("sosfilt_zi", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"b", "a", "x", "axis", "padtype", "padlen", "method", "irlen"},
            types = {PyObject.class, PyObject.class, PyObject.class, Integer.class, String.class, Integer.class, String.class, Integer.class})
    public PyObject filtfilt(PyObject[] args, String[] keywords) {
        return JythonBridge.call("filtfilt", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"sos", "x", "axis", "padtype", "padlen"},
            types = {PyObject.class, PyObject.class, Integer.class, String.class, Integer.class})
    public PyObject sosfiltfilt(PyObject[] args, String[] keywords) {
        return JythonBridge.call("sosfiltfilt", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"window_length", "polyorder", "deriv", "delta", "pos", "use"},
            types = {Integer.class, Integer.class, Integer.class, Double.class, Double.class, String.class})
    public PyObject savgol_coeffs(PyObject[] args, String[] keywords) {
        return JythonBridge.call("savgol_coeffs", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"x", "window_length", "polyorder", "deriv", "delta", "axis", "mode", "cval"},
            types = {PyObject.class, Integer.class, Integer.class, Integer.class, Double.class, Integer.class, String.class, Double.class})
    public PyObject savgol_filter(PyObject[] args, String[] keywords) {
        return JythonBridge.call("savgol_filter", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"volume", "kernel_size"},
            types = {PyObject.class, Integer.class})
    public PyObject medfilt(PyObject[] args, String[] keywords) {
        return JythonBridge.call("medfilt", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"x", "height", "threshold", "distance", "prominence", "width", "wlen", "rel_height", "plateau_size"},
            types = {PyObject.class, PyObject.class, PyObject.class, Double.class, PyObject.class, PyObject.class, Double.class, Double.class, PyObject.class})
    public PyObject find_peaks(PyObject[] args, String[] keywords) {
        return JythonBridge.call("find_peaks", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"x", "peaks", "wlen"},
            types = {PyObject.class, PyObject.class, Double.class})
    public PyObject peak_prominences(PyObject[] args, String[] keywords) {
        return JythonBridge.call("peak_prominences", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"x", "peaks", "rel_height", "prominence_data", "wlen"},
            types = {PyObject.class, PyObject.class, Double.class, PyObject.class, Double.class})
    public PyObject peak_widths(PyObject[] args, String[] keywords) {
        return JythonBridge.call("peak_widths", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"x", "fs", "window", "nperseg", "noverlap", "nfft", "detrend", "return_onesided", "scaling", "axis", "average"},
            types = {PyObject.class, Double.class, PyObject.class, Integer.class, Integer.class, Integer.class, PyObject.class, Boolean.class, String.class, Integer.class, String.class})
    public PyObject welch(PyObject[] args, String[] keywords) {
        return JythonBridge.call("welch", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"x", "fs", "window", "nfft", "detrend", "return_onesided", "scaling", "axis"},
            types = {PyObject.class, Double.class, PyObject.class, Integer.class, PyObject.class, Boolean.class, String.class, Integer.class})
    public PyObject periodogram(PyObject[] args, String[] keywords) {
        return JythonBridge.call("periodogram", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"x", "fs", "window", "nperseg", "noverlap", "nfft", "detrend", "return_onesided", "scaling", "axis", "mode"},
            types = {PyObject.class, Double.class, PyObject.class, Integer.class, Integer.class, Integer.class, PyObject.class, Boolean.class, String.class, Integer.class, String.class})
    public PyObject spectrogram(PyObject[] args, String[] keywords) {
        return JythonBridge.call("spectrogram", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"x", "N", "axis"},
            types = {PyObject.class, Integer.class, Integer.class})
    public PyObject hilbert(PyObject[] args, String[] keywords) {
        return JythonBridge.call("hilbert", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"x", "num", "t", "axis", "window", "domain"},
            types = {PyObject.class, Integer.class, PyObject.class, Integer.class, PyObject.class, String.class})
    public PyObject resample(PyObject[] args, String[] keywords) {
        return JythonBridge.call("resample", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"x", "up", "down", "axis", "window", "padtype", "cval"},
            types = {PyObject.class, Integer.class, Integer.class, Integer.class, PyObject.class, String.class, Double.class})
    public PyObject resample_poly(PyObject[] args, String[] keywords) {
        return JythonBridge.call("resample_poly", args, keywords);
    }

    @ScriptFunction(docBundlePrefix = DOCS)
    @KeywordArgs(names = {"x", "q", "n", "ftype", "axis", "zero_phase"},
            types = {PyObject.class, Integer.class, Integer.class, String.class, Integer.class, Boolean.class})
    public PyObject decimate(PyObject[] args, String[] keywords) {
        return JythonBridge.call("decimate", args, keywords);
    }
}
