package com.nedama.signal.api;

/** Computes one function from its bound arguments and returns a result value (see {@link Functions}). */
@FunctionalInterface
public interface Handler {

    Object call(Args args);
}
