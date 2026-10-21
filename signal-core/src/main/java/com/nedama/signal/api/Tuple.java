package com.nedama.signal.api;

import java.util.AbstractList;

/**
 * A result that scipy returns as a tuple, such as {@code (f, Pxx)} from {@code welch}. The Ignition layer
 * turns it into a Python tuple, so {@code f, Pxx = system.signal.welch(x)} unpacks as it does with scipy.
 */
public final class Tuple extends AbstractList<Object> {

    private final Object[] items;

    private Tuple(Object[] items) {
        this.items = items;
    }

    public static Tuple of(Object... items) {
        return new Tuple(items.clone());
    }

    @Override
    public Object get(int index) {
        return items[index];
    }

    @Override
    public int size() {
        return items.length;
    }
}
