package com.nedama.signal;

/**
 * An immutable complex number. Arithmetic follows numpy's formulas (division uses Smith's method, as numpy
 * does), so iterative algorithms ported from scipy take the same path.
 */
public final class Complex {

    public static final Complex ZERO = new Complex(0.0, 0.0);
    public static final Complex ONE = new Complex(1.0, 0.0);

    public final double re;
    public final double im;

    public Complex(double re, double im) {
        this.re = re;
        this.im = im;
    }

    public static Complex of(double re, double im) {
        return new Complex(re, im);
    }

    public Complex plus(Complex o) {
        return new Complex(re + o.re, im + o.im);
    }

    public Complex plus(double d) {
        return new Complex(re + d, im);
    }

    public Complex minus(Complex o) {
        return new Complex(re - o.re, im - o.im);
    }

    public Complex times(Complex o) {
        return new Complex(re * o.re - im * o.im, re * o.im + im * o.re);
    }

    public Complex times(double d) {
        return new Complex(re * d, im * d);
    }

    public Complex negate() {
        return new Complex(-re, -im);
    }

    public Complex conj() {
        return new Complex(re, -im);
    }

    /** this / o, by Smith's method with numpy's branches. */
    public Complex div(Complex o) {
        double absR = Math.abs(o.re);
        double absI = Math.abs(o.im);
        if (absR >= absI) {
            if (absR == 0 && absI == 0) {
                return new Complex(re / absR, im / absI);
            }
            double rat = o.im / o.re;
            double scl = 1.0 / (o.re + o.im * rat);
            return new Complex((re + im * rat) * scl, (im - re * rat) * scl);
        }
        double rat = o.re / o.im;
        double scl = 1.0 / (o.im + o.re * rat);
        return new Complex((re * rat + im) * scl, (im * rat - re) * scl);
    }

    /** 1 / this. */
    public Complex reciprocal() {
        return ONE.div(this);
    }

    public double abs() {
        return Math.hypot(re, im);
    }

    /** this * this, as numpy computes z**2. */
    public Complex square() {
        return times(this);
    }

    /** Principal square root, with the branch handling of the C library's csqrt that numpy calls. */
    public Complex sqrt() {
        if (im == 0.0) {
            if (re < 0) {
                return new Complex(0.0, Math.copySign(Math.sqrt(-re), im));
            }
            return new Complex(Math.abs(Math.sqrt(re)), Math.copySign(0.0, im));
        }
        if (re == 0.0) {
            double r = Math.sqrt(0.5 * Math.abs(im));
            return new Complex(r, Math.copySign(r, im));
        }
        double d = Math.hypot(re, im);
        double r;
        double s;
        if (re > 0) {
            r = Math.sqrt(0.5 * (d + re));
            s = 0.5 * (im / r);
        } else {
            s = Math.sqrt(0.5 * (d - re));
            r = Math.abs(0.5 * (im / s));
        }
        return new Complex(r, Math.copySign(s, im));
    }

    /** e^this. */
    public Complex exp() {
        if (re == 0.0) {
            return new Complex(Math.cos(im), Math.sin(im));
        }
        double e = Math.exp(re);
        return new Complex(e * Math.cos(im), e * Math.sin(im));
    }

    /** sinh(this) = (sinh x cos y, cosh x sin y). */
    public Complex sinh() {
        return new Complex(Math.sinh(re) * Math.cos(im), Math.cosh(re) * Math.sin(im));
    }

    /** this / d for a real d, as numpy does it: d becomes d + 0j and Smith's method multiplies by 1/d. */
    public Complex divReal(double d) {
        return div(new Complex(d, 0.0));
    }

    /** True when the imaginary part is exactly zero, as numpy.isreal tests. */
    public boolean isReal() {
        return im == 0.0;
    }

    public boolean isFinite() {
        return Double.isFinite(re) && Double.isFinite(im);
    }

    @Override
    public String toString() {
        return "(" + re + (im < 0 || (im == 0 && 1 / im < 0) ? "-" : "+") + Math.abs(im) + "j)";
    }
}
