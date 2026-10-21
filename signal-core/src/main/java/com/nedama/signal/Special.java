package com.nedama.signal;

/**
 * Special functions used by the filter designs and windows, ported from the cephes code that scipy.special uses
 * (scipy 1.18.1, subprojects/xsf/include/xsf/cephes: i0.h, ellpk.h, ellpj.h, chbevl.h, polevl.h). Same
 * algorithms and coefficients, so results agree with scipy to within the rounding of the elementary functions.
 */
public final class Special {

    private Special() {
    }

    /** 2**-53, cephes MACHEP. */
    static final double MACHEP = 1.1102230246251565404236316680908203125E-16;

    private static final double[] I0_A = {
        -4.41534164647933937950E-18, 3.33079451882223809783E-17, -2.43127984654795469359E-16,
        1.71539128555513303061E-15, -1.16853328779934516808E-14, 7.67618549860493561688E-14,
        -4.85644678311192946090E-13, 2.95505266312963983461E-12, -1.72682629144155570723E-11,
        9.67580903537323691224E-11, -5.18979560163526290666E-10, 2.65982372468238665035E-9,
        -1.30002500998624804212E-8, 6.04699502254191894932E-8, -2.67079385394061173391E-7,
        1.11738753912010371815E-6, -4.41673835845875056359E-6, 1.64484480707288970893E-5,
        -5.75419501008210370398E-5, 1.88502885095841655729E-4, -5.76375574538582365885E-4,
        1.63947561694133579842E-3, -4.32430999505057594430E-3, 1.05464603945949983183E-2,
        -2.37374148058994688156E-2, 4.93052842396707084878E-2, -9.49010970480476444210E-2,
        1.71620901522208775349E-1, -3.04682672343198398683E-1, 6.76795274409476084995E-1
    };

    private static final double[] I0_B = {
        -7.23318048787475395456E-18, -4.83050448594418207126E-18, 4.46562142029675999901E-17,
        3.46122286769746109310E-17, -2.82762398051658348494E-16, -3.42548561967721913462E-16,
        1.77256013305652638360E-15, 3.81168066935262242075E-15, -9.55484669882830764870E-15,
        -4.15056934728722208663E-14, 1.54008621752140982691E-14, 3.85277838274214270114E-13,
        7.18012445138366623367E-13, -1.79417853150680611778E-12, -1.32158118404477131188E-11,
        -3.14991652796324136454E-11, 1.18891471078464383424E-11, 4.94060238822496958910E-10,
        3.39623202570838634515E-9, 2.26666899049817806459E-8, 2.04891858946906374183E-7,
        2.89137052083475648297E-6, 6.88975834691682398426E-5, 3.36911647825569408990E-3,
        8.04490411014108831608E-1
    };

    private static final double[] ELLPK_P = {
        1.37982864606273237150E-4, 2.28025724005875567385E-3, 7.97404013220415179367E-3,
        9.85821379021226008714E-3, 6.87489687449949877925E-3, 6.18901033637687613229E-3,
        8.79078273952743772254E-3, 1.49380448916805252718E-2, 3.08851465246711995998E-2,
        9.65735902811690126535E-2, 1.38629436111989062502E0
    };

    private static final double[] ELLPK_Q = {
        2.94078955048598507511E-5, 9.14184723865917226571E-4, 5.94058303753167793257E-3,
        1.54850516649762399335E-2, 2.39089602715924892727E-2, 3.01204715227604046988E-2,
        3.73774314173823228969E-2, 4.88280347570998239232E-2, 7.03124996963957469739E-2,
        1.24999999999870820058E-1, 4.99999999999999999821E-1
    };

    /** log(4). */
    private static final double ELLPK_C1 = 1.3862943611198906188E0;

    /** Modified Bessel function of the first kind, order 0 (scipy.special.i0). */
    public static double i0(double x) {
        if (x < 0) {
            x = -x;
        }
        if (x <= 8.0) {
            double y = (x / 2.0) - 2.0;
            return Math.exp(x) * chbevl(y, I0_A, 30);
        }
        return Math.exp(x) * chbevl(32.0 / x - 2.0, I0_B, 25) / Math.sqrt(x);
    }

    /** Complete elliptic integral of the first kind K(m) (scipy.special.ellipk). */
    public static double ellipk(double m) {
        return ellpk(1.0 - m);
    }

    /** K(1 - p), accurate for small p (scipy.special.ellipkm1). */
    public static double ellipkm1(double p) {
        return ellpk(p);
    }

    /** cephes ellpk: K(1 - x). NaN below 0, infinity at 0. */
    static double ellpk(double x) {
        if (x < 0.0) {
            return Double.NaN;
        }
        if (x > 1.0) {
            if (Double.isInfinite(x)) {
                return 0.0;
            }
            return ellpk(1 / x) / Math.sqrt(x);
        }
        if (x > MACHEP) {
            return polevl(x, ELLPK_P, 10) - Math.log(x) * polevl(x, ELLPK_Q, 10);
        }
        if (x == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return ELLPK_C1 - 0.5 * Math.log(x);
    }

    /**
     * Jacobi elliptic functions sn, cn, dn and the amplitude ph of u for parameter m (scipy.special.ellipj).
     * Returns {sn, cn, dn, ph}; all NaN when m is outside [0, 1] or NaN.
     */
    public static double[] ellipj(double u, double m) {
        double sn;
        double cn;
        double dn;
        double ph;
        if (m < 0.0 || m > 1.0 || Double.isNaN(m)) {
            return new double[] {Double.NaN, Double.NaN, Double.NaN, Double.NaN};
        }
        if (m < 1.0e-9) {
            double t = Math.sin(u);
            double b = Math.cos(u);
            double ai = 0.25 * m * (u - t * b);
            sn = t - ai * b;
            cn = b + ai * t;
            ph = u - ai;
            dn = 1.0 - 0.5 * m * t * t;
            return new double[] {sn, cn, dn, ph};
        }
        if (m >= 0.9999999999) {
            double ai = 0.25 * (1.0 - m);
            double b = Math.cosh(u);
            double t = Math.tanh(u);
            double phi = 1.0 / b;
            double twon = b * Math.sinh(u);
            sn = t + ai * (twon - u) / (b * b);
            ph = 2.0 * Math.atan(Math.exp(u)) - Math.PI / 2 + ai * (twon - u) / b;
            ai *= t * phi;
            cn = phi - ai * (twon - u);
            dn = phi + ai * (twon + u);
            return new double[] {sn, cn, dn, ph};
        }

        // Arithmetic-geometric mean scale, DLMF 22.20(ii).
        double[] a = new double[9];
        double[] c = new double[9];
        a[0] = 1.0;
        double b = Math.sqrt(1.0 - m);
        c[0] = Math.sqrt(m);
        double twon = 1.0;
        int i = 0;
        while (Math.abs(c[i] / a[i]) > MACHEP) {
            if (i > 7) {
                break;
            }
            double ai = a[i];
            ++i;
            c[i] = (ai - b) / 2.0;
            double t = Math.sqrt(ai * b);
            a[i] = (ai + b) / 2.0;
            b = t;
            twon *= 2.0;
        }

        // Backward recurrence.
        double phi = twon * a[i] * u;
        do {
            double t = c[i] * Math.sin(phi) / a[i];
            b = phi;
            phi = (Math.asin(t) + phi) / 2.0;
        } while (--i != 0);

        sn = Math.sin(phi);
        double t = Math.cos(phi);
        cn = t;
        double dnfac = Math.cos(phi - b);
        // See the discussion after DLMF 22.20.5.
        if (Math.abs(dnfac) < 0.1) {
            dn = Math.sqrt(1 - m * sn * sn);
        } else {
            dn = t / dnfac;
        }
        ph = phi;
        return new double[] {sn, cn, dn, ph};
    }

    /** Chebyshev series evaluation (cephes chbevl). */
    static double chbevl(double x, double[] array, int n) {
        int p = 0;
        double b0 = array[p++];
        double b1 = 0.0;
        double b2;
        int i = n - 1;
        do {
            b2 = b1;
            b1 = b0;
            b0 = x * b1 - b2 + array[p++];
        } while (--i != 0);
        return 0.5 * (b0 - b2);
    }

    /** Polynomial evaluation with coefficients from the highest power down (cephes polevl). */
    static double polevl(double x, double[] coef, int n) {
        int p = 0;
        double ans = coef[p++];
        int i = n;
        do {
            ans = ans * x + coef[p++];
        } while (--i != 0);
        return ans;
    }

    /** Inverse hyperbolic cosine, accurate near 1. NaN below 1. */
    public static double acosh(double x) {
        if (Double.isNaN(x) || x < 1.0) {
            return Double.NaN;
        }
        if (x > 1e8) {
            return Math.log(x) + Math.log(2.0);
        }
        double t = x - 1.0;
        return Math.log1p(t + Math.sqrt(2.0 * t + t * t));
    }

    /** sin(pi x) / (pi x), with numpy's handling of x = 0. */
    public static double sinc(double x) {
        double y = Math.PI * (x == 0 ? 1.0e-20 : x);
        return Math.sin(y) / y;
    }
}
