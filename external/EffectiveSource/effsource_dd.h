/*******************************************************************************
 * Double-double arithmetic for the prepare functions of effsource_equatorial_point.h.
 *
 * A value is the unevaluated sum hi + lo of two doubles with |lo| <= ulp(hi)/2,
 * which carries about 106 bits. Only IEEE double operations are used, so the
 * results are the same on every platform. Products use fma when the compiler
 * reports that it is fast (FP_FAST_FMA) and Dekker's splitting otherwise.
 * The algorithms follow Hida, Li and Bailey, "Library for double-double and
 * quad-double arithmetic" (2008).
 ******************************************************************************/

#ifndef EFFSOURCE_DD_H
#define EFFSOURCE_DD_H

#include <math.h>

typedef struct {
  double hi, lo;
} effsource_dd;

static inline effsource_dd dd_d(double a)
{
  return (effsource_dd){a, 0.0};
}

static inline double dd_to_double(effsource_dd a)
{
  return a.hi + a.lo;
}

static inline effsource_dd dd_quick_two_sum(double a, double b)
{
  const double s = a + b;
  return (effsource_dd){s, b - (s - a)};
}

static inline effsource_dd dd_two_sum(double a, double b)
{
  const double s = a + b;
  const double bb = s - a;
  return (effsource_dd){s, (a - (s - bb)) + (b - bb)};
}

static inline effsource_dd dd_two_prod(double a, double b)
{
  const double p = a * b;
#ifdef FP_FAST_FMA
  return (effsource_dd){p, fma(a, b, -p)};
#else
  const double split = 134217729.0;
  const double ta = split * a, tb = split * b;
  const double ahi = ta - (ta - a), alo = a - ahi;
  const double bhi = tb - (tb - b), blo = b - bhi;
  return (effsource_dd){p, ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo};
#endif
}

static inline effsource_dd dd_neg(effsource_dd a)
{
  return (effsource_dd){-a.hi, -a.lo};
}

static inline effsource_dd dd_add(effsource_dd a, effsource_dd b)
{
  effsource_dd s = dd_two_sum(a.hi, b.hi);
  const effsource_dd t = dd_two_sum(a.lo, b.lo);
  s.lo += t.hi;
  s = dd_quick_two_sum(s.hi, s.lo);
  s.lo += t.lo;
  return dd_quick_two_sum(s.hi, s.lo);
}

static inline effsource_dd dd_sub(effsource_dd a, effsource_dd b)
{
  return dd_add(a, dd_neg(b));
}

static inline effsource_dd dd_mul(effsource_dd a, effsource_dd b)
{
  effsource_dd p = dd_two_prod(a.hi, b.hi);
  p.lo += a.hi * b.lo + a.lo * b.hi;
  return dd_quick_two_sum(p.hi, p.lo);
}

static inline effsource_dd dd_mul_d(effsource_dd a, double b)
{
  effsource_dd p = dd_two_prod(a.hi, b);
  p.lo += a.lo * b;
  return dd_quick_two_sum(p.hi, p.lo);
}

static inline effsource_dd dd_div(effsource_dd a, effsource_dd b)
{
  const double q1 = a.hi / b.hi;
  effsource_dd r = dd_sub(a, dd_mul_d(b, q1));
  const double q2 = r.hi / b.hi;
  r = dd_sub(r, dd_mul_d(b, q2));
  const double q3 = r.hi / b.hi;
  return dd_add(dd_quick_two_sum(q1, q2), dd_d(q3));
}

static inline effsource_dd dd_div_d(effsource_dd a, double b)
{
  const double q1 = a.hi / b;
  const effsource_dd p = dd_two_prod(q1, b);
  effsource_dd s = dd_two_sum(a.hi, -p.hi);
  s.lo -= p.lo;
  s.lo += a.lo;
  const double q2 = (s.hi + s.lo) / b;
  return dd_quick_two_sum(q1, q2);
}

/* Square root of a > 0 by one Newton step from the double square root. */
static inline effsource_dd dd_sqrt(effsource_dd a)
{
  const double x = sqrt(a.hi);
  const effsource_dd x2 = dd_two_prod(x, x);
  const double correction = (a.hi - x2.hi - x2.lo + a.lo) / (2.0 * x);
  return dd_quick_two_sum(x, correction);
}

/* sin(x) and cos(x) for |x| <= pi/2: halve the argument until |x| < 2^-4, sum
   the Taylor series to double-double accuracy, then apply the double-angle
   formulas. */
static inline void dd_sincos(effsource_dd x, effsource_dd *sin_x, effsource_dd *cos_x)
{
  int halvings = 0;
  while (fabs(x.hi) > 0.0625) {
    x = dd_mul_d(x, 0.5);
    halvings++;
  }
  const effsource_dd x2 = dd_mul(x, x);
  effsource_dd s = dd_d(0.0), c = dd_d(0.0);
  for (int n = 8; n >= 1; n--) {
    s = dd_div(dd_sub(dd_d(1.0), dd_mul(s, x2)), dd_d((2.0 * n) * (2.0 * n + 1.0)));
    c = dd_div(dd_sub(dd_d(1.0), dd_mul(c, x2)), dd_d((2.0 * n - 1.0) * (2.0 * n)));
  }
  s = dd_mul(x, dd_sub(dd_d(1.0), dd_mul(s, x2)));
  c = dd_sub(dd_d(1.0), dd_mul(c, x2));
  for (int i = 0; i < halvings; i++) {
    const effsource_dd s2 = dd_mul_d(dd_mul(s, c), 2.0);
    c = dd_sub(dd_mul_d(dd_mul(c, c), 2.0), dd_d(1.0));
    s = s2;
  }
  *sin_x = s;
  *cos_x = c;
}

#endif /* EFFSOURCE_DD_H */
