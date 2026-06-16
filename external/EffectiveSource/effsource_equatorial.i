/*******************************************************************************
 * Copyright (C) 2012 Barry Wardell
 ******************************************************************************/

#ifndef EFFSOURCE_EQUATORIAL_H
#define EFFSOURCE_EQUATORIAL_H

#include "effsource.h"

/* Thread-safe context-based API for equatorial orbits */
struct effsource_equatorial_ctx {
  struct coordinate xp;
  double M, a;

  /* Series expansion coefficients (set by effsource_equatorial_ctx_set_particle) */
  double A0060, A0061, A0080, A0081, A0240, A0241, A0260, A0261, A0420, A0421,
         A0440, A0441, A0600, A0601, A0620, A0621, A0800, A0801, A1060, A1061,
         A1080, A1240, A1241, A1260, A1420, A1421, A1440, A1600, A1601, A1620,
         A1800, A2040, A2041, A2060, A2061, A2220, A2221, A2240, A2241, A2400,
         A2401, A2420, A2421, A2600, A2601, A3040, A3041, A3060, A3220, A3221,
         A3240, A3400, A3401, A3420, A3600, A4020, A4021, A4040, A4041, A4200,
         A4201, A4220, A4221, A4400, A4401, A5020, A5021, A5040, A5200, A5201,
         A5220, A5400, A6000, A6001, A6020, A6021, A6200, A6201, A7000, A7001,
         A7020, A7200, A8000, A8001, A9000;
  double alpha20, alpha02, beta, c;
  double rt, urt, rtt, urtt, phit, phitt;

  /* Time derivative coefficients (set by effsource_equatorial_ctx_set_particle_dt) */
  double dAdt0060, dAdt0061, dAdt0080, dAdt0081, dAdt0240, dAdt0241, dAdt0260,
         dAdt0261, dAdt0420, dAdt0421, dAdt0440, dAdt0441, dAdt0600, dAdt0601,
         dAdt0620, dAdt0621, dAdt0800, dAdt0801, dAdt1060, dAdt1061, dAdt1080,
         dAdt1240, dAdt1241, dAdt1260, dAdt1420, dAdt1421, dAdt1440, dAdt1600,
         dAdt1601, dAdt1620, dAdt1800, dAdt2040, dAdt2041, dAdt2060, dAdt2061,
         dAdt2220, dAdt2221, dAdt2240, dAdt2241, dAdt2400, dAdt2401, dAdt2420,
         dAdt2421, dAdt2600, dAdt2601, dAdt3040, dAdt3041, dAdt3060, dAdt3220,
         dAdt3221, dAdt3240, dAdt3400, dAdt3401, dAdt3420, dAdt3600, dAdt4020,
         dAdt4021, dAdt4040, dAdt4041, dAdt4200, dAdt4201, dAdt4220, dAdt4221,
         dAdt4400, dAdt4401, dAdt5020, dAdt5021, dAdt5040, dAdt5200, dAdt5201,
         dAdt5220, dAdt5400, dAdt6000, dAdt6001, dAdt6020, dAdt6021, dAdt6200,
         dAdt6201, dAdt7000, dAdt7001, dAdt7020, dAdt7200, dAdt8000, dAdt8001,
         dAdt9000;
  double dalphadt20, dalphadt02, dbetadt, dcdt;
  double dC1_dt02, dC1_dt10, dC1_dt20;

  /* Second time derivative coefficients (set by effsource_equatorial_ctx_set_particle_dtt) */
  double d2Adt20060, d2Adt20061, d2Adt20080, d2Adt20081, d2Adt20240, d2Adt20241,
         d2Adt20260, d2Adt20261, d2Adt20420, d2Adt20421, d2Adt20440, d2Adt20441,
         d2Adt20600, d2Adt20601, d2Adt20620, d2Adt20621, d2Adt20800, d2Adt20801,
         d2Adt21060, d2Adt21061, d2Adt21080, d2Adt21240, d2Adt21241, d2Adt21260,
         d2Adt21420, d2Adt21421, d2Adt21440, d2Adt21600, d2Adt21601, d2Adt21620,
         d2Adt21800, d2Adt22040, d2Adt22041, d2Adt22060, d2Adt22061, d2Adt22220,
         d2Adt22221, d2Adt22240, d2Adt22241, d2Adt22400, d2Adt22401, d2Adt22420,
         d2Adt22421, d2Adt22600, d2Adt22601, d2Adt23040, d2Adt23041, d2Adt23060,
         d2Adt23220, d2Adt23221, d2Adt23240, d2Adt23400, d2Adt23401, d2Adt23420,
         d2Adt23600, d2Adt24020, d2Adt24021, d2Adt24040, d2Adt24041, d2Adt24200,
         d2Adt24201, d2Adt24220, d2Adt24221, d2Adt24400, d2Adt24401, d2Adt25020,
         d2Adt25021, d2Adt25040, d2Adt25200, d2Adt25201, d2Adt25220, d2Adt25400,
         d2Adt26000, d2Adt26001, d2Adt26020, d2Adt26021, d2Adt26200, d2Adt26201,
         d2Adt27000, d2Adt27001, d2Adt27020, d2Adt27200, d2Adt28000, d2Adt28001,
         d2Adt29000;
  double d2alphadt220, d2alphadt202, d2betadt2, d2cdt2;
  double d2C1_dt200, d2C1_dt202, d2C1_dt210, d2C1_dt220;
};

struct effsource_equatorial_ctx * effsource_equatorial_create(double M, double a);
void effsource_equatorial_free(struct effsource_equatorial_ctx * ctx);
void effsource_equatorial_ctx_set_particle(struct effsource_equatorial_ctx * ctx, struct coordinate * x_p, double E, double L, double ur);
void effsource_equatorial_ctx_set_particle_dt(struct effsource_equatorial_ctx * ctx, struct coordinate * x_p, double E, double L, double ur);
void effsource_equatorial_ctx_set_particle_dtt(struct effsource_equatorial_ctx * ctx, struct coordinate * x_p, double E, double L, double ur);
void effsource_equatorial_ctx_PhiS(struct effsource_equatorial_ctx * ctx, struct coordinate * x, double * PhiS);
void effsource_equatorial_ctx_calc(struct effsource_equatorial_ctx * ctx, struct coordinate * x,
  double * PhiS, double * dPhiS_dx, double * d2PhiS_dx2, double * src);
void effsource_equatorial_ctx_PhiS_m(struct effsource_equatorial_ctx * ctx, int m, struct coordinate * x, double * PhiS);
void effsource_equatorial_ctx_calc_m(struct effsource_equatorial_ctx * ctx, int m, struct coordinate * x,
  double * PhiS, double * dPhiS_dx, double * d2PhiS_dx2, double * src);

#endif /* EFFSOURCE_EQUATORIAL_H */
