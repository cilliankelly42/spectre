/*******************************************************************************
 * Numerical m-modes of the equatorial effective source and singular field from
 * the coefficients of effsource_equatorial_ctx_prepare.
 *
 * The m-mode f_m = int_0^{2 pi} f e^{-i m phi} dphi is computed by the
 * trapezoidal rule on N nodes centred on the peak of the source,
 *
 *   phi_k = phi0 + psi_k,  psi_k = 2 pi k / N,  phi0 = xp.phi + c (r - xp.r),
 *
 * so that dQ = sin(psi_k / 2), u = dQ^2, dQ cQ = sin(psi_k) / 2 and the phases
 * cos(m psi_k) and sin(m psi_k) depend on N and m only. They are tabulated once
 * and reused for every field point. The src of effsource_equatorial_ctx_point_src,
 *
 *   src = [P(u) + dQ cQ Q(u)] / s^11,  s^2 = src_alpha + beta u,
 *
 * has an even part E = P(u) / s^11 and an odd part O = sin(psi) Q(u) / (2 s^11)
 * in psi, so the nodes psi_k and -psi_k share one evaluation:
 *
 *   S_m = (2 pi / N) e^{-i m phi0} [E_0 + (-1)^m E_{N/2}
 *         + 2 sum_{k=1}^{N/2-1} (E_k cos(m psi_k) - i O_k sin(m psi_k))].
 *
 * The number of nodes follows the rule
 *
 *   N = 2^ceil(log2 max(64, max(|m|, 20) + 46 / log(rho), 2.5 |m|)),
 *   log(rho) = log(1 + 2 C1 + 2 sqrt(C1 (1 + C1))),
 *
 * with C1 = src_alpha / beta. The Fourier coefficients of src decrease as
 * rho^-mu, so the aliasing error of mode m decreases as rho^-(N - m). With
 * (N - m) log(rho) >= 46 it is below 1e-14 of int |src| dphi, the rounding of
 * the sum, for m <= 20 and C1 from 1e-7 to 1. For small C1, 46 / log(rho) is
 * about 23 / sqrt(C1). Powers of two let one table serve every smaller N, read
 * with a stride.
 ******************************************************************************/

#ifndef EFFSOURCE_EQUATORIAL_NUMERICAL_H
#define EFFSOURCE_EQUATORIAL_NUMERICAL_H

#include <stddef.h>

#include "effsource_equatorial_point.h"

/* Smallest and largest number of nodes of the rule. */
#define EFFSOURCE_NUMERICAL_MIN_PHI 64
#define EFFSOURCE_NUMERICAL_MAX_PHI ((size_t)1 << 22)

/* Smallest value of (N - |m|) log(rho) of the rule. */
#define EFFSOURCE_NUMERICAL_DECAY 46.0

/* C1 = src_alpha / beta of a prepared field point. */
double effsource_equatorial_point_C1(const struct effsource_equatorial_ctx *ctx,
  const struct effsource_equatorial_point *point);

/* Number of nodes of the rule for mode m at separation C1. Returns 0 unless
   C1 > 0, and if the rule needs more than EFFSOURCE_NUMERICAL_MAX_PHI nodes. */
size_t effsource_equatorial_numerical_num_phi(double C1, int m);

/* Node values for |m| on num_phi nodes psi_k = 2 pi k / num_phi, k = 0..num_phi/2. */
struct effsource_equatorial_numerical_table {
  int m;             /* |m| */
  size_t num_phi;    /* power of two, at least 4 */
  double *u;         /* sin^2(psi_k / 2) */
  double *half_sin;  /* sin(psi_k) / 2 */
  double *cos_m;     /* cos(m psi_k) */
  double *sin_m;     /* sin(m psi_k) */
};

/* Fill the table for |m| and num_phi nodes. Returns 0, 1 if num_phi is not a
   power of two of at least 4, or 2 if the allocation fails; the table needs no
   freeing unless 0 is returned. */
int effsource_equatorial_numerical_table_init(struct effsource_equatorial_numerical_table *table,
  int m, size_t num_phi);

/* Free the arrays of a table filled by effsource_equatorial_numerical_table_init. */
void effsource_equatorial_numerical_table_free(struct effsource_equatorial_numerical_table *table);

/* The m-mode S_m = re + i im of the src of effsource_equatorial_ctx_point_src for
   m = table->m on num_phi nodes. num_phi must be a power of two of at least 4
   that divides table->num_phi; the table is read with stride table->num_phi / num_phi.
   Returns 0, or 1 for a bad num_phi. */
int effsource_equatorial_ctx_numerical_src_mode(const struct effsource_equatorial_ctx *ctx,
  const struct effsource_equatorial_point *point,
  const struct effsource_equatorial_numerical_table *table, size_t num_phi,
  double *re, double *im);

/* The m-modes of PhiS, dPhiS/dr and dPhiS/dtheta of effsource_equatorial_ctx_point_calc
   (index 0, 1, 2) for m = table->m on the nodes of
   effsource_equatorial_ctx_numerical_src_mode, without the even/odd halving.
   Same rules for num_phi and return values. */
int effsource_equatorial_ctx_numerical_field_modes(const struct effsource_equatorial_ctx *ctx,
  const struct effsource_equatorial_point *point,
  const struct effsource_equatorial_numerical_table *table, size_t num_phi,
  double re[3], double im[3]);

#endif /* EFFSOURCE_EQUATORIAL_NUMERICAL_H */
