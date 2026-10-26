/*******************************************************************************
 * Numerical m-modes of the equatorial effective source and singular field. See
 * effsource_equatorial_numerical.h for the formulas.
 ******************************************************************************/

#include <math.h>
#include <stdlib.h>

#include "effsource_equatorial_numerical.h"

static int is_power_of_two(size_t n)
{
  return n != 0 && (n & (n - 1)) == 0;
}

double effsource_equatorial_point_C1(const struct effsource_equatorial_ctx *ctx,
  const struct effsource_equatorial_point *point)
{
  return point->src_alpha / ctx->beta;
}

size_t effsource_equatorial_numerical_num_phi(double C1, int m)
{
  if (!(C1 > 0.0))
    return 0;
  const double abs_m = abs(m);
  const double log_rho = log1p(2.0 * C1 + 2.0 * sqrt(C1 * (1.0 + C1)));
  const double want = fmax(EFFSOURCE_NUMERICAL_MIN_PHI,
                           fmax(fmax(abs_m, 20.0) + EFFSOURCE_NUMERICAL_DECAY / log_rho, 2.5 * abs_m));
  if (!(want <= EFFSOURCE_NUMERICAL_MAX_PHI))
    return 0;
  size_t n = EFFSOURCE_NUMERICAL_MIN_PHI;
  while (n < want)
    n *= 2;
  return n;
}

int effsource_equatorial_numerical_table_init(struct effsource_equatorial_numerical_table *table,
  int m, size_t num_phi)
{
  if (num_phi < 4 || !is_power_of_two(num_phi))
    return 1;
  const size_t size = num_phi / 2 + 1;
  double *storage = malloc(4 * size * sizeof *storage);
  if (storage == NULL)
    return 2;
  table->m = abs(m);
  table->num_phi = num_phi;
  table->u = storage;
  table->half_sin = storage + size;
  table->cos_m = storage + 2 * size;
  table->sin_m = storage + 3 * size;
  /* Arguments are reduced exactly in integers: sin(psi_k / 2) = sin(pi k / N),
     sin(psi_k) = sin(pi (N - 2k) / N) and m psi_k = 2 pi ((m k) mod N) / N. */
  for (size_t k = 0; k < size; k++) {
    const double half = sin(M_PI * (double)k / (double)num_phi);
    table->u[k] = half * half;
    table->half_sin[k] = 0.5 * sin(M_PI * (double)(num_phi - 2 * k) / (double)num_phi);
    const double angle = 2.0 * M_PI * (double)(((size_t)table->m * k) % num_phi) / (double)num_phi;
    table->cos_m[k] = cos(angle);
    table->sin_m[k] = sin(angle);
  }
  return 0;
}

void effsource_equatorial_numerical_table_free(struct effsource_equatorial_numerical_table *table)
{
  free(table->u);
  table->u = table->half_sin = table->cos_m = table->sin_m = NULL;
}

/* Stride for num_phi nodes, or 0 for a bad num_phi. */
static size_t stride_of(const struct effsource_equatorial_numerical_table *table, size_t num_phi)
{
  if (num_phi < 4 || !is_power_of_two(num_phi) || num_phi > table->num_phi)
    return 0;
  return table->num_phi / num_phi;
}

/* (re + i im) e^{-i m phi0} 2 pi / N. */
static void rotate(const struct effsource_equatorial_ctx *ctx, const struct effsource_equatorial_point *point,
                   int m, size_t num_phi, double *re, double *im)
{
  const double phi0 = ctx->xp.phi + ctx->c * (point->r - ctx->xp.r);
  const double c = cos(m * phi0), s = sin(m * phi0), scale = 2.0 * M_PI / (double)num_phi;
  const double x = *re, y = *im;
  *re = scale * (c * x + s * y);
  *im = scale * (c * y - s * x);
}

int effsource_equatorial_ctx_numerical_src_mode(const struct effsource_equatorial_ctx *ctx,
  const struct effsource_equatorial_point *point,
  const struct effsource_equatorial_numerical_table *table, size_t num_phi,
  double *re, double *im)
{
  const size_t stride = stride_of(table, num_phi);
  if (stride == 0)
    return 1;
  const double *P = point->src_P, *Q = point->src_Q;
  const double alpha = point->src_alpha, beta = ctx->beta;
  double even = 0.0, odd = 0.0;
  /* Sum from the far side of the circle towards the peak, so that the many small
     terms are added before the few large ones. */
  for (size_t k = num_phi / 2; k-- > 1;) {
    const size_t j = k * stride;
    const double u = table->u[j];
    const double s2 = alpha + beta * u, s4 = s2 * s2;
    const double inverse = 1.0 / (s4 * s4 * s2 * sqrt(s2));
    const double Pu = P[0] + u * (P[1] + u * (P[2] + u * (P[3] + u * (P[4] + u * (P[5] + u * (P[6] + u * P[7]))))));
    const double Qu = Q[0] + u * (Q[1] + u * (Q[2] + u * (Q[3] + u * (Q[4] + u * (Q[5] + u * Q[6])))));
    even += Pu * inverse * table->cos_m[j];
    odd += table->half_sin[j] * Qu * inverse * table->sin_m[j];
  }
  const double u_far = table->u[table->num_phi / 2];
  const double s2_far = alpha + beta * u_far, s4_far = s2_far * s2_far;
  const double P_far = P[0] + u_far * (P[1] + u_far * (P[2] + u_far * (P[3] + u_far * (P[4] + u_far * (P[5]
                       + u_far * (P[6] + u_far * P[7]))))));
  const double E_far = P_far / (s4_far * s4_far * s2_far * sqrt(s2_far));
  const double s4_0 = alpha * alpha;
  const double E_0 = P[0] / (s4_0 * s4_0 * alpha * sqrt(alpha));
  *re = 2.0 * even + (table->m % 2 ? -E_far : E_far) + E_0;
  *im = -2.0 * odd;
  rotate(ctx, point, table->m, num_phi, re, im);
  return 0;
}

int effsource_equatorial_ctx_numerical_field_modes(const struct effsource_equatorial_ctx *ctx,
  const struct effsource_equatorial_point *point,
  const struct effsource_equatorial_numerical_table *table, size_t num_phi,
  double re[3], double im[3])
{
  const size_t stride = stride_of(table, num_phi);
  if (stride == 0)
    return 1;
  const double phi0 = ctx->xp.phi + ctx->c * (point->r - ctx->xp.r);
  for (int f = 0; f < 3; f++)
    re[f] = im[f] = 0.0;
  /* Node k > N/2 is node N - k mirrored: psi_{N-k} = -psi_k, so
     e^{-i m psi_{N-k}} = cos(m psi_k) + i sin(m psi_k). */
  for (size_t k = num_phi / 2 + 1; k-- > 0;) {
    const size_t j = k * stride;
    const int num_signs = (k == 0 || k == num_phi / 2) ? 1 : 2;
    for (int sign_index = 0; sign_index < num_signs; sign_index++) {
      const double sign = sign_index == 0 ? 1.0 : -1.0;
      const double psi = sign * 2.0 * M_PI * (double)k / (double)num_phi;
      double PhiS, dPhiS_dx[4], d2PhiS_dx2[10], src;
      effsource_equatorial_ctx_point_calc(ctx, point, phi0 + psi, &PhiS, dPhiS_dx, d2PhiS_dx2, &src);
      const double values[3] = {PhiS, dPhiS_dx[1], dPhiS_dx[2]};
      for (int f = 0; f < 3; f++) {
        re[f] += values[f] * table->cos_m[j];
        im[f] -= sign * values[f] * table->sin_m[j];
      }
    }
  }
  for (int f = 0; f < 3; f++)
    rotate(ctx, point, table->m, num_phi, &re[f], &im[f]);
  return 0;
}
