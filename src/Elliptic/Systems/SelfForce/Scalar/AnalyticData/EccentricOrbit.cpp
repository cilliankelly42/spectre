// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/Scalar/AnalyticData/EccentricOrbit.hpp"

#include <algorithm>
#include <blaze/math/Vector.h>
#include <boost/iterator/is_iterator.hpp>
#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf.h>
#include <iomanip>
#include <ios>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <string>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/AnalyticData/SelfForceBackground.hpp"
extern "C" {
#include "korb.h"
}

#include <cmath>
#include <complex>
#include <cstddef>
#include <effsource_equatorial.hpp>
#include <gsl/gsl_errno.h>
#include <utility>

#include "DataStructures/Blaze/IntegerPow.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "NumericalAlgorithms/Integration/GslQuadAdaptive.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpline.hpp"
#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/System/ParallelInfo.hpp"

namespace ScalarSelfForce::AnalyticData {

namespace {
std::pair<DataVector, DataVector> boost_function_and_deriv(
    const DataVector& r_star, const std::array<double, 4>& transition_points) {
  return {
      smoothstep<1>(transition_points[0], transition_points[1], r_star) +
          smoothstep<1>(transition_points[2], transition_points[3], r_star) -
          1.0,
      smoothstep_deriv<1>(transition_points[0], transition_points[1], r_star) +
          smoothstep_deriv<1>(transition_points[2], transition_points[3],
                              r_star)};
}
}  // namespace

EccentricOrbit::EccentricOrbit(const double black_hole_mass,
                               const double black_hole_spin,
                               const double semi_latus_rectum,
                               const double eccentricity,
                               const int m_mode_number, 
                               const int n_mode_number,
                               const std::optional<std::array<double, 4>>
                                   hyperboloidal_slicing_transitions,
                               const bool penetrating_horizon,
                               const bool impose_equatorial_symmetry,
                               const size_t time_points)
    : black_hole_mass_(black_hole_mass),
      black_hole_spin_(black_hole_spin),
      semi_latus_rectum_(semi_latus_rectum),
      eccentricity_(eccentricity),
      m_mode_number_(m_mode_number),
      n_mode_number_(n_mode_number),
      hyperboloidal_slicing_transitions_(hyperboloidal_slicing_transitions),
      penetrating_horizon_(penetrating_horizon),
      impose_equatorial_symmetry_(impose_equatorial_symmetry),
      time_points_(time_points) {
  if (penetrating_horizon_ and
      not hyperboloidal_slicing_transitions_.has_value()) {
    ERROR(
        "Hyperboloidal slicing must be enabled when penetrating_horizon is "
        "true.");
  }
  compute_trajectory(time_points_);
}

EccentricOrbit::EccentricOrbit(CkMigrateMessage* m)
    : SelfForceBackground(m), elliptic::analytic_data::InitialGuess(m) {}

tnsr::I<double, 2> EccentricOrbit::puncture_position() const {
  const double M = black_hole_mass_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_0 = semi_latus_rectum_;
  if (penetrating_horizon_) {
    return tnsr::I<double, 2>{{{r_0, 0.}}};
  } else {
  const double r_star = gr::tortoise_radius_from_boyer_lindquist_minus_r_plus(
      r_0 - r_plus, M, black_hole_spin_);
  return tnsr::I<double, 2>{{{r_star, 0.}}};
  }
}

// Function to compute n-modes of a Scalar<ComplexDataVector> time series
ComplexDataVector EccentricOrbit::compute_n_mode(
    std::vector<Scalar<ComplexDataVector>>& time_series_data, size_t num_points,
    double integral_tolerance) const {
  ComplexDataVector result;
  result.destructive_resize(num_points);

  const integration::GslQuadAdaptive<
      integration::GslIntegralType::StandardGaussKronrod> 
      integration{5000};

  std::vector<double> re_gridpoint_to_interpolate(t_values.size());
  std::vector<double> im_gridpoint_to_interpolate(t_values.size());

  for (size_t i = 0; i < num_points; i++) {
    for (size_t j = 0; j < t_values.size(); j++) {
      std::complex<double> phase_factor = std::exp(std::complex<double>(
              0, (m_mode_number_ * Omega_phi + n_mode_number_ * Omega_r) * t_values[j]));
      re_gridpoint_to_interpolate[j] = 
        (get(time_series_data[j])[i].real() * phase_factor.real()) -
        (get(time_series_data[j])[i].imag() * phase_factor.imag());

      im_gridpoint_to_interpolate[j] = 
        (get(time_series_data[j])[i].imag() * phase_factor.real()) + 
        (get(time_series_data[j])[i].real() * phase_factor.imag());
    }
    intrp::CubicSpline re_interpolated_integrand{t_values,
                                                 re_gridpoint_to_interpolate};
    intrp::CubicSpline im_interpolated_integrand{t_values,
                                                 im_gridpoint_to_interpolate};

    double re_integral = 0;
    double im_integral = 0;

    re_integral = integration(
        [this, &re_interpolated_integrand](double t) {
          return (1 / t_values[t_values.size() - 1]) *
          re_interpolated_integrand(t);
        },
        0, t_values[t_values.size() - 1], integral_tolerance, 6);

    im_integral = integration(
        [this, &im_interpolated_integrand](double t) {
          return (1 / t_values[t_values.size() - 1]) *
          im_interpolated_integrand(t);
        },
        0, t_values[t_values.size() - 1], integral_tolerance, 6);

    result[i] = std::complex<double>(re_integral, im_integral);
  }
  return result;
}

tnsr::i<ComplexDataVector, 2> EccentricOrbit::compute_n_mode(
    std::vector<tnsr::i<ComplexDataVector, 2>>& time_series_data,
    size_t num_points, double integral_tolerance) const {
  tnsr::i<ComplexDataVector, 2> result;
  get<0>(result).destructive_resize(num_points);
  get<1>(result).destructive_resize(num_points);

  std::vector<double> re_rstar_deriv_integrand(t_values.size());
  std::vector<double> im_rstar_deriv_integrand(t_values.size());
  std::vector<double> re_theta_deriv_integrand(t_values.size());
  std::vector<double> im_theta_deriv_integrand(t_values.size());

  for (size_t i = 0; i < num_points; i++) {
    for (size_t j = 0; j < t_values.size(); j++) {
      std::complex<double> phase_factor = std::exp(std::complex<double>(
              0, (m_mode_number_ * Omega_phi + n_mode_number_ * Omega_r) * t_values[j]));
      re_rstar_deriv_integrand[j] = 
        (get<0>(time_series_data[j])[i].real() * phase_factor.real()) - 
        (get<0>(time_series_data[j])[i].imag() * phase_factor.imag());
      im_rstar_deriv_integrand[j] = 
        (get<0>(time_series_data[j])[i].imag() * phase_factor.real()) + 
        (get<0>(time_series_data[j])[i].real() * phase_factor.imag());
      re_theta_deriv_integrand[j] = 
        (get<1>(time_series_data[j])[i].real() * phase_factor.real()) - 
        (get<1>(time_series_data[j])[i].imag() * phase_factor.imag());
      im_theta_deriv_integrand[j] = 
        (get<1>(time_series_data[j])[i].imag() * phase_factor.real()) +
        (get<1>(time_series_data[j])[i].real() * phase_factor.imag());
    }

    intrp::CubicSpline re_interpolated_rstar_integrand{
        t_values, re_rstar_deriv_integrand};
    intrp::CubicSpline im_interpolated_rstar_integrand{
        t_values, im_rstar_deriv_integrand};
    intrp::CubicSpline re_interpolated_theta_integrand{
        t_values, re_theta_deriv_integrand};
    intrp::CubicSpline im_interpolated_theta_integrand{
        t_values, im_theta_deriv_integrand};

    double re_integral_0 = 0;
    double im_integral_0 = 0;
    double re_integral_1 = 0;
    double im_integral_1 = 0;

    const integration::GslQuadAdaptive<
        integration::GslIntegralType::StandardGaussKronrod>
        integration{5000};
    re_integral_0 = integration(
      [this, &re_interpolated_rstar_integrand](double t) 
      {
        return (1 / t_values[t_values.size() - 1]) * 
        re_interpolated_rstar_integrand(t);
      },
      0, t_values[t_values.size() - 1], integral_tolerance, 6);

    im_integral_0 = integration(
        [this, &im_interpolated_rstar_integrand](double t) {
          return (1 / t_values[t_values.size() - 1]) *
          im_interpolated_rstar_integrand(t);
        },
        0, t_values[t_values.size() - 1], integral_tolerance, 6);

      re_integral_1 = integration(
        [this, &re_interpolated_theta_integrand](double t) {
          return (1 / t_values[t_values.size() - 1]) *
          re_interpolated_theta_integrand(t);
        },
        0, t_values[t_values.size() - 1], integral_tolerance, 6);

    im_integral_1 = integration(
        [this, &im_interpolated_theta_integrand](double t) {
          return (1 / t_values[t_values.size() - 1]) *
          im_interpolated_theta_integrand(t);
        },
        0, t_values[t_values.size() - 1], integral_tolerance, 6);

    get<0>(result)[i] = std::complex<double>(re_integral_0, im_integral_0);
    get<1>(result)[i] = std::complex<double>(re_integral_1, im_integral_1);
  }
  return result;
}

// Compute the particle trajectory and specify the number of time points used 
// over one radial period
void EccentricOrbit::compute_trajectory(size_t time_points) {
  const int ecc = 1;
  const int inclined = 0;
  const double err = 1.0e-12;
  const double x_inclination = 1.0;
  korb_params orbpar;
  korb_getparams(ecc, inclined, black_hole_spin_, semi_latus_rectum_,
                 eccentricity_, x_inclination, err, &orbpar);
  if(eccentricity_ == 0)
  {
    orbpar.wr = orbpar.wphi; // For eccentricity = 0 case
  }
  energy = orbpar.E;
  angular_momentum = orbpar.Lz;
  Omega_r = orbpar.wr;
  Omega_phi = orbpar.wphi;
  r_of_t.resize(time_points);
  phi_of_t.resize(time_points);
  t_values.resize(time_points);
  u_r.resize(time_points);
  double lambda_max = 2 * M_PI / (orbpar.Ga * orbpar.wr);
  double delta_lambda = lambda_max / (time_points - 1);
  double t_max = 2 * M_PI / (orbpar.wr);
  double delta_t = t_max / (time_points - 1);
  std::vector<double> t_of_lambda(time_points + 1);
  std::vector<double> lambda_values(time_points + 1);
  double current_lambda = 0;

  for (size_t j = 0; j <= time_points; j++) {
    t_of_lambda[j] = korb_tfromla(current_lambda, orbpar);
    lambda_values[j] = current_lambda;
    current_lambda += delta_lambda;
  }

  intrp::CubicSpline lambda_of_t{t_of_lambda, lambda_values};

  std::vector<double> chi_of_t(time_points + 1);
  double current_t = 0;
  size_t j = 0;
  for (j = 0; j < time_points; j++) {
    r_of_t[j] =
        korb_rfrompsi(korb_psifromla(lambda_of_t(current_t), orbpar), orbpar);
    phi_of_t[j] = korb_phifromla(lambda_of_t(current_t), orbpar);
    chi_of_t[j] = korb_psifromla(lambda_of_t(current_t), orbpar);
    t_values[j] = current_t;
    current_t += delta_t;
  }

  // Compute the radial four velocity at each time
  double X_four_velocity = orbpar.Lz - (black_hole_spin_ * orbpar.E);
  for (size_t i = 0; i < u_r.size(); i++) {
    u_r[i] = (eccentricity_ * gsl_sf_sin(chi_of_t[i]) / semi_latus_rectum_) *
             sqrt(square(X_four_velocity) + square(black_hole_spin_) +
                  (2 * X_four_velocity * black_hole_spin_ * orbpar.E) -
                  (2 * square(X_four_velocity) *
                   (3 + eccentricity_ * gsl_sf_cos(chi_of_t[i])) /
                   semi_latus_rectum_));
  }
  korb_freepar(orbpar);
}

// Background
tuples::TaggedTuple<Tags::Alpha, Tags::Beta, Tags::Gamma>
EccentricOrbit::variables(
  const tnsr::I<DataVector, 2>& x,
  tmpl::list<Tags::Alpha, Tags::Beta, Tags::Gamma> /*meta*/) const {
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_minus = M * (1. - sqrt(1. - square(black_hole_spin_)));
  const double k = m_mode_number() * Omega_phi + n_mode_number() * Omega_r;

  // Resolve coordinates
  const auto& r_star_or_r = get<0>(x);
  DataVector r;
  DataVector r_star;
  DataVector r_minus_r_plus;
  if (penetrating_horizon_) {
    // NOLINTNEXTLINE
    r.set_data_ref(const_cast<DataVector*>(&r_star_or_r));
    r_minus_r_plus = r - r_plus;
    r_star = gr::tortoise_radius_from_boyer_lindquist_minus_r_plus(
        r_minus_r_plus, M, black_hole_spin_);
  } else {
    // NOLINTNEXTLINE
    r_star.set_data_ref(const_cast<DataVector*>(&r_star_or_r));
    r_minus_r_plus = gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(
        r_star, M, black_hole_spin_);
    r = r_minus_r_plus + r_plus;
  }
  const auto& cos_theta_or_sq = get<1>(x);
  DataVector cos_theta_sq;
  if (impose_equatorial_symmetry_) {
    // NOLINTNEXTLINE
    cos_theta_sq.set_data_ref(const_cast<DataVector*>(&cos_theta_or_sq));
  } else {
    cos_theta_sq = square(cos_theta_or_sq);
  }
  const DataVector delta = r_minus_r_plus * (r - r_minus);
  const DataVector r_sq_plus_a_sq = square(r) + square(a);
  const DataVector r_sq_plus_a_sq_sq = square(r_sq_plus_a_sq);
  const DataVector sin_theta_squared = 1. - cos_theta_sq;
  const DataVector sigma_squared =
      r_sq_plus_a_sq_sq - square(a) * delta * sin_theta_squared;
  tuples::TaggedTuple<Tags::Alpha, Tags::Beta, Tags::Gamma> result{};
  // Hyperboloidal slicing
  ComplexDataVector H;
  ComplexDataVector dH;
  if (hyperboloidal_slicing_transitions_.has_value()) {
    std::tie(H, dH) = boost_function_and_deriv(
        r_star_or_r, hyperboloidal_slicing_transitions_.value());
  } else {
    H = make_with_value<ComplexDataVector>(r_star_or_r, 0.);
    dH = make_with_value<ComplexDataVector>(r_star_or_r, 0.);
  }
  auto& alpha = get<Tags::Alpha>(result);
  auto& beta = get<Tags::Beta>(result);
  auto& gamma = get<Tags::Gamma>(result);
  if (penetrating_horizon_) {
    get<0>(alpha) = delta / r_sq_plus_a_sq;
    get<1>(alpha) = 1.0 / r_sq_plus_a_sq;
  } else {
    get<0>(alpha) = make_with_value<DataVector>(r_star_or_r, 1.0);
    get<1>(alpha) = delta / r_sq_plus_a_sq_sq;
  }
  get(beta) = make_with_value<ComplexDataVector>(r_star_or_r, 0.);
  for (size_t p = 0; p < get(beta).size(); ++p) {
    if (penetrating_horizon_ and equal_within_roundoff(r[p], r_plus)) {
      // The following terms are zero at the horizon. Skip them to avoid
      // division by zero.
      continue;
    }
    get(beta)[p] =
        square(k) *
            (square(H[p]) - sigma_squared[p] / r_sq_plus_a_sq_sq[p]) +
        2. * a * m_mode_number_ * k *
            (2. * M * r[p] / r_sq_plus_a_sq[p] + H[p]) / r_sq_plus_a_sq[p];
    if (penetrating_horizon_) {
      get(beta)[p] /= get<0>(alpha)[p];
    }
  }
  get(beta) +=
      get<1>(alpha) * (m_mode_number_ * (m_mode_number_ + 1) +
                       2. * M / r * (1. - square(a) / M / r) +
                       std::complex<double>(0., 2. * a * m_mode_number_) *
                           (1. + a * Omega_phi * H) / r) -
      std::complex<double>(0., k) * dH;
  get<0>(gamma) =
      2. * square(a) * delta / (r * r_sq_plus_a_sq_sq) -
      std::complex<double>(0., 2. * a * m_mode_number_) / r_sq_plus_a_sq -
      std::complex<double>(0., 2. * k) * H;
  get<1>(gamma) = 2. * m_mode_number_ * cos_theta_or_sq * get<1>(alpha);
  if (impose_equatorial_symmetry_) {
    get<1>(gamma) += sin_theta_squared * get<1>(alpha);
    get<1>(gamma) *= 2.0;
  }
  get<1>(alpha) *= sin_theta_squared;
  if (impose_equatorial_symmetry_) {
    get<1>(alpha) *= 4. * cos_theta_sq;
  }
  return result;
}

// Initial guess
tuples::TaggedTuple<Tags::MMode> EccentricOrbit::variables(
    const tnsr::I<DataVector, 2>& x, tmpl::list<Tags::MMode> /*meta*/) {
  tuples::TaggedTuple<Tags::MMode> result{};
  auto& field = get<Tags::MMode>(result);
  get(field) = ComplexDataVector{get<0>(x).size(), 0.};
  return result;
}

// Fixed sources
tuples::TaggedTuple<
    ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
    ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
    Tags::BoyerLindquistRadius>
EccentricOrbit::variables(
    const tnsr::I<DataVector, 2>& x, bool on_worldtube_boundary,
    tmpl::list<
        ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
        Tags::BoyerLindquistRadius> /*meta*/) const {
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_minus = M * (1. - sqrt(1. - square(black_hole_spin_)));
  const auto& r_star_or_r = get<0>(x);
  if (hyperboloidal_slicing_transitions_.has_value() and
      ((min(r_star_or_r) < (*hyperboloidal_slicing_transitions_)[1] and
        not equal_within_roundoff(min(r_star_or_r),
                                  (*hyperboloidal_slicing_transitions_)[1])) or
       (max(r_star_or_r) > (*hyperboloidal_slicing_transitions_)[2] and
        not equal_within_roundoff(max(r_star_or_r),
                                  (*hyperboloidal_slicing_transitions_)[2])))) {
    ERROR(
        "The effective source is only valid where no hyperboloidal slicing is "
        "applied, which is in the radial range ["
        << (*hyperboloidal_slicing_transitions_)[1] << ", "
        << (*hyperboloidal_slicing_transitions_)[2]
        << "], but was requested in the range [" << min(r_star_or_r) << ", "
        << max(r_star_or_r) << "]");
  }
  DataVector r;
  DataVector r_star;
  DataVector r_minus_r_plus;
  if (penetrating_horizon_) {
    // NOLINTNEXTLINE
    r.set_data_ref(const_cast<DataVector*>(&r_star_or_r));
    r_minus_r_plus = r - r_plus;
    r_star = gr::tortoise_radius_from_boyer_lindquist_minus_r_plus(
        r_minus_r_plus, M, black_hole_spin_);
  } else {
    // NOLINTNEXTLINE
    r_star.set_data_ref(const_cast<DataVector*>(&r_star_or_r));
    r_minus_r_plus = gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(
        r_star, M, black_hole_spin_);
    r = r_minus_r_plus + r_plus;
  }
  const auto& cos_theta_or_sq = get<1>(x);
  DataVector cos_theta;
  DataVector cos_theta_sq;
  if (impose_equatorial_symmetry_) {
    // NOLINTNEXTLINE
    cos_theta_sq.set_data_ref(const_cast<DataVector*>(&cos_theta_or_sq));
    cos_theta = sqrt(cos_theta_or_sq);
  } else {
    // NOLINTNEXTLINE
    cos_theta.set_data_ref(const_cast<DataVector*>(&cos_theta_or_sq));
    cos_theta_sq = square(cos_theta_or_sq);
  }
  const DataVector delta = r_minus_r_plus * (r - r_minus);
  const DataVector r_sq_plus_a_sq = square(r) + square(a);
  const DataVector r_sq_plus_a_sq_sq = square(r_sq_plus_a_sq);
  const DataVector sin_theta_sq = 1. - cos_theta_sq;
  const DataVector sin_theta = sqrt(sin_theta_sq);
  const DataVector sin_theta_pow_m = integer_pow(sin_theta, m_mode_number_);
  const DataVector delta_phi = m_mode_number_ * a / (r_plus - r_minus) *
                               log((r - r_plus) / (r - r_minus));
  const ComplexDataVector rotation =
      cos(delta_phi) - std::complex<double>(0., 1.) * sin(delta_phi);
  tuples::TaggedTuple<
      ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
      ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
      Tags::BoyerLindquistRadius>
      result{};
  get(get<Tags::BoyerLindquistRadius>(result)) = r;
  const size_t num_points = get<0>(x).size();
  Scalar<ComplexDataVector>& effective_source =
      get<::Tags::FixedSource<Tags::MMode>>(result);
  get(effective_source).destructive_resize(num_points);
  Scalar<ComplexDataVector>& singular_field = get<Tags::SingularField>(result);
  get(singular_field).destructive_resize(num_points);
  tnsr::i<ComplexDataVector, 2>& deriv_singular_field =
      get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
          result);
  get<0>(deriv_singular_field).destructive_resize(num_points);
  get<1>(deriv_singular_field).destructive_resize(num_points);

  // Define fixed sources at a given instance of time ("snapshots")
  Scalar<ComplexDataVector> snapshot_effective_source;
  get(snapshot_effective_source).destructive_resize(num_points);
  Scalar<ComplexDataVector> snapshot_singular_field;
  get(snapshot_singular_field).destructive_resize(num_points);
  tnsr::i<ComplexDataVector, 2> snapshot_deriv_singular_field;
  get<0>(snapshot_deriv_singular_field).destructive_resize(num_points);
  get<1>(snapshot_deriv_singular_field).destructive_resize(num_points);

  // Define the evolution of these fields
  std::vector<Scalar<ComplexDataVector>> effective_source_evolution;
  effective_source_evolution.resize(time_points());
  std::vector<Scalar<ComplexDataVector>> singular_field_evolution;
  singular_field_evolution.resize(time_points());
  std::vector<tnsr::i<ComplexDataVector, 2>> deriv_singular_field_evolution;
  deriv_singular_field_evolution.resize(time_points());
  deriv_singular_field_evolution.resize(time_points());
  std::vector<Scalar<ComplexDataVector>> raw_source_evolution;
  raw_source_evolution.resize(time_points());

  // Resize the inner array in the time evolutions
  for (auto& inner_array : singular_field_evolution) {
    get(inner_array).destructive_resize(num_points);
  }
  for (auto& inner_array : deriv_singular_field_evolution) {
    get<0>(inner_array).destructive_resize(num_points);
    get<1>(inner_array).destructive_resize(num_points);
  }
  for (auto& inner_array : effective_source_evolution) {
    get(inner_array).destructive_resize(num_points);
  }
  for (auto& inner_array : raw_source_evolution) {
    get(inner_array).destructive_resize(num_points);
  }

  {
    // Call into effsource
    coordinate x_i{};
    std::array<double, 2> PhiS{};
    std::array<double, 8> dPhiS_dx{};
    std::array<double, 20> d2PhiS_dx2{};
    std::array<double, 2> src{};
    struct effsource_equatorial_ctx * ctx = effsource_equatorial_create(M, a);
    for (size_t i = 0; i < t_values.size(); i++) {
      // Initialize effsource
      coordinate xp{};
      xp.t = t_values[i];
      xp.r = r_of_t[i];
      xp.theta = M_PI_2;
      xp.phi = phi_of_t[i];
      x_i.t = t_values[i];
      effsource_equatorial_ctx_set_particle(ctx, &xp, energy, angular_momentum, u_r[i]);

      for (size_t j = 0; j < num_points; j++) {
        x_i.r = r[j];
        x_i.theta = acos(cos_theta[j]);
        x_i.phi = 0;
        effsource_equatorial_ctx_calc_m(ctx, m_mode_number_, &x_i, PhiS.data(), dPhiS_dx.data(),
            d2PhiS_dx2.data(), src.data());
        get(snapshot_effective_source)[j] =
            src[0] + std::complex<double>(0., 1.) * src[1];
        get(snapshot_singular_field)[j] =
            PhiS[0] + std::complex<double>(0., 1.) * PhiS[1];
        get<0>(snapshot_deriv_singular_field)[j] =
            dPhiS_dx[2] + std::complex<double>(0., 1.) * dPhiS_dx[3];
        get<1>(snapshot_deriv_singular_field)[j] =
            dPhiS_dx[4] + std::complex<double>(0., 1.) * dPhiS_dx[5];
      }

      // Rotate the source by delta_phi and multiply by r / 2 pi
      get(effective_source_evolution[i]) = get(snapshot_effective_source);
      get(raw_source_evolution[i]) = get(snapshot_effective_source);
      get(singular_field_evolution[i]) = get(snapshot_singular_field);
      get<0>(deriv_singular_field_evolution[i]) =
          get<0>(snapshot_deriv_singular_field);
      get<1>(deriv_singular_field_evolution[i]) =
          get<1>(snapshot_deriv_singular_field);
      get(effective_source_evolution[i]) *= rotation * 0.5 * r / M_PI;
      // Factor Delta * (r^2 + a^2 cos^2(theta)) / Sigma^2
      // Factor Sigma^2 / (r^2 + a^2)^2 from first-order formulation
      // Factor 1/sin(theta)^m from change of variables
      get(effective_source_evolution[i]) *= (square(r) + square(a) * cos_theta_sq) /
                           (r_sq_plus_a_sq * sin_theta_pow_m);
      if (not penetrating_horizon_) {
        get(effective_source_evolution[i]) *= delta / r_sq_plus_a_sq;
      }
      get(singular_field_evolution[i]) *= rotation * 0.5 * r / M_PI / sin_theta_pow_m;
      get<0>(deriv_singular_field_evolution[i]) *= rotation * 0.5 * r / M_PI / sin_theta_pow_m;
      get<0>(deriv_singular_field_evolution[i]) +=
          get(singular_field_evolution[i]) / r - std::complex<double>(0., a * m_mode_number_) /
            delta * get(singular_field_evolution[i]);
      if (not penetrating_horizon_) {
        get<0>(deriv_singular_field_evolution[i]) *= delta / r_sq_plus_a_sq;
      }
      get<1>(deriv_singular_field_evolution[i]) *= rotation * 0.5 * r / M_PI / sin_theta_pow_m;
      get<1>(deriv_singular_field_evolution[i]) /= -sin_theta;
      if (impose_equatorial_symmetry_) {
        get<1>(deriv_singular_field_evolution[i]) /= 2. * cos_theta;
      }
      {
        ComplexDataVector add_term =
            m_mode_number_ * get(singular_field_evolution[i]) / sin_theta_sq;
        if (impose_equatorial_symmetry_) {
          add_term *= 0.5;
        } else {
          add_term *= cos_theta;
        }
        get<1>(deriv_singular_field_evolution[i]) += add_term;
      }
    }
    effsource_equatorial_free(ctx);
  }

  // only compute the n_modes of the singular field and its derivatives when 
  // crossing the worldtube 
  if(on_worldtube_boundary){
    deriv_singular_field =
      compute_n_mode(deriv_singular_field_evolution, num_points, 1e-8);
    try{
    get(singular_field) =
      compute_n_mode(singular_field_evolution, num_points, 1e-8);
    } catch (const convergence_error& e)
    {
        std::cout << "Failed at function call\n";
    }
  }

  get(effective_source) =
    compute_n_mode(effective_source_evolution, num_points, 1e-10);

  return result;
}

void EccentricOrbit::pup(PUP::er& p) {
  SelfForceBackground::pup(p);
  elliptic::analytic_data::InitialGuess::pup(p);
  p | black_hole_mass_;
  p | black_hole_spin_;
  p | semi_latus_rectum_;
  p | eccentricity_;
  p | m_mode_number_;
  p | n_mode_number_;
  p | hyperboloidal_slicing_transitions_;
  p | penetrating_horizon_;
  p | impose_equatorial_symmetry_;
  p | time_points_;
  p | t_values;
  p | r_of_t;
  p | phi_of_t;
  p | u_r;
  p | Omega_phi;
  p | Omega_r;
  p | energy;
  p | angular_momentum;
}

bool operator==(const EccentricOrbit& lhs, const EccentricOrbit& rhs) {
  return lhs.black_hole_mass_ == rhs.black_hole_mass_ and
         lhs.black_hole_spin_ == rhs.black_hole_spin_ and
         lhs.semi_latus_rectum_ == rhs.semi_latus_rectum_ and
         lhs.eccentricity_ == rhs.eccentricity_ and
         lhs.m_mode_number_ == rhs.m_mode_number_ and
         lhs.n_mode_number_ == rhs.n_mode_number_ and
         lhs.hyperboloidal_slicing_transitions_ ==
             rhs.hyperboloidal_slicing_transitions_ and
         lhs.penetrating_horizon_ == rhs.penetrating_horizon_ and
         lhs.impose_equatorial_symmetry_ == rhs.impose_equatorial_symmetry_ and
         lhs.time_points_ == rhs.time_points_;
         }

bool operator!=(const EccentricOrbit& lhs, const EccentricOrbit& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID EccentricOrbit::my_PUP_ID = 0;  // NOLINT

}  // namespace ScalarSelfForce::AnalyticData
