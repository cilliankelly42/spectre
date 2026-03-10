// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/SelfForce/Scalar/EccentricOrbit.hpp"

#include <algorithm>
#include <blaze/math/Vector.h>
#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include "DataStructures/Tensor/IndexType.hpp"
#include "Utilities/TaggedTuple.hpp"
extern "C" {
#include "korb.h"
}

#include <cmath>
#include <complex>
#include <cstddef>
#include <effsource.hpp>
#include <gsl/gsl_errno.h>
#include <fstream>
#include <iostream>
#include <mutex>
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
#include "Utilities/ErrorHandling/GslErrorHandler.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/System/ParallelInfo.hpp"

namespace ScalarSelfForce::AnalyticData {

namespace {
template <size_t Order>
std::pair<DataVector, DataVector> boost_function_and_deriv(
    const DataVector& r_star, const std::array<double, 4>& transition_points) {
  return {
      smoothstep<Order>(transition_points[0], transition_points[1], r_star) +
          smoothstep<Order>(transition_points[2], transition_points[3],
                            r_star) -
          1.0,
      smoothstep_deriv<Order>(transition_points[0], transition_points[1],
                              r_star) +
          smoothstep_deriv<Order>(transition_points[2], transition_points[3],
                                  r_star)};
}
}  // namespace

EccentricOrbit::EccentricOrbit(const double black_hole_mass,
                               const double black_hole_spin,
                               const double semi_latus_rectum,
                               const double eccentricity,
                               const int m_mode_number, const int n_mode_number,
                               const std::optional<std::array<double, 4>>
                                   hyperboloidal_slicing_transitions,
                               const bool impose_equatorial_symmetry,
                               const size_t time_points)
    : black_hole_mass_(black_hole_mass),
      black_hole_spin_(black_hole_spin),
      semi_latus_rectum_(semi_latus_rectum),
      eccentricity_(eccentricity),
      m_mode_number_(m_mode_number),
      n_mode_number_(n_mode_number),
      hyperboloidal_slicing_transitions_(hyperboloidal_slicing_transitions),
      impose_equatorial_symmetry_(impose_equatorial_symmetry),
      time_points_(time_points) {
  compute_trajectory(time_points_);
}

EccentricOrbit::EccentricOrbit(CkMigrateMessage* m)
    : elliptic::analytic_data::Background(m),
      elliptic::analytic_data::InitialGuess(m) {}

tnsr::I<double, 2> EccentricOrbit::puncture_position() const {
  const double M = black_hole_mass_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_0 = semi_latus_rectum_;
  const double r_star = gr::tortoise_radius_from_boyer_lindquist_minus_r_plus(
      r_0 - r_plus, M, black_hole_spin_);
  return tnsr::I<double, 2>{{{r_star, 0.}}};
}

// Function to compute n-modes of a Scalar<ComplexDataVector> time series
ComplexDataVector EccentricOrbit::compute_n_mode(
    std::vector<Scalar<ComplexDataVector>>& time_series_data, size_t num_points,
    double integral_tolerance) const {
  gsl_throw_exceptions();
  ComplexDataVector result;
  result.destructive_resize(num_points);

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

    const integration::GslQuadAdaptive<
        integration::GslIntegralType::StandardGaussKronrod>
        integration{5000};

    double re_integral = 0;
    double im_integral = 0;

    /* std::ofstream integrand_full_steps("/home/user/debug_gsl/full_step.txt");
    std::ofstream integrand_half_steps("/home/user/debug_gsl/half_step.txt");
    integrand_full_steps << "time" << "\t"
      << "re_integrand_full_step" << "\t" 
      << "im_integrand_full_step" << "\n";
    integrand_half_steps << "time" << "\t"
      << "re_integrand_half_step" << "\t"
      << "im_integrand_half_step" << "\n";
    double dt = (t_values[1] - t_values[0]);
    for(size_t k=0; k < time_points_; k++)
    {
      double full_step = k*dt;
      integrand_full_steps << full_step << "\t"
        << re_interpolated_integrand(full_step)  << "\t"
        << im_interpolated_integrand(full_step) << "\n";
    }
    for(size_t k=0; k < time_points_ - 1; k++)
    {
      double half_step = (k*dt) + (dt/2);
      integrand_half_steps << half_step << "\t"
        << re_interpolated_integrand(half_step) << "\t"
        << im_interpolated_integrand(half_step) << "\n";
    }
    integrand_full_steps.close();
    integrand_half_steps.close(); */

    // try{
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
/*
      } catch (const std::runtime_error& e) {
        std::cout << "integration is FAILING";
        std::ofstream singular_field_integrand("/home/user/debug_gsl/singular_field_integral.txt");
        singular_field_integrand << "times" << "\t" << "singular_field_integrand" << "\n";
        std::cout << "n_mode_number: " << n_mode_number_ << "\n";
        std::cout << "omega_r " << Omega_r << "\n";
        for(double time : t_values)
        {
          singular_field_integrand << time << "\t" 
            << re_interpolated_integrand(time) << "\n";
        }
        singular_field_integrand.close();
        throw e;
    } */
    result[i] = std::complex<double>(re_integral, im_integral);
  }
  return result;
}

tnsr::i<ComplexDataVector, 2> EccentricOrbit::compute_n_mode(
    std::vector<tnsr::i<ComplexDataVector, 2>>& time_series_data,
    size_t num_points, double integral_tolerance) const {
  gsl_throw_exceptions();
  tnsr::i<ComplexDataVector, 2> result;
  get<0>(result).destructive_resize(num_points);
  get<1>(result).destructive_resize(num_points);

  std::vector<double> re_rstar_deriv_integrand(t_values.size());
  std::vector<double> im_rstar_deriv_integrand(t_values.size());
  std::vector<double> re_theta_deriv_integrand(t_values.size());
  std::vector<double> im_theta_deriv_integrand(t_values.size());

  std::ofstream re_grid_interp_data("/home/user/spline_testing/data_to_interpolate.txt");
  re_grid_interp_data << "times" << "\t" << "data_to_interpolate" << "\n";
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
      re_grid_interp_data << t_values[j] << "\t"
        << re_theta_deriv_integrand[j] << "\n";
    }

    re_grid_interp_data.close();
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

    // try{
      re_integral_1 = integration(
        [this, &re_interpolated_theta_integrand](double t) {
          return (1 / t_values[t_values.size() - 1]) *
          re_interpolated_theta_integrand(t);
        },
        0, t_values[t_values.size() - 1], integral_tolerance, 6);
      /* } catch(const std::runtime_error& e) {
        std::cout << "Integration failed here: " << "\n";
        std::ofstream integrand("/home/user/spline_testing/integrand.txt");
        integrand << "time" << "\t" 
          << "re(theta_deriv)_integrand" << "\t" 
          << "re(rstar_deriv)_integrand" 
          << "\n";
        std::vector<double> test_times(999);
        double delta_t = (t_values[1] - t_values[0])/2;
        for(size_t k=0; k < test_times.size(); k++)
        {
          test_times[k] = k*delta_t;
          std::complex<double> phase_factor = std::exp(std::complex<double>(
                0, (m_mode_number_ * Omega_phi + n_mode_number_ * Omega_r) * test_times[k]));
          integrand << test_times[k] << "\t" 
            <<  (1 / t_values[t_values.size() - 1]) *
                  re_interpolated_theta_integrand(test_times[k]) << "\t"
            << (1 / t_values[t_values.size() - 1]) *
                 re_interpolated_rstar_integrand(test_times[k])
            << "\n";
        }
        integrand.close();
        throw e; 
      }*/

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

// Define the orbital parameters as global variables
const int ecc = 1;
const int inclined = 0;
const double err = 1.0e-12;
const double x_inclination = 1.0;
korb_params orbpar;

// Compute the particle trajectory and specify the number of time points
// to store the coordinates at over one radial period
void EccentricOrbit::compute_trajectory(size_t time_points) {
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
}

// Background
tuples::tagged_tuple_from_typelist<typename EccentricOrbit::background_tags>
EccentricOrbit::variables(const tnsr::I<DataVector, 2>& x,
                          background_tags /*meta*/) const {
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_minus = M * (1. - sqrt(1. - square(black_hole_spin_)));
  const double r_0 = semi_latus_rectum_;  // Adjust for AMR
  const auto& r_star = get<0>(x);
  const auto& cos_theta_or_sq = get<1>(x);
  DataVector cos_theta_sq;
  if (impose_equatorial_symmetry_) {
    // NOLINTNEXTLINE
    cos_theta_sq.set_data_ref(const_cast<DataVector*>(&cos_theta_or_sq));
  } else {
    cos_theta_sq = square(cos_theta_or_sq);
  }
  const DataVector r_minus_r_plus =
      gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(r_star, M,
                                                            black_hole_spin_);
  const DataVector r = r_minus_r_plus + r_plus;
  const DataVector delta = r_minus_r_plus * (r - r_minus);
  const DataVector r_sq_plus_a_sq = square(r) + square(a);
  const DataVector r_sq_plus_a_sq_sq = square(r_sq_plus_a_sq);
  const DataVector sin_theta_squared = 1. - cos_theta_sq;
  const DataVector sigma_squared =
      r_sq_plus_a_sq_sq - square(a) * delta * sin_theta_squared;
  tuples::tagged_tuple_from_typelist<background_tags> result{};
  auto& alpha = get<Tags::Alpha>(result);
  auto& beta = get<Tags::Beta>(result);
  auto& gamma = get<Tags::Gamma>(result);
  get(alpha) = delta / r_sq_plus_a_sq_sq;
  const ComplexDataVector temp1 =
      1. / r * std::complex<double>(0., 2. * a * m_mode_number_);
  get(beta) = (-square(m_mode_number_ * Omega_phi + n_mode_number_* Omega_r) *
      sigma_squared + 4. * a * m_mode_number_ *
      (m_mode_number_ * Omega_phi + n_mode_number_ * Omega_r) * M * r +
               delta * (m_mode_number_ * (m_mode_number_ + 1) +
                        2. * M / r * (1. - square(a) / M / r) + temp1)) /
              r_sq_plus_a_sq_sq;
  get<0>(gamma) =
      -1. / r_sq_plus_a_sq * std::complex<double>(0., 2. * a * m_mode_number_) +
      2. * square(a) * get(alpha) / r;
  get<1>(gamma) = 2. * m_mode_number_ * cos_theta_or_sq * get(alpha);
  if (impose_equatorial_symmetry_) {
    get<1>(gamma) += sin_theta_squared * get(alpha);
    get<1>(gamma) *= 2.0;
  }
  get(alpha) *= sin_theta_squared;
  if (impose_equatorial_symmetry_) {
    get(alpha) *= 4. * cos_theta_sq;
  }
  // Hyperboloidal slicing
  if (hyperboloidal_slicing_transitions_.has_value()) {
    const auto [H, dH] = boost_function_and_deriv<1>(
        r_star, hyperboloidal_slicing_transitions_.value());
    get(get<Tags::BoostFunction>(result)) = H;
    get(get<Tags::BoostFunctionDeriv>(result)) = dH;
    const double k = m_mode_number_ * Omega_phi;
    get(beta) += std::complex<double>(0., -k) * dH + square(k) * square(H) +
                 std::complex<double>(0., k) * get<0>(gamma) * H;
    get<0>(gamma) -= std::complex<double>(0., 2. * k) * H;
  } else {
    get(get<Tags::BoostFunction>(result)) =
        ComplexDataVector{r_star.size(), 0.};
    get(get<Tags::BoostFunctionDeriv>(result)) =
        ComplexDataVector{r_star.size(), 0.};
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
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<
        ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
        Tags::BoyerLindquistRadius> /*meta*/) const {
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_0 = semi_latus_rectum_;  // Set to rmin/rmax
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_minus = M * (1. - sqrt(1. - square(black_hole_spin_)));

  const auto& r_star = get<0>(x);
  if (hyperboloidal_slicing_transitions_.has_value() and
      not((min(r_star) >= (*hyperboloidal_slicing_transitions_)[1] or
           equal_within_roundoff(min(r_star),
                                 (*hyperboloidal_slicing_transitions_)[1])) and
          (max(r_star) <= (*hyperboloidal_slicing_transitions_)[2] or
           equal_within_roundoff(max(r_star),
                                 (*hyperboloidal_slicing_transitions_)[2])))) {
    ERROR(
        "The effective source is only valid where no hyperboloidal slicing is "
        "applied, which is in the r_* range ["
        << (*hyperboloidal_slicing_transitions_)[1] << ", "
        << (*hyperboloidal_slicing_transitions_)[2]
        << "], but was requested in the range [" << min(r_star) << ", "
        << max(r_star) << "]");
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
  const DataVector r_minus_r_plus =
    gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(r_star, M,
                                                            black_hole_spin_);
  const DataVector r = r_minus_r_plus + r_plus;
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

  size_t gridpoint = 0;
  {
    std::lock_guard<std::mutex> lock(*effsource_mutex);
    // Call into effsource
    coordinate x_i{};
    std::array<double, 2> PhiS{};
    std::array<double, 8> dPhiS_dx{};
    std::array<double, 20> d2PhiS_dx2{};
    std::array<double, 2> src{};
    effsource_init(M, a);

    for (size_t i = 0; i < t_values.size(); i++) {
      // Initialize effsource
      coordinate xp{};
      xp.t = t_values[i];
      xp.r = r_of_t[i];
      xp.theta = M_PI_2;
      xp.phi = phi_of_t[i];
      x_i.t = t_values[i];
      effsource_set_particle(&xp, energy, angular_momentum, u_r[i]);

      for (size_t j = 0; j < num_points; j++) {
        if((12.75 < get<0>(x)[j]) && (get<0>(x)[j] < 12.78))
        {
          gridpoint = j;
        }
        x_i.r = r[j];
        x_i.theta = acos(cos_theta[j]);
        x_i.phi = 0;
        effsource_calc_m(m_mode_number_, &x_i, PhiS.data(), dPhiS_dx.data(),
                         d2PhiS_dx2.data(), src.data());
        get(snapshot_effective_source)[j] =
            src[0] + std::complex<double>(0., 1.) * src[1];
        get(snapshot_singular_field)[j] =
            PhiS[0] + std::complex<double>(0., 1.) * PhiS[1];
        get<0>(snapshot_deriv_singular_field)[j] =
            dPhiS_dx[2] + std::complex<double>(0., 1.) * dPhiS_dx[3];
        get<1>(snapshot_deriv_singular_field)[j] =
            dPhiS_dx[4] + std::complex<double>(0., 1.) * dPhiS_dx[5];
        get(effective_source_evolution[i]) = get(snapshot_effective_source);
      }
      get(singular_field_evolution[i]) = get(snapshot_singular_field);
      get<0>(deriv_singular_field_evolution[i]) =
          get<0>(snapshot_deriv_singular_field);
      get<1>(deriv_singular_field_evolution[i]) =
          get<1>(snapshot_deriv_singular_field);
            get(effective_source_evolution[i]) *= rotation * 0.5 * r / M_PI;

      // Factor Delta * (r^2 + a^2 cos^2(theta)) / Sigma^2
      // Factor Sigma^2 / (r^2 + a^2)^2 from first-order formulation
      // Factor 1/sin(theta)^m from change of variables

      get(effective_source_evolution[i]) *= delta * (square(r) + square(a) *
          cos_theta_sq) / r_sq_plus_a_sq_sq / sin_theta_pow_m;
      get(singular_field_evolution[i]) *= rotation * 0.5 * r / M_PI
        / sin_theta_pow_m;

      get<0>(deriv_singular_field_evolution[i]) *= rotation *
        0.5 * r / M_PI / sin_theta_pow_m;

      get<0>(deriv_singular_field_evolution[i]) +=
          get(singular_field_evolution[i]) / r -
          std::complex<double>(0., a * m_mode_number_) /
            delta * get(singular_field_evolution[i]);

      get<0>(deriv_singular_field_evolution[i]) *= delta / r_sq_plus_a_sq;

      get<1>(deriv_singular_field_evolution[i]) *= rotation * 0.5 *
        r / M_PI / sin_theta_pow_m;

    // This division is ok because the singular field is only evaluated at the
    // worldtube boundary where sin(theta) != 0. Also, only the normal to the
    // boundary is needed, so the angular derivative is discarded on the
    // boundary that extends to cos_theta = 0. On Gauss-Lobatto grid we may
    // have to work around this.

      get<1>(deriv_singular_field_evolution[i]) /= -sin_theta;

      if (impose_equatorial_symmetry_) {
        get<1>(deriv_singular_field_evolution[i]) /= 2. * cos_theta;
      }

      {
        ComplexDataVector add_term =
            m_mode_number_ * get(singular_field_evolution[i]) /
            sin_theta_sq;
        if (impose_equatorial_symmetry_) {
          add_term *= 0.5;
        } else {
          add_term *= cos_theta;
        }
        get<1>(deriv_singular_field_evolution[i]) += add_term;
      }
    }
  }
  // try{
  get(singular_field) =
      compute_n_mode(singular_field_evolution, num_points, 1e-8);
  /* } catch(const std::runtime_error& error){
    std::cerr << "Integration failed  on singular field " << "\n";
  } */

  try{
  deriv_singular_field =
      compute_n_mode(deriv_singular_field_evolution, num_points, 1e-8);
  } catch(const std::runtime_error& error){
    std::cerr << "Failed on deriv_singular_field " << "\n";
    std::ofstream integral_data("/home/user/field_points.txt");
    if(integral_data.is_open()){
      integral_data << "r_star" << "\t" << "cos(theta)" << "\t" << "time" << "\t" << 
        "re(rstar_deriv_singular_field)" << "\t" << 
        "im(rstar_deriv_singular_field)"<< "\t" << 
        "re(costheta_deriv_sing_field)" << "\t" <<
        "im(costheta_deriv_sing_field)" << "\t" << 
        "re(effsource)" << "\t" << "im(effsource)"<< "\t"
        "real_singular_field" << "\t" << "imag_singular_field" <<"\n";

      for(size_t i=0; i < num_points; i++){
        for(size_t j=0; j < time_points_; j++){
          integral_data << get<0>(x)[i] << "\t" << get<1>(x)[i] << "\t" << 
            t_values[j] << "\t" << 
            get<0>(deriv_singular_field_evolution[j])[i].real() << "\t" <<
            get<0>(deriv_singular_field_evolution[j])[i].imag() << "\t" <<
            get<1>(deriv_singular_field_evolution[j])[i].real() << "\t" <<
            get<1>(deriv_singular_field_evolution[j])[i].imag() << "\t" << 
            get(effective_source_evolution[j])[i].real() << "\t" <<
            get(effective_source_evolution[j])[i].imag() << "\t"  << 
            get(singular_field_evolution[j])[i].real() << "\t" <<
            get(singular_field_evolution[j])[i].imag() << "\n";
        }
      }
    integral_data.close();
    std::cout << "Written to file" << "\n";
    throw error;
    }
  }

  // try{
  get(effective_source) =
    compute_n_mode(effective_source_evolution, num_points, 1e-8);
  /* } catch(const std::runtime_error& error){
    std::cerr << "Failed on effsource integration" << "\n";
  } */

  /* if(gridpoint != 0)
  {
  std::cout << puncture_position() << "\n";
  std::cout << gridpoint << "\n";
  std::ofstream all_data("/home/user/field_data.txt");
  if(all_data.is_open()){
    all_data << "r_star" << "\t" << "cos(theta)" << "\t" << "time" << "\t" << 
      "re(rstar_deriv_singular_field)" << "\t" << 
      "im(rstar_deriv_singular_field)"<< "\t" << 
      "re(costheta_deriv_sing_field)" << "\t" <<
      "im(costheta_deriv_sing_field)" << "\t" << 
      "re(effsource)" << "\t" << "im(effsource)"<< "\t"
      "real_singular_field" << "\t" << "imag_singular_field" <<"\n";

    for(size_t j=0; j < time_points_; j++){
      all_data << get<0>(x)[gridpoint] << "\t" << get<1>(x)[gridpoint] << "\t" << 
      t_values[j] << "\t" << 
      get<0>(deriv_singular_field_evolution[j])[gridpoint].real() << "\t" <<
      get<0>(deriv_singular_field_evolution[j])[gridpoint].imag() << "\t" <<
      get<1>(deriv_singular_field_evolution[j])[gridpoint].real() << "\t" <<
      get<1>(deriv_singular_field_evolution[j])[gridpoint].imag() << "\t" << 
      get(effective_source_evolution[j])[gridpoint].real() << "\t" <<
      get(effective_source_evolution[j])[gridpoint].imag() << "\t"  << 
      get(singular_field_evolution[j])[gridpoint].real() << "\t" <<
      get(singular_field_evolution[j])[gridpoint].imag() << "\n";
    }
  }
  all_data.close();
  std::cout << "Written to file" << "\n"; 
  } */
  return result;
}

void EccentricOrbit::pup(PUP::er& p) {
  elliptic::analytic_data::Background::pup(p);
  elliptic::analytic_data::InitialGuess::pup(p);
  p | black_hole_mass_;
  p | black_hole_spin_;
  p | semi_latus_rectum_;
  p | eccentricity_;
  p | m_mode_number_;
  p | n_mode_number_;
  p | hyperboloidal_slicing_transitions_;
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
         lhs.impose_equatorial_symmetry_ == rhs.impose_equatorial_symmetry_ and
         lhs.time_points_ == rhs.time_points_;
         }

bool operator!=(const EccentricOrbit& lhs, const EccentricOrbit& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID EccentricOrbit::my_PUP_ID = 0;  // NOLINT

}  // namespace ScalarSelfForce::AnalyticData
