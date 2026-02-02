// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <catch2/catch_message.hpp>
#include <complex>
#include <cstddef>

#include "DataStructures/Blaze/IntegerPow.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Equations.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "PointwiseFunctions/AnalyticData/SelfForce/Scalar/EccentricOrbit.hpp"
#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ScalarSelfForce::AnalyticData {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarSelfForce.EccentricOrbit",
                  "[PointwiseFunctions][Unit]") {
  // Set up a domain
  const double costheta_offset = 0.1;
  const double delta_costheta = 0.2;
  const double rstar_offset = 0.;
  const double delta_rstar = 5.;
  const size_t npoints = 10;
  const domain::creators::Rectangle domain_creator{
      {{rstar_offset, costheta_offset}},
      {{rstar_offset + delta_rstar, costheta_offset + delta_costheta}},
      {{0, 0}},
      {{npoints, npoints}},
      {{false, false}}};
  const auto domain = domain_creator.create_domain();
  const auto& block = domain.blocks()[0];
  const ElementId<2> element_id{0};
  const ElementMap<2, Frame::Inertial> element_map{element_id, block};
  const Mesh<2> mesh{npoints, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const auto xi = logical_coordinates(mesh);
  const auto x = element_map(xi);
  const auto inv_jacobian = element_map.inv_jacobian(xi);
  const auto& r_star = get<0>(x);
  const auto& cos_theta_or_sq = get<1>(x);
  CAPTURE(min(r_star));
  CAPTURE(max(r_star));
  CAPTURE(min(cos_theta_or_sq));
  CAPTURE(max(cos_theta_or_sq));

  size_t num_time_points = 300;
  size_t grid_point = 10;
  size_t time_point = 5;
  int m_mode_number = 1;
  int n_mode_number = 2;
  auto eccentric_orbit = EccentricOrbit{
    1., 0.9, 10., 0.6, m_mode_number, n_mode_number, {{-25., -5., 20., 40.}},
    false, num_time_points};
  auto evolved_sources =
      eccentric_orbit.evolve_sources(x, EccentricOrbit::evolution_tags{});
  std::vector<Scalar<ComplexDataVector>>& singular_field_evolution =
    get<Tags::SingularFieldEvolution>(evolved_sources);
  std::vector<tnsr::i<ComplexDataVector,2>>& deriv_singular_field_evolution =
    get<Tags::DerivSingularFieldEvolution>(evolved_sources);
  std::vector<Scalar<ComplexDataVector>>& effective_source_evolution =
    get<Tags::EffectiveSourceEvolution>(evolved_sources);

  double M = eccentric_orbit.black_hole_mass();
  double spin = eccentric_orbit.black_hole_spin();
  const DataVector r_minus_r_plus =
    gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(
        r_star,
        M,
        spin);
  const double r_plus = M * (1. + sqrt(1. - square(spin)));
  const double r_minus = M * (1. - sqrt(1. - square(spin)));
  const DataVector r = r_minus_r_plus + r_plus;
  const DataVector delta = r_minus_r_plus * (r - r_minus);
  const DataVector delta_phi =
    m_mode_number * spin / (r_plus - r_minus) *
      log((r - r_plus) / (r - r_minus));
  const ComplexDataVector rotation =
      cos(delta_phi) + std::complex<double>(0., 1.) * sin(delta_phi);
  DataVector cos_theta;
  DataVector cos_theta_sq;
  if (eccentric_orbit.impose_equatorial_symmetry()) {
    // NOLINTNEXTLINE
    cos_theta_sq.set_data_ref(const_cast<DataVector*>(&cos_theta_or_sq));
    cos_theta = sqrt(cos_theta_or_sq);
  } else {
    // NOLINTNEXTLINE
    cos_theta.set_data_ref(const_cast<DataVector*>(&cos_theta_or_sq));
    cos_theta_sq = square(cos_theta_or_sq);
  }
  const DataVector r_sq_plus_a_sq = square(r) + square(spin);
  const DataVector r_sq_plus_a_sq_sq = square(r_sq_plus_a_sq);
  const DataVector sin_theta_sq = 1. - cos_theta_sq;
  const DataVector sin_theta = sqrt(sin_theta_sq);
  const DataVector sin_theta_pow_m = integer_pow(sin_theta, m_mode_number);

  // Sum nmodes to n_max at time given by index time
  int n_max = 30;
  Scalar<ComplexDataVector> n_mode_sum_eff_source;
  Scalar<ComplexDataVector> n_mode_sum_singular_field;
  tnsr::i<ComplexDataVector,2> n_mode_sum_deriv_singular_field;
  get(n_mode_sum_eff_source).destructive_resize(get<0>(x).size());
  get(n_mode_sum_singular_field).destructive_resize(get<0>(x).size());
  get<0>(n_mode_sum_deriv_singular_field).destructive_resize(get<0>(x).size());
  get<1>(n_mode_sum_deriv_singular_field).destructive_resize(get<0>(x).size());
  for(int n = -n_max; n <= n_max; n++)
  {
    eccentric_orbit = EccentricOrbit{
      1., 0.9, 10., 0.6, m_mode_number, n,
      {{-25., -5., 20., 40.}}, false, num_time_points};
    auto vars =
      eccentric_orbit.variables(
        x, EccentricOrbit::source_tags{},
        effective_source_evolution, singular_field_evolution,
        deriv_singular_field_evolution);
    auto n_mode_effective_source =
      get<::Tags::FixedSource<Tags::NMode>>(vars);
    auto n_mode_singular_field =
      get<Tags::SingularField>(vars);
    auto n_mode_deriv_singular_field =
      get<
      ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>
      >(vars);

    std::complex<double> phase_factor =
        exp(std::complex<double>(
          0, -((n * eccentric_orbit.Omega_r) +
          (m_mode_number * eccentric_orbit.Omega_phi)) *
          eccentric_orbit.t_values[time_point]));

    get(n_mode_sum_eff_source) +=
      get(n_mode_effective_source) * phase_factor;
    get(n_mode_sum_singular_field) +=
      get(n_mode_singular_field) * phase_factor;
    get<0>(n_mode_sum_deriv_singular_field) +=
      get<0>(n_mode_deriv_singular_field) * phase_factor;
    get<1>(n_mode_sum_deriv_singular_field) +=
      get<1>(n_mode_deriv_singular_field) * phase_factor;
  }

  get(n_mode_sum_eff_source) *= (2*M_PI/r)*rotation *
    1/(delta * (square(r) + square(spin) * cos_theta_sq) /
                         r_sq_plus_a_sq_sq / sin_theta_pow_m);

  get(n_mode_sum_singular_field) *=
    rotation * 1 / (0.5 * r / M_PI / sin_theta_pow_m);

  get<0>(n_mode_sum_deriv_singular_field) *= rotation *
    1/(0.5 * r / M_PI / sin_theta_pow_m);
  get<0>(n_mode_sum_deriv_singular_field) -=
      get(singular_field_evolution[time_point]) / r +
      std::complex<double>(0., spin * m_mode_number) /
      delta * get(singular_field_evolution[time_point]);
  get<0>(n_mode_sum_deriv_singular_field) *= 1/(delta / r_sq_plus_a_sq);

  /*
  get<1>(deriv_singular_field) *= rotation * 0.5 * r / M_PI / sin_theta_pow_m;
  // This division is ok because the singular field is only evaluated at the
  // worldtube boundary where sin(theta) != 0. Also, only the normal to the
  // boundary is needed, so the angular derivative is discarded on the boundary
  // that extends to cos_theta = 0. On Gauss-Lobatto grid we may have to work
  // around this.

  get<1>(deriv_singular_field) /= -sin_theta;
  if (impose_equatorial_symmetry_) {
    get<1>(deriv_singular_field) /= 2. * cos_theta;
  }
  {
    ComplexDataVector add_term =
        m_mode_number_ * get(singular_field) / sin_theta_sq;
    if (impose_equatorial_symmetry_) {
      add_term *= 0.5;
    } else {
      add_term *= cos_theta;
    }
    get<1>(deriv_singular_field) += add_term;
  }
*/
  get<1>(n_mode_sum_deriv_singular_field) *= rotation *
    1/(0.5 * r / M_PI / sin_theta_pow_m);
  // This division is ok because the singular field is only evaluated at the
  // worldtube boundary where sin(theta) != 0. Also, only the normal to the
  // boundary is needed, so the angular derivative is discarded on the boundary
  // that extends to cos_theta = 0. On Gauss-Lobatto grid we may have to work
  // around this.

  get<1>(n_mode_sum_deriv_singular_field) *= -sin_theta;
  if (eccentric_orbit.impose_equatorial_symmetry()) {
    get<1>(n_mode_sum_deriv_singular_field) *= 2. * cos_theta;
  }
  {
    ComplexDataVector add_term =
        m_mode_number * get(singular_field_evolution[time_point])
        / sin_theta_sq;
    if (eccentric_orbit.impose_equatorial_symmetry()) {
      add_term *= 0.5;
    } else {
      add_term *= cos_theta;
    }
    get<1>(n_mode_sum_deriv_singular_field) -= add_term;
  }

  CAPTURE(get(n_mode_sum_eff_source)[grid_point]);
  CAPTURE(get(effective_source_evolution[time_point])[grid_point]);
  CAPTURE(get(n_mode_sum_singular_field)[grid_point]);
  CAPTURE(get(singular_field_evolution[time_point])[grid_point]);
  CAPTURE(get<0>(n_mode_sum_deriv_singular_field)[grid_point]);
  CAPTURE(get<0>(deriv_singular_field_evolution[time_point])[grid_point]);
  CAPTURE(get<1>(n_mode_sum_deriv_singular_field)[grid_point]);
  CAPTURE(get<1>(deriv_singular_field_evolution[time_point])[grid_point]);

  const Approx custom_approx = Approx::custom().epsilon(1.e-5).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(n_mode_sum_eff_source)[grid_point],
      get(effective_source_evolution[time_point])[grid_point],
      custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(n_mode_sum_singular_field)[grid_point],
      get(singular_field_evolution[time_point])[grid_point],
      custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get<0>(n_mode_sum_deriv_singular_field)[grid_point],
      get<0>(deriv_singular_field_evolution[time_point])[grid_point],
      custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get<1>(n_mode_sum_deriv_singular_field)[grid_point],
      get<1>(deriv_singular_field_evolution[time_point])[grid_point],
      custom_approx);
}
}  // Namespace ScalarSelfForce::AnalyticData
