// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <catch2/catch_message.hpp>
#include <complex>
#include <cstddef>
#include <iostream>

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

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarSelfForce.Basic_Test",
                  "[PointwiseFunctions][Unit]") {
  // Set up a domain
  const double costheta_offset = 0.1;
  const double delta_costheta = 0.2;
  const double rstar_offset = 0.;
  const double delta_rstar = 5.;
  const size_t npoints = 20;
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
  const auto& cos_theta = get<1>(x);
  CAPTURE(min(r_star));
  CAPTURE(max(r_star));
  CAPTURE(min(cos_theta));
  CAPTURE(max(cos_theta));

  int m_mode_number = 1;
  int max_n_mode_number = 30;
  auto eccentric_orbit = EccentricOrbit{
      1., 0.9, 10., 0.6, m_mode_number, 0, {{-25., -5., 20., 40.}}, false};
  eccentric_orbit.compute_trajectory();
  decltype(eccentric_orbit.variables(x, EccentricOrbit::source_tags{})) vars;
  // Select grid points [by index from 0 to get<0>(x).size()]
  // to compare m-mode effective source and sum over n-modes
  std::vector<size_t> grid_point_indices = {20};
  // Select time points (by index from 0 to time_points)
  // to compare m-mode effective source and sum over n-modes
  std::vector<size_t> time_indices = {29};
  std::vector<double> t_values_to_test(time_indices.size());
  for (size_t i = 0; i < time_indices.size(); i++) {
    t_values_to_test[i] = eccentric_orbit.t_values[time_indices[i]];
  }

  const Approx custom_approx = Approx::custom().epsilon(1.e-6).scale(1.);

  // Can test for multiple times by adding more indices to time_indices
  for (size_t t : time_indices) {
    Scalar<ComplexDataVector> n_mode_sum(get<0>(x).size(),
                                         std::complex<double>(0, 0));
    for (int n_mode_number = -max_n_mode_number;
         n_mode_number <= max_n_mode_number; n_mode_number++) {
      eccentric_orbit = EccentricOrbit{1.,
                                       0.9,
                                       10.,
                                       0.6,
                                       m_mode_number,
                                       n_mode_number,
                                       {{-25., -5., 20., 40.}},
                                       false};
      vars = eccentric_orbit.variables(x, EccentricOrbit::source_tags{});
      const auto& n_mode = get<Tags::NMode>(vars);
      get(n_mode_sum) +=
          get(n_mode) * exp(std::complex<double>(
                            0, -((n_mode_number * eccentric_orbit.Omega_r) +
                                 (m_mode_number * eccentric_orbit.Omega_phi)) *
                                   eccentric_orbit.t_values[t]));
    }
    CAPTURE(t);
    // Can test for multiple grid points by
    // adding more indices to grid_point_indices
    for (size_t grid_index : grid_point_indices) {
      CAPTURE(get(n_mode_sum)[grid_index]);
      CAPTURE(get(get<Tags::EffectiveSourceEvolution>(vars)[t])[grid_index]);
      CAPTURE(get(n_mode_sum)[grid_index] -
              get(get<Tags::EffectiveSourceEvolution>(vars)[t])[grid_index]);
      CHECK_ITERABLE_CUSTOM_APPROX(
          get(n_mode_sum)[grid_index],
          get(get<Tags::EffectiveSourceEvolution>(vars)[t])[grid_index],
          custom_approx);
    }
  }
}
}  // Namespace ScalarSelfForce::AnalyticData
