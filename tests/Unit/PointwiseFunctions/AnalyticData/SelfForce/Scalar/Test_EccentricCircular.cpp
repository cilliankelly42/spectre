// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
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
#include "PointwiseFunctions/AnalyticData/SelfForce/Scalar/CircularOrbit.hpp"
#include "PointwiseFunctions/AnalyticData/SelfForce/Scalar/EccentricOrbit.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ScalarSelfForce::AnalyticData {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarSelfForce.EccentricCircular",
                  "[PointwiseFunctions][Unit]") {
  // This test checks both the self-force equations and the effective source
  // computation in a very robust way: it ensures that the elliptic operator
  // applied to the singular field gives the effective source.
  // This is done numerically on a rectangular grid in (r_*, cos(theta)) near
  // the puncture.

  // Set up a domain
  const double costheta_offset = 0.1;
  const double delta_costheta = 0.2;
  const double rstar_offset = 0.;
  const double delta_rstar = 5.;
  const size_t npoints = 2;
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

  // Get the analytic fields for circular orbits
    const int m_mode_number = 2;
    const auto circular_orbit = CircularOrbit{
    1., 0.9, 6., m_mode_number, {{-25., -5., 20., 40.}}, false};
    CAPTURE(circular_orbit.puncture_position());
    const auto background_circular =
        circular_orbit.variables(x, CircularOrbit::background_tags{});
    const auto& alpha_circular = get<Tags::Alpha>(background_circular);
    const auto& beta_circular = get<Tags::Beta>(background_circular);
    const auto& gamma_circular = get<Tags::Gamma>(background_circular);
    const auto vars_circular =
      circular_orbit.variables(x, CircularOrbit::source_tags{});
    const auto& singular_field_circular =
      get<Tags::SingularField>(vars_circular);
    const auto& deriv_singular_field_circular = get<
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
        vars_circular);
    const auto& effective_source_circular =
      get<::Tags::FixedSource<Tags::MMode>>(vars_circular);

    const Approx custom_approx = Approx::custom().epsilon(1.e-10).scale(1.);

    tnsr::I<ComplexDataVector, 2> flux_singular_field_circular{};
    ScalarSelfForce::Fluxes::apply(make_not_null(&flux_singular_field_circular),
            alpha_circular, {}, deriv_singular_field_circular);
    auto scalar_eqn_circular =
      divergence(flux_singular_field_circular, mesh, inv_jacobian);
    get(scalar_eqn_circular) *= -1.;
    ScalarSelfForce::Sources::apply(
        make_not_null(&scalar_eqn_circular), beta_circular, gamma_circular,
        singular_field_circular, deriv_singular_field_circular,
        flux_singular_field_circular);

    const int n_mode_number = 0;
    const auto eccentric_orbit = EccentricOrbit{
    1., 0.9, 6., 0, m_mode_number, n_mode_number,
      {{-25., -5., 20., 40.}}, false, 5};
    CAPTURE(eccentric_orbit.puncture_position());
    const auto background_eccentric=
        eccentric_orbit.variables(x, EccentricOrbit::background_tags{});
    const auto& alpha_eccentric = get<Tags::Alpha>(background_eccentric);
    const auto& beta_eccentric = get<Tags::Beta>(background_eccentric);
    const auto& gamma_eccentric = get<Tags::Gamma>(background_eccentric);
    const auto vars_eccentric =
      eccentric_orbit.variables(x, EccentricOrbit::source_tags{});
    const auto& singular_field_eccentric=
      get<Tags::SingularField>(vars_eccentric);
    const auto& deriv_singular_field_eccentric = get<
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
        vars_circular);
    const auto& effective_source_eccentric =
      get<::Tags::FixedSource<Tags::MMode>>(vars_eccentric);

    tnsr::I<ComplexDataVector, 2> flux_singular_field_eccentric{};
    ScalarSelfForce::Fluxes::apply(
        make_not_null(&flux_singular_field_eccentric),
        alpha_eccentric, {}, deriv_singular_field_eccentric);
    auto scalar_eqn_eccentric=
      divergence(flux_singular_field_eccentric, mesh, inv_jacobian);
    get(scalar_eqn_circular) *= -1.;
    ScalarSelfForce::Sources::apply(
        make_not_null(&scalar_eqn_eccentric), beta_eccentric, gamma_eccentric,
        singular_field_eccentric, deriv_singular_field_eccentric,
        flux_singular_field_eccentric);
 }
} // Namespace ScalarSelfForce::AnalyticData
