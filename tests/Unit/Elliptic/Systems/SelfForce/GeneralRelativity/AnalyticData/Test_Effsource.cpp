// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <fstream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Equations.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace GrSelfForce::AnalyticData {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GrSelfForce.EffectiveSource",
                  "[PointwiseFunctions][Unit]") {

    const double rstar_offset = 12.6409;
    const double delta_rstar = 0.249847;
    const size_t npoints = 40;
    const domain::creators::Rectangle domain_creator{
      {{rstar_offset, M_PI_2 - 0.1}},
      {{rstar_offset + delta_rstar, M_PI_2 + 0.1}},
      {{0, 0}},
      {{npoints, 3}},
      {{false, false}}};
    const auto domain = domain_creator.create_domain();
    const auto& block = domain.blocks()[0];
    const ElementId<2> element_id{0};
    const ElementMap<2, Frame::Inertial> element_map{element_id, block};
    const Mesh<2> mesh{{npoints,3}, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered};
    const auto xi = logical_coordinates(mesh);
    const auto x = element_map(xi);
    const auto inv_jacobian = element_map.inv_jacobian(xi);
    const auto& r_star = get<0>(x);
    const auto& theta = get<1>(x);
    CAPTURE(min(r_star));
    CAPTURE(max(r_star));
    CAPTURE(min(theta));
    CAPTURE(max(theta));

    std::ofstream output_data("/home/cillian/debug/GRNewCircularEffectiveSource/spectre_source_off_particle.dat");
    output_data << 
      "i \t r_star \t theta \t scalar_eqn_re \t scalar_eqn_im \t effsource_re \t effsource_im \n";

    const auto circular_orbit = CircularOrbit{1., 0.5, 20.,3};
    CAPTURE(circular_orbit.puncture_position());
    const auto background =
        circular_orbit.variables(x, CircularOrbit::background_tags{});
    const auto& alpha = get<Tags::Alpha>(background);
    const auto& beta = get<Tags::Beta>(background);
    const auto& gamma_rstar = get<Tags::GammaRstar>(background);
    const auto& gamma_theta = get<Tags::GammaTheta>(background);
    const auto vars = circular_orbit.variables(x, CircularOrbit::source_tags{});
    const auto& singular_field = get<Tags::SingularField>(vars);
    const auto& deriv_singular_field = get<
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
        vars);
    const auto& effective_source = get<::Tags::FixedSource<Tags::MMode>>(vars);

    Variables<
        tmpl::list<::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>>
        fluxes{mesh.number_of_grid_points()};
    auto& flux_singular_field =
        get<::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>(
            fluxes);
    GrSelfForce::Fluxes::apply(make_not_null(&flux_singular_field), alpha, {},
                               deriv_singular_field);
    auto divs = divergence(fluxes, mesh, inv_jacobian);
    auto& scalar_eqn = get<::Tags::div<
        ::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>>(divs);
    for (size_t i = 0; i < scalar_eqn.size(); ++i) {
      scalar_eqn[i] *= -1.;
    }
    GrSelfForce::Sources::apply(make_not_null(&scalar_eqn), beta, gamma_rstar,
                                gamma_theta, singular_field,
                                flux_singular_field);
    for (size_t i = 0; i < scalar_eqn.size(); ++i) {
      for(size_t j=0; j < scalar_eqn[i].size(); j++)
      {
        output_data << i <<  "\t" << get<0>(x)[j] << "\t" << get<1>(x)[j] 
          << "\t" << scalar_eqn[i][j].real() << "\t" << scalar_eqn[i][j].imag()
          << "\t" << -effective_source[i][j].real() << "\t" << -effective_source[i][j].imag()
          << "\n";
      }
      // std::cout << scalar_eqn[i]/(-effective_source[i]) << "\n";
    }
}
} //namespace GrSelfForce::AnalyticData
