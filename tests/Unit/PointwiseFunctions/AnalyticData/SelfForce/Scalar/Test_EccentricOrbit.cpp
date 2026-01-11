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
#include "PointwiseFunctions/AnalyticData/SelfForce/Scalar/EccentricOrbit.hpp"
#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"
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

  
  
  /*
  variables function needs to return the evolution of the m_mode effective_source. Then each element of that array is the value of the m_mode_effective_source at the given value of t_values 
  Then loop over the t_values and for each t_value compute the resummation and look at the difference between that and the value of the effective source
  */

  int m_mode_number = 0;
  const auto eccentric_orbit = EccentricOrbit{1., 0.9, 10., 0.6 , m_mode_number, 0 /*n_mode number irrelevant here because object only created to call compute_trajectory()*/, {{-25., -5., 20., 40.}}, false};
  eccentric_orbit.compute_trajectory();

  // Create and fill an array of containing times at which to compute the resummation and compare with the original time series data
  std::vector<size_t> time_indices = {100,200,300,400,500}; // Can randomly generate N integers in range (0, eccentric_orbit.t_values.size() - 1) in the future for a random test
  std::vector<double> t_values_to_test(time_indices.size());
  for(size_t i=0; i < time_indices.size(); i++)
  {
    t_values_to_test[i] = eccentric_orbit.t_values[time_indices[i]];
  }

  // Initialise n_mode_array to be zeros
  Scalar<ComplexDataVector> n_mode_sum(get<0>(x).size(), std::complex<double>(0,0));
  /*
  std::vector<Scalar<ComplexDataVector>> n_mode_array(5 /*How many n_modes do you want to sum*/); 
  for(auto& complex_data_vector : n_mode_array)
  {
    get(complex_data_vector) = ComplexDataVector(get<0>(x).size(), std::complex<double>(0,0));
  }
  */
  // Compute n_mode resummation for each time and make sure difference is zero up to specified tolerance
  
  for(size_t t : time_indices)
  {
    for(size_t n_mode_number = 0; n_mode_number < n_mode_array.size(); n_mode_number++)
    {
      const auto eccentric_orbit = EccentricOrbit{1., 0.9, 10., 0.6 , m_mode_number, n_mode_number, {{-25., -5., 20., 40.}}, false};
      const auto vars = eccentric_orbit.variables(x, EccentricOrbit::source_tags{}); 
      const auto& n_mode = get<Tags::NMode>(vars);
      get(n_mode_sum) += 
        get(n_mode) * exp(std::complex<double>(0, - (n_mode_number * eccentric_orbit.Omega_r + m_mode_number * eccentric_orbit.Omega_phi) * eccentric_orbit.t_values[t]));
      get(n_mode_array[n_mode_number]) += 
        get(n_mode) * exp(std::complex<double>(0, - (n_mode_number * eccentric_orbit.Omega_r + m_mode_number * eccentric_orbit.Omega_phi) * eccentric_orbit.t_values[t]));
    }
    std::cout << get(n_mode_sum) << "\n\n";

    /*
      Compare with the current effective_source_evolution here 
    */

  }

  /*
  double r_plus = 1 + sqrt(1 - 0.9*0.9);
  std::cout << gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(get<0>(x), 1, 0.9)[30] + r_plus << "\n";
  std::cout << cos_theta[30] << "\n";
  std::cout << get(n_mode)[30] << "\n";
*/

  // Loop and sum over n-modes to check convergence to time series data within a certain tolerance 

  //std::vector<Scalar<ComplexDataVector>> n_mode_array(1); 
  //Scalar<ComplexDataVector> n_mode_summand(get<0>(x).size()); // n_mode_summand is an array storing Seff_nm* exp(-in OMega_r t) where each outer element in the vector is for each n from 0 to n_max
  /*
  for(size_t n_mode_number = 0; n_mode_number < n_mode_array.size(); n_mode_number++)
  {
    const auto eccentric_orbit = EccentricOrbit{1., 0.9, 10., 0.6 , m_mode_number, n_mode_number, {{-25., -5., 20., 40.}}, false};
    eccentric_orbit.compute_trajectory();
    const auto vars = eccentric_orbit.variables(x, EccentricOrbit::source_tags{}); 
    const auto& n_mode = get<Tags::NMode>(vars);
    auto n_mode_summand = [&n_mode, &n_mode_number, &eccentric_orbit](double t){ return get(n_mode) * exp(std::complex<double>(0, - n_mode_number * eccentric_orbit.Omega_r * t)); };
    //std::cout << n_mode_summand(2) << "\n\n";
    Scalar<ComplexDataVector>& test_n_mode_type = n_mode_summand(2);
    //n_mode_array[n_mode_number] = n_mode_summand(10); 
  }
  */

  //std::cout << get(n_mode_array[0]) * exp(std::complex<double>(0, )) << "\n";

  /*
  Scalar<ComplexDataVector> resummed_n_modes;
  get(resummed_n_modes).destructive_resize(get<0>(x).size());
  std::cout << get(resummed_n_modes) << "\n";
  for(size_t i=0; i < n_mode_array.size(); i++)
  {
    get(resummed_n_modes) += get(n_mode_array[i]);
  }
  std::cout << get(resummed_n_modes) << "\n";
*/
}
}