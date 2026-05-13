// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/ComplexDataVector.hpp"
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
#include "Domain/BlockLogicalCoordinates.hpp"

namespace ScalarSelfForce::AnalyticData {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarSelfForce.ComputeNMode",
                  "[PointwiseFunctions][Unit]") {
  const auto eccentric_orbit = EccentricOrbit{1,0.5,10,0.1,1,0,{{1,2,3,4}},false,950};
  std::vector<double> times = eccentric_orbit.t_values;
  std::vector<Scalar<ComplexDataVector>> cos_vector;
  cos_vector.resize(eccentric_orbit.time_points());
  for(size_t i=0; i<eccentric_orbit.time_points(); i++)
  {
    get(cos_vector[i]).destructive_resize(1);
    get(cos_vector[i]) = 
      std::complex<double>(
          cos(times[i]/(4*times[times.size() - 1])),
          sin(times[i]/(4*times[times.size() - 1])));
  }
  std::cout << "Omega_r" <<eccentric_orbit.Omega_r << "\n";
  std::cout << "Omega_phi" << eccentric_orbit.Omega_phi << "\n";
  ComplexDataVector result = eccentric_orbit.compute_n_mode(cos_vector, 1, 1e-10);
  std::cout << "result" << result << "\n";
  }
} //namespace ScalarSelfForce::AnalyticData
