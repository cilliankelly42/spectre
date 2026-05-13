// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestingFramework.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <iostream>
#include <unordered_set>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Element.hpp"
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

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarSelfForce.DomainTest",
                  "[PointwiseFunctions][Unit]") {
  // This test checks both the self-force equations and the effective source
  // computation in a very robust way: it ensures that the elliptic operator
  // applied to the singular field gives the effective source.
  // This is done numerically on a rectangular grid in (r_*, cos(theta)) near
  // the puncture.

  // Set up a domain
  const double costheta_offset = -0.1;
  const double delta_costheta = 0.2;
  const double rstar_offset = 11.;
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
  const auto& block_map = block.stationary_map();
  tnsr::I<double, 2, Frame::BlockLogical> lower_corner{ {-1.0, -1.0} };
  tnsr::I<double, 2, Frame::BlockLogical> upper_corner{ {1.0, 1.0} };
  const auto global_lower = block_map(lower_corner);
  const auto global_upper = block_map(upper_corner); 
  std::cout << "Global upper: r " << get<0>(global_upper) << "\n";
  std::cout << "Global upper: costheta " << get<1>(global_upper) << "\n";
  std::cout << "Global lower: r " << get<0>(global_lower) << "\n";
  std::cout << "Global lower: costheta " << get<1>(global_lower) << "\n";
  std::cout << "block is: " << block << "\n";
  std::cout << "Block id is " << block.id() << "\n";

  const ElementId<2> element_id1{0};
  const ElementId<2> element_id2{1};
  const ElementId<2> element_id3{2};
  const ElementId<2> element_id4{3};

  const OrientationMap<2> block_orientation = OrientationMap<2>::create_aligned();
  std::cout << "block external boundaries: " << block.external_boundaries() << "\n";
  std::cout << "block topologies: " << block.topologies() << "\n";

  std::unordered_set<ElementId<2>> neighbours_id1{element_id2};
  std::unordered_set<ElementId<2>> neighbours_id2{element_id1};

  Neighbors<2> neighbors_1{neighbours_id1, block_orientation};
  Neighbors<2> neighbors_2{neighbours_id2, block_orientation};

  DirectionMap<2, Neighbors<2>> direction_map_1;
  DirectionMap<2, Neighbors<2>> direction_map_2;
  direction_map_1[Direction<2>::upper_xi()] = neighbors_2;
  direction_map_2[Direction<2>::lower_xi()] = neighbors_1;

  const Element<2> element_1{element_id1, direction_map_1};
  const Element<2> element_2{element_id2, direction_map_2};

  std::cout << "Element 1 is: " << element_1 << "\n";
  std::cout << "Element 2 is: " << element_2 << "\n";
  std::cout << "Element 1 has neighbors: " << element_1.number_of_neighbors()<< "\n";
  std::cout << "And external boundary directions: " << element_1.external_boundaries()<< "\n";
  std::cout << "And internal boundary directions: " << element_1.internal_boundaries() << "\n";
  std::cout << "Element 2 has neighbors: " << element_2.number_of_neighbors()<< "\n";
  std::cout << "And external boundary directions: " << element_2.external_boundaries()<< "\n";
  std::cout << "And internal boundary directions: " << element_2.internal_boundaries() << "\n";
  std::cout << "The face type for element 1 is: " << element_1.face_types() << "\n";
  std::cout << "The face type for element 2 is: " << element_2.face_types() << "\n";

  const ElementMap<2, Frame::Inertial> element_map{element_id1, block};
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
}
}// namespace ScalarSelfForce::AnalyticData
