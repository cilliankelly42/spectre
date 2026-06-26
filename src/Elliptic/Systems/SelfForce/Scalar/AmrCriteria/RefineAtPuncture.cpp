// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/Scalar/AmrCriteria/RefineAtPuncture.hpp"

#include <array>
#include <cstddef>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/AnalyticData/CircularOrbit.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/AnalyticData/EccentricOrbit.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/AnalyticData/SelfForceBackground.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"
#include "Utilities/MakeArray.hpp"

namespace ScalarSelfForce::AmrCriteria {

std::array<amr::Flag, 2> RefineAtPuncture::impl(
    const elliptic::analytic_data::Background& background,
    const Domain<2>& domain, const ElementId<2>& element_id) {
    const auto& orbit = dynamic_cast<const ScalarSelfForce::AnalyticData::SelfForceBackground&>
        (background);

    //Create a bounding box to represent the particle 
    using point = boost::geometry::model::d2::point_xy<double>;
    using box = boost::geometry::model::box<point>; 
    const double M = orbit.black_hole_mass();
    const double r_plus = M * (1. + sqrt(1. - square(orbit.black_hole_spin())));
    double r_min = 0;
    double r_max = 0;
    double r_or_rstar_min = get<0>(orbit.puncture_position());
    double r_or_rstar_max = get<0>(orbit.puncture_position());
    // If the orbit is eccentric, need to create a line
    /* if(dynamic_cast<const ScalarSelfForce::AnalyticData::EccentricOrbit*>(&orbit))
    { */
    r_min = orbit.semi_latus_rectum()/(1 + orbit.eccentricity());
    r_max = orbit.semi_latus_rectum()/(1 - orbit.eccentricity());
    if(orbit.penetrating_horizon()){
        r_or_rstar_min = r_min;
        r_or_rstar_max = r_max;
    } else {
    r_or_rstar_min = 
        gr::tortoise_radius_from_boyer_lindquist_minus_r_plus(
            r_min - r_plus, M, orbit.black_hole_spin());
    r_or_rstar_max = 
        gr::tortoise_radius_from_boyer_lindquist_minus_r_plus(
          r_max - r_plus, M, orbit.black_hole_spin());
    }
    //}

    // Create a bounding box to represent the particle
    box puncture_box{{r_or_rstar_min, 0}, {r_or_rstar_max,0}};

    // Check if the block contains the puncture box
    const auto& block = domain.blocks()[element_id.block_id()];
    const auto& block_map = block.stationary_map();
    tnsr::I<double, 2, Frame::BlockLogical> block_lower_corner{ {-1.0, -1.0} };
    tnsr::I<double, 2, Frame::BlockLogical> block_upper_corner{ {1.0, 1.0} };
    const auto global_lower = block_map(block_lower_corner);
    const auto global_upper = block_map(block_upper_corner);
    box block_box{
        {get<0>(global_lower), get<1>(global_lower)},
        {get<0>(global_upper), get<1>(global_upper)}
    };

    if(not boost::geometry::intersects(puncture_box, block_box)){
       return make_array<2>(amr::Flag::DoNothing);
    }

    // Check if the element intersects the puncture box (need to get the
    // coordinates of the element and do a comparison)
    const auto lower_coordinate_0 = 
      -1 + 2 * (element_id.segment_id(0).index())/pow(2,element_id.refinement_levels()[0]);
    const auto upper_coordinate_0= 
      -1 + 2 * (1 + element_id.segment_id(0).index())/pow(2,element_id.refinement_levels()[0]);
    const auto lower_coordinate_1 = 
      -1 + 2 * (element_id.segment_id(1).index())/pow(2,element_id.refinement_levels()[1]);
    const auto upper_coordinate_1 =
      -1 + 2 * (1 + element_id.segment_id(1).index())/pow(2,element_id.refinement_levels()[1]);

    tnsr::I<double, 2, Frame::BlockLogical> element_lower_corner{{lower_coordinate_0, lower_coordinate_1}};
    tnsr::I<double, 2, Frame::BlockLogical> element_upper_corner{{upper_coordinate_0, upper_coordinate_1}};
    const auto inertial_element_lower = block_map(element_lower_corner);
    const auto inertial_element_upper = block_map(element_upper_corner);
    box element_box{
        {get<0>(inertial_element_lower), get<1>(inertial_element_lower)},
        {get<0>(inertial_element_upper), get<1>(inertial_element_upper)}
    };
    if(not boost::geometry::intersects(puncture_box, element_box)){
        return make_array<2>(amr::Flag::DoNothing);
    }

    return make_array<2>(amr::Flag::Split);
}

PUP::able::PUP_ID RefineAtPuncture::my_PUP_ID = 0;  // NOLINT

}  // namespace ScalarSelfForce::AmrCriteria
