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
#include "PointwiseFunctions/AnalyticData/SelfForce/Scalar/CircularOrbit.hpp"
#include "PointwiseFunctions/AnalyticData/SelfForce/Scalar/EccentricOrbit.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"
#include "Utilities/MakeArray.hpp"

namespace ScalarSelfForce::AmrCriteria {

std::array<amr::Flag, 2> RefineAtPuncture::impl(
    const elliptic::analytic_data::Background& background,
    const Domain<2>& domain, const ElementId<2>& element_id) {
    /* const auto puncture_position =
        dynamic_cast<const ScalarSelfForce::AnalyticData::CircularOrbit&>(
            background)
            .puncture_position(); */

    //Just use background here.. No dynamic cast, then setup a self force 
    //version of the elliptic background class where you define virtual 
    //functions for all of the functions in each class. 

    const auto& eccentric_orbit = 
        dynamic_cast<const ScalarSelfForce::AnalyticData::EccentricOrbit&>(
                background
                );


    // const auto puncture_position = eccentric_orbit.puncture_position();

    /* const auto puncture_position =
        dynamic_cast<const ScalarSelfForce::AnalyticData::EccentricOrbit&>(
            background)
            .puncture_position(); */

    using point = boost::geometry::model::d2::point_xy<double>;
    using box = boost::geometry::model::box<point>; 
    const double M = eccentric_orbit.black_hole_mass();
    const double r_plus = M * (1. + sqrt(1. - square(eccentric_orbit.black_hole_spin())));
    const double r_min = eccentric_orbit.semi_latus_rectum()/(1 + eccentric_orbit.eccentricity());
    const double r_max = eccentric_orbit.semi_latus_rectum()/(1 - eccentric_orbit.eccentricity());
    const double r_star_min = 
    gr::tortoise_radius_from_boyer_lindquist_minus_r_plus(
        r_min - r_plus, M, eccentric_orbit.black_hole_spin());
    const double r_star_max = 
    gr::tortoise_radius_from_boyer_lindquist_minus_r_plus(
      r_max - r_plus, M, eccentric_orbit.black_hole_spin());

    box puncture_box{{r_star_min, 0}, {r_star_max,0}};
    //Make the puncture box here

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
/*
    // Split (h-refine) the element if it contains the puncture
    const auto& block = domain.blocks()[element_id.block_id()];
    // Check if the puncture is in the block
    const auto block_logical_coords =
        block_logical_coordinates_single_point(puncture_position, block);
    if (not block_logical_coords.has_value()) {
      return make_array<2>(amr::Flag::DoNothing);
    }
    if (not element_logical_coordinates(*block_logical_coords, element_id)) {
      return make_array<2>(amr::Flag::DoNothing);
    }
    return make_array<2>(amr::Flag::Split);
} */

PUP::able::PUP_ID RefineAtPuncture::my_PUP_ID = 0;  // NOLINT

}  // namespace ScalarSelfForce::AmrCriteria
