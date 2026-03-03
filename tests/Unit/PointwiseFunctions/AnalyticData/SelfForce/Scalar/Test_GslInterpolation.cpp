#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <istream>
#include <limits>
#include <random>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Integration/GslQuadAdaptive.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpline.hpp"
#include "Utilities/Math.hpp"

namespace ScalarSelfForce::AnalyticData {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarSelfForce.TestSpline",
                  "[PointwiseFunctions][Unit]") {
    /* std::vector<double> times(500);
    std::vector<double> data_to_interpolate(500);
    std::ifstream input_data_to_interpolate(
            "/home/user/spline_testing/data_to_interpolate.txt");
    std::string header;
    std::getline(input_data_to_interpolate, header);
    std::cout << "header is: " << header << "\n";
    for(size_t i=0; i < 500; i++)
    {
        input_data_to_interpolate >> times[i];
        input_data_to_interpolate >> data_to_interpolate[i];
    }
    input_data_to_interpolate.close();
    intrp::CubicSpline interpolant{times, data_to_interpolate};
    std::ofstream interpolated_data("/home/user/spline_testing/interpolated_data.txt");
    interpolated_data << "time" << "\t"<<"interpolant(t)"<< "\n";
    for(double t : times)
    {
        interpolated_data << t << "\t" 
            << interpolant(t) << "\n";
    }
    interpolated_data.close(); */
    /* const integration::GslQuadAdaptive<
        integration::GslIntegralType::StandardGaussKronrod>
        integration{5000};

    double integral = integration(
            [](double t){ return exp(std::complex<double>(0, t)); },
            0,
            2 * M_PI,
            0, 6, 1e-12
            );
    std::cout << integral << "\n"; */
}
} // Namespace ScalarSelfForce::AnalyticData
