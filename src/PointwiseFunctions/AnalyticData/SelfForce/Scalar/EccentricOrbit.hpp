// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <pup.h>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/FirstOrderSystem.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ScalarSelfForce::AnalyticData {

class EccentricOrbit : public elliptic::analytic_data::Background,
                      public elliptic::analytic_data::InitialGuess {
 public:
  struct BlackHoleMass {
    static constexpr Options::String help =
        "Kerr mass parameter 'M' of the black hole";
    using type = double;
  };
  struct BlackHoleSpin {
    static constexpr Options::String help =
        "Kerr dimensionless spin parameter 'chi' of the black hole";
    using type = double;
  };
  struct SemiLatusRectum{
    static constexpr Options::String help =
        "The semilatus rectum p of the eccentric orbit";
    using type = double;
  };
  struct Eccentricity{
    static constexpr Options::String help =
        "The eccentricity of the orbit";
    using type = double;
  };
  struct MModeNumber {
    static constexpr Options::String help =
        "Azimuthal mode number 'm' of the scalar field";
    using type = int;
  };
  struct NModeNumber {
    static constexpr Options::String help =
        "Radial mode number 'n' of the scalar field";
    using type = int;
  };
  struct HyperboloidalSlicingTransitions {
    static constexpr Options::String help =
        "Enable hyperboloidal slicing by specifying the transition points for "
        "the boost function. The boost function transitions from -1 to zero "
        "between the first two points and from zero to 1 between the last "
        "two points. The effective source can only be evaluated where the "
        "boost function is zero, so the regularized region must be between "
        "the second and third points.";
    using type = Options::Auto<std::array<double, 4>, Options::AutoLabel::None>;
  };
  struct ImposeEquatorialSymmetry {
    static constexpr Options::String help =
        "Impose symmetry across the equatorial plane by using cos(theta)^2 "
        "as the angular coordinate instead of cos(theta). This means the "
        "domain should span [0, 1] instead of [-1, 1].";
    using type = bool;
  };
  struct TimePoints {
    static constexpr Options::String help =
      "Specify the number of time points at which to calculate the coordinates "
      "of the particle. This also amounts to choosing the number of time "
      "points to use in the time series data for the m mode effective source "
      "which is used to calculate the n-modes of the fixed sources";
    using type = size_t;
  };
  using options =
      tmpl::list<BlackHoleMass, BlackHoleSpin, SemiLatusRectum, Eccentricity,
        MModeNumber, NModeNumber, HyperboloidalSlicingTransitions,
        ImposeEquatorialSymmetry, TimePoints>;
  static constexpr Options::String help =
      "Quasicircular orbit of a scalar point charge in Kerr spacetime";

  EccentricOrbit() = default;
  EccentricOrbit(const EccentricOrbit&) = default;
  EccentricOrbit& operator=(const EccentricOrbit&) = default;
  EccentricOrbit(EccentricOrbit&&) = default;
  EccentricOrbit& operator=(EccentricOrbit&&) = default;
  ~EccentricOrbit() override = default;

  // Modify for Eccentric orbit
  EccentricOrbit(
      double black_hole_mass, double black_hole_spin, double semi_latus_rectum,
      double eccentricity, int m_mode_number, int n_mode_number,
      std::optional<std::array<double, 4>> hyperboloidal_slicing_transitions,
      bool impose_equatorial_symmetry, size_t time_points);

  explicit EccentricOrbit(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(EccentricOrbit);

  tnsr::I<double, 2> puncture_position() const;
  double black_hole_mass() const { return black_hole_mass_; }
  double black_hole_spin() const { return black_hole_spin_; }
  double semi_latus_rectum() const { return semi_latus_rectum_; }
  double eccentricity() const { return eccentricity_; }
  int m_mode_number() const { return m_mode_number_; }
  int n_mode_number() const { return n_mode_number_ ; }
  size_t time_points() const { return time_points_; }
  std::optional<std::array<double, 4>> hyperboloidal_slicing_transitions()
      const {
    return hyperboloidal_slicing_transitions_;
  }
  bool impose_equatorial_symmetry() const {
    return impose_equatorial_symmetry_;
  }

  using evolution_tags = tmpl::list<
    Tags::EffectiveSourceEvolution,
    Tags::SingularFieldEvolution,
    Tags::DerivSingularFieldEvolution
    >;
  using background_tags =
      typename ScalarSelfForce::FirstOrderSystem::background_fields;
  using source_tags = tmpl::list<
      ::Tags::FixedSource<Tags::MMode>,
      Tags::SingularField,
      ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
      Tags::BoyerLindquistRadius>;

  // Evolve the m-mode puncture, derivatives and effective source
  void evolve_sources(
      const tnsr::I<DataVector, 2>& x, evolution_tags);

  // Background
  tuples::tagged_tuple_from_typelist<background_tags> variables(
      const tnsr::I<DataVector, 2>& x, background_tags /*meta*/) const;

  // Initial guess
  static tuples::TaggedTuple<Tags::MMode> variables(
      const tnsr::I<DataVector, 2>& x, tmpl::list<Tags::MMode> /*meta*/);

  // Fixed sources
  tuples::tagged_tuple_from_typelist<source_tags> variables(
      const tnsr::I<DataVector, 2>& x, source_tags /*meta*/) const;

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 2>& x, const Mesh<2>& /*mesh*/,
      const InverseJacobian<DataVector, 2, Frame::ElementLogical,
                            Frame::Inertial>& /*inv_jacobian*/,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables(x, tmpl::list<RequestedTags...>{});
  }

  ComplexDataVector compute_n_mode(
      std::vector<Scalar<ComplexDataVector>>& time_series_data,
      size_t num_points,
      double integral_tolerance) const;

  tnsr::i<ComplexDataVector, 2> compute_n_mode(
      std::vector<tnsr::i<ComplexDataVector, 2>>& time_series_data,
      size_t num_points,
      double integral_tolerance) const;

  void compute_trajectory(size_t time_points);

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override;

  double energy;
  double angular_momentum;
  double Omega_r;
  double Omega_phi;
  std::vector<double> r_of_t;
  std::vector<double> phi_of_t;
  std::vector<double> t_values;
  std::vector<double> u_r;

 private:
  friend bool operator==(const EccentricOrbit& lhs, const EccentricOrbit& rhs);
  double black_hole_mass_{std::numeric_limits<double>::signaling_NaN()};
  double black_hole_spin_{std::numeric_limits<double>::signaling_NaN()};
  double semi_latus_rectum_{std::numeric_limits<double>::signaling_NaN()};
  double eccentricity_{std::numeric_limits<double>::signaling_NaN()};
  int m_mode_number_{};
  int n_mode_number_{};
  std::optional<std::array<double, 4>> hyperboloidal_slicing_transitions_{};
  bool impose_equatorial_symmetry_{false};
  size_t time_points_{};
  };

bool operator!=(const EccentricOrbit& lhs, const EccentricOrbit& rhs);

}  // namespace ScalarSelfForce::AnalyticData'numToStr/Comment.nvim',
