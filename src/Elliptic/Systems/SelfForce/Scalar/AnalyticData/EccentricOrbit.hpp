// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <complex>
#include <cstddef>
extern "C" {
    #include <effsource_equatorial.h>
}
#include <limits>
#include <optional>
#include <pup.h>
#include <vector>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <effsource_equatorial.hpp>

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
#include "Elliptic/Systems/SelfForce/Scalar/AnalyticData/SelfForceBackground.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "DataStructures/TaggedTuple.hpp"

namespace ScalarSelfForce::AnalyticData {

namespace EccentricOrbit_detail {
/*!
 * \brief Number of azimuthal samples for one time sample of the numerical
 * m-mode projection.
 *
 * \details The trapezoidal sum over the field azimuth converges like
 * \f$\rho^{-(N_\phi - m)}\f$ with
 *
 * \f{align*}{
 * \log\rho = \log\left(1 + 2 C_1 + 2 \sqrt{C_1 (1 + C_1)}\right),
 * \quad
 * C_1 = \frac{\alpha_{20}\,\delta r^2 + \alpha_{02}\,\delta\theta^2}{\beta},
 * \f}
 *
 * where \f$C_1\f$ measures the separation of the field point from the particle
 * and \f$\alpha_{20}\f$, \f$\alpha_{02}\f$ and \f$\beta\f$ are the puncture
 * coefficients of the particle at that time. The function returns
 * `effsource_equatorial_numerical_num_phi`, the smallest power of two that is
 * at least \f$\max(64, \max(m, 20) + 46 / \log\rho, 5 m / 2)\f$. For
 * \f$m \le 20\f$ and \f$C_1\f$ from \f$10^{-9}\f$ to 1 the quadrature error
 * of the m-mode of the effective source is then below \f$10^{-14}\f$ of the
 * integral of its absolute value over \f$\phi\f$.
 *
 * It is an error if \f$C_1 \le 0\f$ (the field point is on the particle) or if
 * the result exceeds \f$2^{22}\f$.
 */
size_t phi_points_for_sample(double c1, int m_mode_number);

/// The m-modes of the effective source and of the singular field and its
/// derivatives at one field point for one particle position
struct MModes {
  std::complex<double> effective_source{};
  std::complex<double> singular_field{};
  std::complex<double> deriv_r_singular_field{};
  std::complex<double> deriv_theta_singular_field{};
};

/*!
 * \brief The azimuthal node values of the numerical m-mode projection for one
 * \f$|m|\f$, shared by the field points and time samples of one call of
 * `EccentricOrbit::variables`.
 *
 * \details `get(num_phi)` rebuilds the table for `num_phi` nodes if it holds
 * fewer, and otherwise returns the present table, which
 * `effsource_equatorial_ctx_numerical_src_mode` reads with a stride. All node
 * counts are powers of two, so every count divides the largest one.
 */
class PhiTable {
 public:
  explicit PhiTable(int m_mode_number);
  PhiTable(const PhiTable&) = delete;
  PhiTable& operator=(const PhiTable&) = delete;
  PhiTable(PhiTable&&) = delete;
  PhiTable& operator=(PhiTable&&) = delete;
  ~PhiTable();

  int abs_m_mode_number() const { return abs_m_mode_number_; }
  const effsource_equatorial_numerical_table& get(size_t num_phi);

 private:
  int abs_m_mode_number_;
  effsource_equatorial_numerical_table table_{};
};

/*!
 * \brief The m-modes of the effective source and of the singular field and its
 * derivatives at the field point \f$(r, \theta)\f$ for the particle in `ctx`.
 *
 * \details The modes are
 *
 * \f{align*}{
 * f_m = \int_0^{2\pi} f(r, \theta, \phi)\, e^{-i m \phi}\, d\phi,
 * \f}
 *
 * where \f$f\f$ is the effective source, the singular field, or the derivative
 * of the singular field with respect to \f$r\f$ or \f$\theta\f$ at fixed
 * \f$t\f$ and \f$\phi\f$, all in Boyer-Lindquist coordinates. They are
 * trapezoidal sums over `phi_points` azimuthal samples, or over
 * `phi_points_for_sample` samples if `phi_points` has no value, centred on the
 * particle and evaluated from the coefficients of
 * `effsource_equatorial_ctx_prepare` (see
 * `effsource_equatorial_ctx_numerical_src_mode` and
 * `effsource_equatorial_ctx_numerical_field_modes`). `phi_table` must have
 * been constructed for \f$|m|\f$.
 *
 * The singular field and its derivatives are only computed if `singular_field`
 * is true, and are zero otherwise. It is an error if the field point is on the
 * particle or if `phi_points` is not a power of two of at least 4.
 */
MModes m_modes(gsl::not_null<PhiTable*> phi_table,
               const effsource_equatorial_ctx& ctx, double r, double theta,
               int m_mode_number, bool singular_field,
               std::optional<size_t> phi_points);
}  // namespace EccentricOrbit_detail

class EccentricOrbit : public SelfForceBackground,
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
  struct PenetratingHorizon {
    static constexpr Options::String help =
        "If 'False', use tortoise radial coordinate where the Kerr horizon is "
        "at negative infinity. If 'True', use Boyer-Lindquist radial "
        "coordinate where the Kerr horizon is at r_+.";
    using type = bool;
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
  struct PhiPoints {
    static constexpr Options::String help =
        "Number of azimuthal samples per time sample in the numerical m-mode "
        "projection of the effective source, the puncture and its "
        "derivatives. 'Auto' chooses the number for each time sample and "
        "field point from the separation of the field point and the "
        "particle (see EccentricOrbit_detail::phi_points_for_sample). A "
        "number, which must be a power of two, uses that fixed count for "
        "every sample.";
    using type = Options::Auto<size_t>;
  };
  using options =
      tmpl::list<BlackHoleMass, BlackHoleSpin, SemiLatusRectum, Eccentricity,
                 MModeNumber, NModeNumber, HyperboloidalSlicingTransitions,
                 PenetratingHorizon, ImposeEquatorialSymmetry, TimePoints,
                 PhiPoints>;
  static constexpr Options::String help =
      "Quasicircular orbit of a scalar point charge in Kerr spacetime";

  EccentricOrbit() = default;
  EccentricOrbit(const EccentricOrbit&) = delete;
  EccentricOrbit& operator=(const EccentricOrbit&) = delete;
  EccentricOrbit(EccentricOrbit&&) = default;
  EccentricOrbit& operator=(EccentricOrbit&&) = default;
  ~EccentricOrbit() override {
      for(auto* ctx : source_coefficients)
      {
          effsource_equatorial_free(ctx);
      }
  }

  // Modify for Eccentric orbit
  EccentricOrbit(
      double black_hole_mass, double black_hole_spin, double semi_latus_rectum,
      double eccentricity, int m_mode_number, int n_mode_number,
      std::optional<std::array<double, 4>> hyperboloidal_slicing_transitions,
      bool penetrating_horizon, bool impose_equatorial_symmetry,
      size_t time_points, std::optional<size_t> phi_points);

  explicit EccentricOrbit(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(EccentricOrbit);

  tnsr::I<double, 2> puncture_position() const override;
  double omega_phi() const override { return Omega_phi; }
  double omega_r() const override { return Omega_r; }
  double black_hole_mass() const override { return black_hole_mass_; }
  double black_hole_spin() const override { return black_hole_spin_; }
  double semi_latus_rectum() const override { return semi_latus_rectum_; }
  double eccentricity() const override { return eccentricity_; }
  double orbital_radius() const override { return semi_latus_rectum_; }
  int m_mode_number() const override { return m_mode_number_; }
  int n_mode_number() const override { return n_mode_number_ ; }
  size_t time_points() const { return time_points_; }
  std::optional<size_t> phi_points() const { return phi_points_; }
  std::optional<std::array<double, 4>> hyperboloidal_slicing_transitions()
      const {
    return hyperboloidal_slicing_transitions_;
  }
  bool penetrating_horizon() const override { return penetrating_horizon_; }
  bool impose_equatorial_symmetry() const override {
    return impose_equatorial_symmetry_;
  }

  using background_tags = tmpl::list<Tags::Alpha, Tags::Beta, Tags::Gamma>;
  using source_tags = tmpl::list<
      ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
      ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
      Tags::BoyerLindquistRadius>;

  // Background
  tuples::tagged_tuple_from_typelist<background_tags> variables(
      const tnsr::I<DataVector, 2>& x, background_tags /*meta*/) const override;

  // Initial guess
  static tuples::TaggedTuple<Tags::MMode> variables(
      const tnsr::I<DataVector, 2>& x, tmpl::list<Tags::MMode> /*meta*/);

  // Fixed sources
  tuples::tagged_tuple_from_typelist<source_tags> variables(
      const tnsr::I<DataVector, 2>& x, bool on_worldtube_boundary,
      source_tags /*meta*/) const override;

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 2>& x, const Mesh<2>& /*mesh*/,
      const InverseJacobian<DataVector, 2, Frame::ElementLogical,
                            Frame::Inertial>& /*inv_jacobian*/,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables(x, tmpl::list<RequestedTags...>{});
  }

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
  // Effective source coefficients for the time points in the first half of
  // the radial period, `t_values[0]` to `t_values[time_points / 2]`. The
  // second half follows from time-reversal symmetry.
  std::vector<effsource_equatorial_ctx*> source_coefficients;

 private:
  friend bool operator==(const EccentricOrbit& lhs, const EccentricOrbit& rhs);

  void create_source_coefficients();

  double black_hole_mass_{std::numeric_limits<double>::signaling_NaN()};
  double black_hole_spin_{std::numeric_limits<double>::signaling_NaN()};
  double semi_latus_rectum_{std::numeric_limits<double>::signaling_NaN()};
  double eccentricity_{std::numeric_limits<double>::signaling_NaN()};
  double orbital_radius_{std::numeric_limits<double>::signaling_NaN()};
  int m_mode_number_{};
  int n_mode_number_{};
  std::optional<std::array<double, 4>> hyperboloidal_slicing_transitions_{};
  bool penetrating_horizon_{false};
  bool impose_equatorial_symmetry_{false};
  size_t time_points_{};
  std::optional<size_t> phi_points_{};
  };

bool operator!=(const EccentricOrbit& lhs, const EccentricOrbit& rhs);

}  // namespace ScalarSelfForce::AnalyticData
