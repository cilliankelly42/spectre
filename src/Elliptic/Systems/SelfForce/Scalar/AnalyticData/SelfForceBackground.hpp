// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Elliptic/Systems/SelfForce/Scalar/FirstOrderSystem.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

namespace ScalarSelfForce::AnalyticData{
    class SelfForceBackground : public elliptic::analytic_data::Background {
        protected:
            SelfForceBackground() = default;

        public:
            ~SelfForceBackground() override = default;

            virtual double black_hole_mass() const = 0;
            virtual double black_hole_spin() const = 0;
            virtual double semi_latus_rectum() const = 0;
            virtual double eccentricity() const = 0;
            virtual double orbital_radius() const = 0;
            virtual int m_mode_number() const = 0;
            virtual int n_mode_number() const = 0;
            virtual double omega_phi() const = 0;
            virtual double omega_r() const = 0;
            virtual bool penetrating_horizon() const = 0;

            using background_tags =
              typename ScalarSelfForce::FirstOrderSystem::background_fields;
            using source_tags = tmpl::list<
                  ::Tags::FixedSource<Tags::MMode>,
                  Tags::SingularField,
                  ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
                  Tags::BoyerLindquistRadius>;

            // Background
            virtual tuples::tagged_tuple_from_typelist<background_tags> variables(
              const tnsr::I<DataVector, 2>& x, background_tags /*meta*/) const = 0;

            // Fixed sources
            virtual tuples::tagged_tuple_from_typelist<source_tags> variables(
              const tnsr::I<DataVector, 2>& x, bool on_worldtube_boundary,
              source_tags /*meta*/) const = 0;

            virtual tnsr::I<double, 2> puncture_position() const = 0;

            /// \cond
            explicit SelfForceBackground(CkMigrateMessage* msg) : PUP::able(msg) {}
            WRAPPED_PUPable_abstract(SelfForceBackground);
            /// \endcond
    };
} //namespace ScalarSelfForce::AnalyticData
