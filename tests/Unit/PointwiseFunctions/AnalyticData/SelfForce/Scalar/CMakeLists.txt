# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ScalarSelfForceAnalyticData")

set(LIBRARY_SOURCES 
    Test_EccentricOrbit.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

#target_compile_options(
#  ${LIBRARY}
#  PRIVATE
#  -g
#  -O0
# -fno-inline
#)

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Utilities
  ScalarSelfForce
  ScalarSelfForceAnalyticData
  Domain
  DomainCreators
  LinearOperators
  KerrGeodesicsC
  Integration 
  )
