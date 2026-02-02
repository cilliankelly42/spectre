# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarSelfForceAnalyticData)

add_spectre_library(${LIBRARY})

target_compile_options(
  ${LIBRARY}
  PRIVATE
  -g
  -O0
  -fno-inline
)

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CircularOrbit.cpp
  EccentricOrbit.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CircularOrbit.hpp
  EccentricOrbit.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  EffectiveSource
  ErrorHandling
  GeneralRelativity
  Options
  Utilities
  KerrGeodesicsC
  )

add_subdirectory(Python)
