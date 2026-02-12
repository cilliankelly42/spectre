# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarSelfForceAmrCriteria)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  RefineAtPuncture.cpp
  )

target_compile_options(
  ${LIBRARY}
  PRIVATE
  -g
  -O0
  -fno-inline
)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  RefineAtPuncture.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Amr
  AmrCriteria
  DataStructures
  DomainStructure
  Elliptic
  InitialDataUtilities
  Options
  Parallel
  Serialization
  Utilities
  PRIVATE
  Domain
  ScalarSelfForceAnalyticData
  )
