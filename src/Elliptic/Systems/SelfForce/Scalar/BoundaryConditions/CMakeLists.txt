# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarSelfForceBoundaryConditions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Angular.cpp
  Sommerfeld.cpp
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
  Factory.hpp
  Angular.hpp
  Sommerfeld.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Elliptic
  Options
  Parallel
  Utilities
  )
