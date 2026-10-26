set(LIBRARY EffectiveSource)

add_library(${LIBRARY} 
  kerr-circular.c 
  kerr-equatorial-ctx.c 
  kerr-equatorial-coeffs-ctx.c 
  kerr-equatorial-dtcoeffs-ctx.c 
  kerr-equatorial-dttcoeffs-ctx.c
  kerr-equatorial-numerical.c
  kerr-equatorial-point-ctx.c
)

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  GSL::gsl
  )

# The numerical m-mode projection evaluates the coefficients of the field point
# at every (grid point, time) sample and sums them over the azimuthal samples,
# and the singular field on the worldtube boundary and the reference in the
# unit test evaluate it at many azimuths, so these files dominate the run time
# even in Debug builds. They compile quickly with optimizations.
# The coefficient files need a lot of memory to optimize and only run once per
# time sample, so they keep the default flags.
set_source_files_properties(
  kerr-equatorial-ctx.c
  kerr-equatorial-numerical.c
  kerr-equatorial-point-ctx.c
  PROPERTIES COMPILE_OPTIONS "$<$<CONFIG:Debug>:-O2>"
  )

target_include_directories(${LIBRARY} SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
