set(LIBRARY EffectiveSource)

add_library(${LIBRARY} 
  kerr-circular.c 
  kerr-equatorial-ctx.c 
  kerr-equatorial-coeffs-ctx.c 
  kerr-equatorial-dtcoeffs-ctx.c 
  kerr-equatorial-dttcoeffs-ctx.c
)

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  GSL::gsl
  )

target_include_directories(${LIBRARY} SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
