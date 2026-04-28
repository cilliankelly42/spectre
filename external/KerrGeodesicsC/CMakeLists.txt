set(LIBRARY KerrGeodesicsC)

add_library(${LIBRARY} korb.c)

find_package(FFTW)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  GSL::gsl
  FFTW::FFTW 
  m
  )

target_include_directories(${LIBRARY} SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
