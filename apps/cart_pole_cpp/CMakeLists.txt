cmake_minimum_required(VERSION 3.2.0)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
  cmake_policy(SET CMP0074 OLD)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release"
    FORCE)
endif()

project(exec LANGUAGES CXX)

set(cxx_warning_flags -Wall -Wextra -Wshadow -Werror)
set(cxx_release_flags -O3 -g)
set(cxx_debug_flags -O0 -g)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT TARGET MPI::MPI_CXX)
  add_library(MPI::MPI_CXX IMPORTED INTERFACE)
  
  set_property(TARGET MPI::MPI_CXX
    PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
  set_property(TARGET MPI::MPI_CXX
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")
  set_property(TARGET MPI::MPI_CXX
    PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
endif()

if(NOT TARGET OpenMP::OpenMP_CXX)
  find_package(Threads REQUIRED)
  add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
  set_property(TARGET OpenMP::OpenMP_CXX
    PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
  # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
  set_property(TARGET OpenMP::OpenMP_CXX
    PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../)
add_executable(exec cart-pole.cpp)
target_include_directories(exec PUBLIC "$ENV{SMARTIES_ROOT}/include/")
target_link_directories(exec PUBLIC "$ENV{SMARTIES_ROOT}/lib/")
target_link_libraries(exec PUBLIC smarties OpenMP::OpenMP_CXX MPI::MPI_CXX)


target_compile_features(exec PUBLIC cxx_std_14)
target_compile_options(exec PRIVATE
  $<$<CONFIG:Debug>:${cxx_debug_flags}>
  $<$<CONFIG:Release>:${cxx_release_flags}>
  )
