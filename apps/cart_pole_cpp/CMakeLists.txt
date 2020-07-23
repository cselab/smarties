cmake_minimum_required(VERSION 3.2.0)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
  cmake_policy(SET CMP0074 OLD)
endif()

if (ENABLE_SANITIZER)
  set(san_opt "-fsanitize=undefined")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${san_opt} -g")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${san_opt}")
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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../)
add_executable(exec cart-pole.cpp)
target_include_directories(exec PUBLIC "$ENV{SMARTIES_ROOT}/include/")
target_link_directories(exec PUBLIC "$ENV{SMARTIES_ROOT}/lib/")
target_link_libraries(exec PUBLIC smarties OpenMP::OpenMP_CXX MPI::MPI_CXX)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
target_link_libraries(exec PUBLIC atomic)
endif()

target_compile_features(exec PUBLIC cxx_std_14)
target_compile_options(exec PRIVATE
  $<$<CONFIG:Debug>:${cxx_debug_flags}>
  $<$<CONFIG:Release>:${cxx_release_flags}>
  )
