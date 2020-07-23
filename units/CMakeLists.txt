cmake_minimum_required(VERSION 3.10)

include(GoogleTest)
option(INSTALL_GTEST OFF)
option(BUILD_GMOCK OFF)

project(smartiestests LANGUAGES CXX )

set(SMARTIES_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../source/")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/googletest")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

function (add_test_executable dirName nodes)
  # Find sources
  file(GLOB SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/${dirName}/*.cpp")

  string(REGEX REPLACE "/" "_" execNameId ${dirName})

  set(EXEC_NAME "test_${execNameId}")

  add_executable(${EXEC_NAME} ${SOURCES})
  #target_include_directories(${EXEC_NAME} PUBLIC "$ENV{SMARTIES_ROOT}/include/")
  target_include_directories(${EXEC_NAME} PUBLIC "$ENV{SMARTIES_ROOT}/source/")
  target_link_directories(${EXEC_NAME} PUBLIC "$ENV{SMARTIES_ROOT}/lib/")
  target_link_libraries(${EXEC_NAME} PUBLIC smarties OpenMP::OpenMP_CXX MPI::MPI_CXX gtest)
  target_compile_features(${EXEC_NAME} PUBLIC cxx_std_14)

  add_test(NAME ${EXEC_NAME}
           COMMAND mpirun -n ${nodes} ./${EXEC_NAME})
endfunction()

add_test_executable(Core 1)
add_test_executable(Math 1)
add_test_executable(Network 1)
