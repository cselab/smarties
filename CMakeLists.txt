# NOTE: Update `install_dependencies.sh` if updating the minimum required version.
cmake_minimum_required(VERSION 3.2)

project(smarties)

# Choose Release mode as default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# List of options

option(COMPILE_PY_SO "Compile Python bindings" ON)

# Smarties library

set(root_folder ${CMAKE_CURRENT_LIST_DIR})
set(source_dir ${root_folder}/source)
set(lib_output ${root_folder}/lib)

set(smarties_sources
    ${source_dir}/smarties/Learners/PPO.cpp
    ${source_dir}/smarties/Learners/MixedPG.cpp
    ${source_dir}/smarties/Learners/NAF.cpp
    ${source_dir}/smarties/Learners/DPG.cpp
    ${source_dir}/smarties/Learners/DQN.cpp
    ${source_dir}/smarties/Learners/ACER.cpp
    ${source_dir}/smarties/Learners/RACER.cpp
    ${source_dir}/smarties/Learners/CMALearner.cpp
    ${source_dir}/smarties/Learners/Learner_pytorch.cpp
    ${source_dir}/smarties/Learners/Learner_approximator.cpp
    ${source_dir}/smarties/Learners/AlgoFactory.cpp
    ${source_dir}/smarties/Learners/Learner.cpp

    ${source_dir}/smarties/ReplayMemory/Episode.cpp
    ${source_dir}/smarties/ReplayMemory/Sampling.cpp
    ${source_dir}/smarties/ReplayMemory/MemoryBuffer.cpp
    ${source_dir}/smarties/ReplayMemory/DataCoordinator.cpp
    ${source_dir}/smarties/ReplayMemory/MemoryProcessing.cpp

    ${source_dir}/smarties/Math/Math_tests.cpp
    ${source_dir}/smarties/Utils/Warnings.cpp
    ${source_dir}/smarties/Utils/Profiler.cpp
    ${source_dir}/smarties/Utils/StatsTracker.cpp
    ${source_dir}/smarties/Utils/DelayedReductor.cpp
    ${source_dir}/smarties/Settings/ExecutionInfo.cpp
    ${source_dir}/smarties/Settings/HyperParameters.cpp

    ${source_dir}/smarties/Core/Launcher.cpp
    ${source_dir}/smarties/Core/Master.cpp
    ${source_dir}/smarties/Core/Worker.cpp
    ${source_dir}/smarties/Core/StateAction.cpp

    ${source_dir}/smarties/Communicator.cpp
    ${source_dir}/smarties/Engine.cpp

    ${source_dir}/smarties/Network/Approximator.cpp
    ${source_dir}/smarties/Network/Builder.cpp
    ${source_dir}/smarties/Network/Network.cpp
    ${source_dir}/smarties/Network/Optimizer.cpp
    ${source_dir}/smarties/Network/CMA_Optimizer.cpp
)

set(smarties_core "libsmarties")
add_library(${smarties_core} SHARED ${smarties_sources})
set_property(TARGET ${smarties_core} PROPERTY PREFIX "") # remove the extra "lib" prefix

# compilation flags
target_compile_features(${smarties_core} PUBLIC cxx_std_14)

set(cxx_private_flags
     $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
          -Wall -Wno-pass-failed >
     $<$<CXX_COMPILER_ID:GNU>:
          -Wall -Wextra -pedantic -Wno-unused-parameter -Wno-maybe-uninitialized -fmax-errors=3  >)
target_compile_options(${smarties_core} PRIVATE ${cxx_private_flags})

target_compile_options(${smarties_core} PRIVATE
  $<$<CONFIG:Debug>:-O0 -g>
  $<$<CONFIG:Release>:-DNDEBUG -O3 -ffast-math>
  )

# include flags
# here interface since no internal source include the following directory
target_include_directories(${smarties_core} INTERFACE ${root_folder}/include)

# -fPIC
set_property(TARGET ${smarties_core} PROPERTY POSITION_INDEPENDENT_CODE ON)

# External libraries

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

# For supporting CMake < 3.9:

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

target_link_libraries(${smarties_core} PUBLIC
  MPI::MPI_CXX
  OpenMP::OpenMP_CXX
  ${BLAS_LIBRARIES})

# fix for llvm that does not link to libatomic
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_link_libraries(${smarties_core} PUBLIC -latomic)
endif()

set_target_properties(${smarties_core} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${lib_output}"
  )

# Python bindings

if (COMPILE_PY_SO)
  
  add_subdirectory(${source_dir}/extern/pybind11)
  set(PYBIND11_CPP_STANDARD -std=c++14)

  set(smarties_py "smarties")
  
  add_library(${smarties_py} MODULE ${source_dir}/smarties/smarties_pybind11.cpp)
  target_link_libraries(${smarties_py} PRIVATE ${smarties_core} pybind11::module)

  set_target_properties(${smarties_py} PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    LIBRARY_OUTPUT_DIRECTORY "${lib_output}"
    )

endif()



