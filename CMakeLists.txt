# NOTE: Update `install_dependencies.sh` if updating the minimum required version.
cmake_minimum_required(VERSION 3.2)
project(smarties)

# Choose Release mode as default.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(ROOT_FOLDER ${CMAKE_CURRENT_LIST_DIR})

option(COMPILE_SO "Compile shared library" ON)
option(COMPILE_PY_SO "Compile Python bindings" ON)

list(APPEND CMAKE_PREFIX_PATH $ENV{SMARTIES_ROOT}/extern/)
################################################################################
# Pybind11.
# Not using `python3-config` because it doesn't seem to work with `pip install --user`.
execute_process(
    COMMAND python3 -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE retcode1)
execute_process(
    #COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX')[:-3])"
    COMMAND python3-config --extension-suffix
    OUTPUT_VARIABLE PY_SO_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE retcode2)
execute_process(
    COMMAND python3-config --ldflags
    OUTPUT_VARIABLE PYTHON_LIBRARIES
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE retcode3)
separate_arguments(PYBIND11_INCLUDES)
set(PYBIND11_FLAGS ${PYBIND11_INCLUDES})
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(PYBIND11_FLAGS ${PYBIND11_FLAGS} -undefined dynamic_lookup)
endif()

if (COMPILE_PY_SO)
    if (NOT "${retcode1}" STREQUAL "0")
        message(FATAL_ERROR "pybind11 not installed, cannot compile Python bindings!")
    endif()
    if (NOT "${retcode2}" STREQUAL "0")
        message(FATAL_ERROR "Cannot get which Python .so suffix to use, cannot compile bindings!")
    endif()
    if (NOT "${retcode3}" STREQUAL "0")
        message(FATAL_ERROR "Command `python3-config --ldflags` failed, cannot compile bindings!")
    endif()
endif()
################################################################################

# MPI - must be before modifying CMAKE_SHARED_LIBRARY_SUFFIX (?).
find_package(MPI REQUIRED)
set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

# Source files and target.


# workaround for pybind11:
set(CMAKE_SHARED_LIBRARY_PREFIX "")      # No "lib" prefix for shared py11 lib.
set(CMAKE_SHARED_LIBRARY_SUFFIX "")      # We already specify it manually.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ".")

set(COMMON_SOURCES
    ${ROOT_FOLDER}/source/Learners/PPO.cpp
    ${ROOT_FOLDER}/source/Learners/MixedPG.cpp
    ${ROOT_FOLDER}/source/Learners/NAF.cpp
    ${ROOT_FOLDER}/source/Learners/DPG.cpp
    ${ROOT_FOLDER}/source/Learners/DQN.cpp
    ${ROOT_FOLDER}/source/Learners/ACER.cpp
    ${ROOT_FOLDER}/source/Learners/RACER.cpp
    ${ROOT_FOLDER}/source/Learners/CMALearner.cpp
    ${ROOT_FOLDER}/source/Learners/Learner_pytorch.cpp
    ${ROOT_FOLDER}/source/Learners/Learner_approximator.cpp
    ${ROOT_FOLDER}/source/Learners/AlgoFactory.cpp
    ${ROOT_FOLDER}/source/Learners/Learner.cpp

    ${ROOT_FOLDER}/source/ReplayMemory/Sampling.cpp
    ${ROOT_FOLDER}/source/ReplayMemory/Collector.cpp
    ${ROOT_FOLDER}/source/ReplayMemory/Sequence.cpp
    ${ROOT_FOLDER}/source/ReplayMemory/MemoryBuffer.cpp
    ${ROOT_FOLDER}/source/ReplayMemory/DataCoordinator.cpp
    ${ROOT_FOLDER}/source/ReplayMemory/MemoryProcessing.cpp

    ${ROOT_FOLDER}/source/Math/Math_tests.cpp
    ${ROOT_FOLDER}/source/Utils/Warnings.cpp
    ${ROOT_FOLDER}/source/Utils/Profiler.cpp
    ${ROOT_FOLDER}/source/Utils/StatsTracker.cpp

    ${ROOT_FOLDER}/source/Core/Launcher.cpp
    ${ROOT_FOLDER}/source/Core/Master.cpp
    ${ROOT_FOLDER}/source/Core/Worker.cpp

    ${ROOT_FOLDER}/source/Communicator.cpp
    ${ROOT_FOLDER}/source/Settings.cpp
    ${ROOT_FOLDER}/source/Engine.cpp

    ${ROOT_FOLDER}/source/Network/Approximator.cpp
    ${ROOT_FOLDER}/source/Network/Builder.cpp
    ${ROOT_FOLDER}/source/Network/Network.cpp
    ${ROOT_FOLDER}/source/Network/Optimizer.cpp
    ${ROOT_FOLDER}/source/Network/CMA_Optimizer.cpp
)

# GCC flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") # -shared
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Wno-unused-parameter")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -O3 -march=native -fstrict-aliasing -ffast-math -falign-functions=32")
set(CMAKE_CXX_EXTENSIONS OFF)  # -std=c++11, not -std=gnu++11!
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# OpenMP support
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# OpenBLAS
# if(FALSE) # OpenBLAS broken on Mac ...
find_package(BLAS REQUIRED)
find_path(BLAS_INCLUDE_DIRS cblas.h
  /usr/include
  /usr/local/include
  /usr/include/openblas
  /opt/OpenBLAS/include
  /usr/include/x86_64-linux-gnu
  /path/to/mkl/include
  $ENV{BLAS_HOME}/include
  $ENV{SMARTIES_ROOT}/extern/include)
include_directories(${BLAS_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_OPENBLAS")
# endif() # ... OpenBLAS broken on Mac

# Targets.
if (COMPILE_SO)
    set(SHARED_LIB "libsmarties.so")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${ROOT_FOLDER}/lib)
    add_library(${SHARED_LIB} SHARED ${COMMON_SOURCES} )
    target_link_libraries(${SHARED_LIB} ${BLAS_LIBRARIES})
endif()

if (COMPILE_PY_SO)
    set(PY_SO "smarties${PY_SO_SUFFIX}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${ROOT_FOLDER}/lib)
    add_library(${PY_SO} SHARED ${COMMON_SOURCES} )
    target_compile_options(${PY_SO} PRIVATE ${PYBIND11_FLAGS})
    #target_link_libraries(${PY_SO} ${PYTHON_LIBRARIES})
    target_link_libraries(${PY_SO} ${BLAS_LIBRARIES})
endif()



