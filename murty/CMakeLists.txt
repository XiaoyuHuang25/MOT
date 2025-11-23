cmake_minimum_required (VERSION 3.5)
project (murty)

# Use Python information from the command line (Python_EXECUTABLE, etc.)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Eigen installed by build_murty.sh under $HOME/local/include/eigen3/Eigen
set(EIGEN3_INCLUDE_DIR "$ENV{HOME}/local/include/eigen3")
if (NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Core")
    message(FATAL_ERROR
        "Eigen not found at ${EIGEN3_INCLUDE_DIR}. "
        "Please run build_murty.sh first or adjust EIGEN3_INCLUDE_DIR.")
endif()
include_directories("${EIGEN3_INCLUDE_DIR}")

# pybind11 installed via pip; path is passed by -Dpybind11_DIR=...
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(murty src/murty.cpp)
