#!/usr/bin/env bash
set -euo pipefail

# Default Python interpreter
DEFAULT_PYTHON="$(which python)"

print_usage() {
    echo "Usage: $0 -b <build_dir> [-p <python_path>] [-o <output_dir>]"
    echo
    echo "Options:"
    echo "  -b, --build-dir   Path to the build directory (required)"
    echo "  -p, --python      Path to the Python interpreter (optional, default: ${DEFAULT_PYTHON})"
    echo "  -o, --output-dir  Directory to copy built .so files into (optional)"
    echo "  -h, --help        Show this help message"
}

# Parse command line arguments
BUILD_DIR=""
PYTHON_PATH="${DEFAULT_PYTHON}"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -p|--python)
            PYTHON_PATH="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [[ -z "${BUILD_DIR}" ]]; then
    echo "Error: build directory is required."
    print_usage
    exit 1
fi

# Resolve absolute paths
BUILD_DIR="$(realpath "${BUILD_DIR}")"
EIGEN_DIR="$HOME/local/include/eigen3/Eigen"
if [[ -n "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR}")"
fi

echo "Build directory : ${BUILD_DIR}"
echo "Python path     : ${PYTHON_PATH}"
echo "Eigen include   : ${EIGEN_DIR}"
if [[ -n "${OUTPUT_DIR}" ]]; then
    echo "Output dir      : ${OUTPUT_DIR}"
fi

# 1) Install pybind11 via pip
echo "Installing / updating pybind11..."
"${PYTHON_PATH}" -m pip install --upgrade pybind11

# 2) Check if a current .so already exists
if [[ -d "${OUTPUT_DIR}" ]] && ls "${OUTPUT_DIR}"/*.so* >/dev/null 2>&1; then
    echo "Already built. Skip rebuilding."
else
    echo "Need to build C++ module."

    # 3) Recreate build directory
    echo "Recreating build directory..."
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"

    # 4) Check / install Eigen locally if needed
    if [[ ! -d "${EIGEN_DIR}" ]]; then
        echo "Eigen not found. Downloading and installing locally..."
        pushd "${BUILD_DIR}" >/dev/null

        EIGEN_TAR="eigen-3.4.0.tar.gz"
        EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/3.4.0/${EIGEN_TAR}"

        wget "${EIGEN_URL}"
        tar -xvzf "${EIGEN_TAR}"

        mkdir -p "$HOME/local/include/eigen3"
        mv eigen-3.4.0/Eigen "$HOME/local/include/eigen3/"

        popd >/dev/null
    else
        echo "Eigen already exists at ${EIGEN_DIR}."
    fi

    # 5) Query Python prefix and version
    echo "Querying Python prefix and version..."
    PY_PREFIX="$("${PYTHON_PATH}" -c 'import sys; print(sys.prefix)')"
    PY_VERSION="$("${PYTHON_PATH}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

    echo "Python prefix   : ${PY_PREFIX}"
    echo "Python version  : ${PY_VERSION}"

    # 6) Configure and build with CMake and Make
    echo "Running CMake..."
    pushd "${BUILD_DIR}" >/dev/null

    cmake \
      -DCMAKE_PREFIX_PATH="${PY_PREFIX}" \
      -DPython_EXECUTABLE="${PY_PREFIX}/bin/python" \
      -DPython_LIBRARY="${PY_PREFIX}/lib/libpython${PY_VERSION}.so" \
      -DPython_INCLUDE_DIR="${PY_PREFIX}/include/python${PY_VERSION}" \
      -Dpybind11_DIR="${PY_PREFIX}/lib/python${PY_VERSION}/site-packages/pybind11/share/cmake/pybind11" \
      ..

    echo "Building with make..."
    make

    popd >/dev/null
fi

# 7) Optionally copy .so files to output dir
if [[ -n "${OUTPUT_DIR}" ]]; then
    echo "Copying built .so files to output directory..."
    mkdir -p "${OUTPUT_DIR}"

    shopt -s nullglob
    so_files=("${BUILD_DIR}"/*.so*)
    shopt -u nullglob

    if [[ ${#so_files[@]} -eq 0 ]]; then
        echo "Warning: no .so files found in ${BUILD_DIR} to copy."
    else
        for f in "${so_files[@]}"; do
            echo "  -> $(basename "$f")"
            cp -f "$f" "${OUTPUT_DIR}/"
        done
        echo "Shared libraries copied to ${OUTPUT_DIR}"
    fi
fi

echo "✅ Murty C++ module build completed."
