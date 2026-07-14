#!/usr/bin/env bash
#
# Build helper for the RPP HAL workflow tooling (benchmarks + tests).
#
# These programs are NOT part of the OpenCV build. They link against an already
# installed OpenCV (built with -DWITH_RPP=ON) and, for the native benchmark,
# directly against RPP + HIP. Use them to verify correctness and measure the
# RPP HAL against OpenCV native.
#
# Environment overrides:
#   OpenCV_DIR   Path to the OpenCV install (with lib/cmake/opencv4 or pkgconfig).
#                Defaults to pkg-config lookup of "opencv4".
#   ROCM_PATH    ROCm install prefix (default: /opt/rocm).
#
# Usage:
#   ./build.sh            # build everything
#   ./build.sh clean      # remove built binaries
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O2 -std=c++17 -Wall}"

if [[ "${1:-}" == "clean" ]]; then
  rm -f benchmark_rpp benchmark_rpp_full benchmark_rpp_native \
        test_rpp_hal test_rpp_correctness
  echo "cleaned."
  exit 0
fi

# --- OpenCV flags (for HAL-integrated tools) ---------------------------------
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists opencv4; then
  OPENCV_CFLAGS="$(pkg-config --cflags opencv4)"
  OPENCV_LIBS="$(pkg-config --libs opencv4)"
elif [[ -n "${OpenCV_DIR:-}" ]]; then
  OPENCV_CFLAGS="-I${OpenCV_DIR}/include/opencv4 -I${OpenCV_DIR}/include"
  OPENCV_LIBS="-L${OpenCV_DIR}/lib -lopencv_core -lopencv_imgproc"
else
  echo "ERROR: OpenCV not found. Set OpenCV_DIR or make pkg-config see opencv4." >&2
  exit 1
fi

# --- RPP + HIP flags (for the native benchmark) ------------------------------
RPP_CFLAGS="-I${ROCM_PATH}/include -D__HIP_PLATFORM_AMD__=1"
RPP_LIBS="-L${ROCM_PATH}/lib -lrpp -lamdhip64"

build_opencv_tool() {
  local src="$1" out="$2"
  echo "  CXX  $out"
  # shellcheck disable=SC2086
  $CXX $CXXFLAGS $OPENCV_CFLAGS "$src" -o "$out" $OPENCV_LIBS
}

echo "Building OpenCV HAL-integrated tools..."
build_opencv_tool test_rpp_correctness.cpp test_rpp_correctness
build_opencv_tool test_rpp_hal.cpp          test_rpp_hal
build_opencv_tool benchmark_rpp.cpp         benchmark_rpp
build_opencv_tool benchmark_rpp_full.cpp    benchmark_rpp_full

echo "Building native RPP benchmark..."
echo "  CXX  benchmark_rpp_native"
# shellcheck disable=SC2086
$CXX $CXXFLAGS $RPP_CFLAGS benchmark_rpp_native.cpp -o benchmark_rpp_native $RPP_LIBS

echo "done. Binaries are in $HERE"
