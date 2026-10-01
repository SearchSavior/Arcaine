#!/usr/bin/env bash
#
# build.sh -- configure and build Arcaine with the SYCL/Intel GPU toolchain.
#
set -euo pipefail

cmake -B build -G Ninja \
  -DCMAKE_CXX_COMPILER=icpx \
#  -DARCAINE_SYCL_TARGETS=intel_gpu_bmg_g31
#  -DARCAINE_SYCL_TARGETS=intel_gpu_bmg_g21 B60, B50, B580
cmake --build build -j"$(nproc)"
