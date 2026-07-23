#!/usr/bin/env bash
set -euo pipefail

build_dir="${BUILD_DIR:-/workspace/build}"
iterations="${1:-30}"

cmake --build "${build_dir}" --target diffusion_int4_fusion_bench -j"$(nproc)"

ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}" \
DIFF_INT4_FUSE_DENSE_GATE_UP=1 \
DIFF_INT4_FUSE_EXPERT_POSTNORM=1 \
DIFF_INT4_FUSE_SELFCOND_ADD_NORM=1 \
"${build_dir}/diffusion_int4_fusion_bench" --iterations "${iterations}"
