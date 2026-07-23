#!/usr/bin/env bash
set -euo pipefail

build_dir="${BUILD_DIR:-/workspace/build}"
model_dir="${MODEL_DIR:-/workspace/models/diffusiongemma-26B-A4B-it-AWQ-INT4}"
warmup="${WARMUP:-1}"
runs="${RUNS:-5}"
baseline_log="${BASELINE_LOG:-/tmp/diffusion_int4_e2e_full_canvas_baseline.log}"
fused_log="${FUSED_LOG:-/tmp/diffusion_int4_e2e_full_canvas_fused.log}"

cmake --build "${build_dir}" --target diffusion_bench -j"$(nproc)"

common_args="--model ${model_dir} --device 0 --layers single --experts layer-owner -p 512,1024,2048,4096 -n 256 -ds 48 -w ${warmup} -r ${runs} --seed 42 --md"

env -u DIFF_INT4_FUSE_DENSE_GATE_UP \
    -u DIFF_INT4_FUSE_EXPERT_POSTNORM \
    -u DIFF_INT4_FUSE_SELFCOND_ADD_NORM \
    ZE_AFFINITY_MASK=0 DIFF_BENCH_FORCE_FULL_CANVAS=1 \
    DIFF_FORCE_DENOISE_STEPS=1 \
    "${build_dir}/diffusion_bench" ${common_args} 2>&1 | tee "${baseline_log}"

ZE_AFFINITY_MASK=0 \
DIFF_BENCH_FORCE_FULL_CANVAS=1 \
DIFF_FORCE_DENOISE_STEPS=1 \
DIFF_INT4_FUSE_DENSE_GATE_UP=1 \
DIFF_INT4_FUSE_EXPERT_POSTNORM=1 \
DIFF_INT4_FUSE_SELFCOND_ADD_NORM=1 \
"${build_dir}/diffusion_bench" ${common_args} 2>&1 | tee "${fused_log}"
