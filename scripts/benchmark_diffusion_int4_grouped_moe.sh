#!/usr/bin/env bash
set -euo pipefail

build_dir="${BUILD_DIR:-/workspace/build}"
iterations="${ITERATIONS:-10}"
local_experts="${LOCAL_EXPERTS:-44}"
first_expert="${FIRST_EXPERT:-0}"
seq_lengths="${SEQ_LENGTHS:-256}"

cmake --build "${build_dir}" --target arcaine_kbench -j"$(nproc)"

IFS=',' read -ra seq_values <<< "${seq_lengths}"
for seq in "${seq_values[@]}"; do
  DIFF_INT4_GROUPED_DPAS_MOE=1 \
  "${build_dir}/arcaine_kbench" diffusion-int4-grouped-moe \
    --experts "${local_experts}" \
    --first "${first_expert}" \
    --seq "${seq}" \
    --iterations "${iterations}"
done
