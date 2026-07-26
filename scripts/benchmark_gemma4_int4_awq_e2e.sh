#!/usr/bin/env bash
# End-to-end benchmark sweep for gemma4_unified INT4 (compressed-tensors AWQ).
#
# Full sweep per AGENTS.md (e2e inference: -p 512,1024,2048,4096) at tg=512 with a
# KV-depth sweep.
#
# The asymmetric zero-point correction in matmul_int4 is baked in (int4.hpp):
# it is load-bearing for correctness — without it the AWQ model collapses to a
# degenerate repeating token — so there is no OFF baseline to A/B against. This
# script measures the throughput of the correct (only) path.
set -euo pipefail

build_dir="${BUILD_DIR:-/workspace/build}"
model_dir="${MODEL_DIR:-/workspace/models/cyankiwi_gemma-4-12B-it-qat-AWQ-INT4}"
pp="${PP:-512,1024,2048,4096}"
tg="${TG:-512}"
depths="${DEPTHS:-0,512,1024,2048,4096,8192}"
warmup="${WARMUP:-1}"
runs="${RUNS:-3}"
out_fmt="${OUT_FMT:-md}"
log="${LOG:-/tmp/gemma4_int4_awq_e2e.log}"

common="--model ${model_dir} -p ${pp} -n ${tg} -d ${depths} -w ${warmup} -r ${runs} -o ${out_fmt}"

echo "===== gemma4_unified INT4 AWQ e2e sweep ====="
echo "model : ${model_dir}"
echo "args  : ${common}"
echo "log   : ${log}"
echo

"${build_dir}/arcaine_mbench" ${common} 2>&1 | tee "${log}"

echo
echo "===== done — log written to ${log} ====="
