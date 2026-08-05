#!/usr/bin/env bash
# A/B benchmark: qwen3_5_moe Gated DeltaNet decode step (S==1, cached state),
# 9-launch split chain vs 2-launch fused kernel, on the exact production
# decode codepath (conv1d_causal_decode included in both; GEMVs excluded --
# identical for both variants).
#
#   split: conv -> extract -> l2norm q/k -> scale -> sigmoid -> compute_g ->
#          recurrent_gated_delta_decode -> gated_rmsnorm
#   fused: conv -> qwen_gdn_decode_fused
#
# Reports numerics of fused vs split (core bf16 max-abs/cos, fp32 state
# max-rel) plus per-pass latency. Production toggle for server/mbench:
# QWEN35_GDN_DECODE_FUSED=1 (default off). Run inside the dev container:
#   docker exec arcaine-dev-run-920bd816043e bash \
#     /workspace/scripts/benchmark_qwen35_moe_gdn_decode_fusion_ab.sh
set -euo pipefail

N="${ARCAINE_QWEN35_MOE_GDN_DECODE_N:-100}"
WARMUP="${ARCAINE_QWEN35_MOE_GDN_DECODE_WARMUP:-2}"
RUNS="${ARCAINE_QWEN35_MOE_GDN_DECODE_RUNS:-5}"
DEVICE="${ARCAINE_QWEN35_MOE_GDN_DECODE_DEVICE:-0}"

/workspace/build/arcaine_kbench qwen35-gdn --mode decode \
  --kernels split,fused -n "${N}" -w "${WARMUP}" -r "${RUNS}" --device "${DEVICE}"
