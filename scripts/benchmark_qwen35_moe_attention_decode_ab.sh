#!/usr/bin/env bash
# A/B benchmark: qwen3_5_moe full-attention DECODE step (seq==1), split chain
# vs fused flash-decoding DPAS kernel, on the exact production decode
# codepath (q/k/v projections excluded -- identical for both variants).
#
#   split: expand_kv x2 -> GEMV scores -> scale -> softmax -> bf16 -> GEMV ctx
#          (~10 launches, 4+ amplified KV passes)
#   fused: qwen_attn_decode_fused partial + combine (2 launches, 1 KV pass)
#
# Reports numerics of fused vs split (ctx bf16 max-abs/max-rel) plus per-pass
# latency across KV depths. Production toggle for server/mbench:
# QWEN35_ATTN_DECODE_FUSED=1 (default off). Run inside the dev container:
#   docker exec arcaine-dev-run-920bd816043e bash \
#     /workspace/scripts/benchmark_qwen35_moe_attention_decode_ab.sh
set -euo pipefail

DEPTHS="${ARCAINE_QWEN35_MOE_ATTN_DEPTHS:-1024,4096,8192,16384,32768,65536}"
N="${ARCAINE_QWEN35_MOE_ATTN_DECODE_N:-20}"
WARMUP="${ARCAINE_QWEN35_MOE_ATTN_DECODE_WARMUP:-1}"
RUNS="${ARCAINE_QWEN35_MOE_ATTN_DECODE_RUNS:-5}"
DEVICE="${ARCAINE_QWEN35_MOE_ATTN_DECODE_DEVICE:-0}"

/workspace/build/arcaine_kbench qwen35moe-attention \
  --kernels split,fused -d "${DEPTHS}" -n "${N}" -w "${WARMUP}" -r "${RUNS}" \
  --device "${DEVICE}"
