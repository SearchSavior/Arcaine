#!/usr/bin/env bash
set -euo pipefail

PREFILL="${ARCAINE_QWEN35_ATTN_PREFILL:-128,512,1024}"
DEPTH="${ARCAINE_QWEN35_ATTN_DEPTH:-0,512,1024,2048}"
WARMUP="${ARCAINE_QWEN35_ATTN_WARMUP:-1}"
RUNS="${ARCAINE_QWEN35_ATTN_RUNS:-5}"
KERNELS="${ARCAINE_QWEN35_ATTN_KERNELS:-baseline,xmx}"

/workspace/build/qwen35_attention_bench \
  --p "${PREFILL}" --d "${DEPTH}" --w "${WARMUP}" --r "${RUNS}" \
  --kernels "${KERNELS}"
