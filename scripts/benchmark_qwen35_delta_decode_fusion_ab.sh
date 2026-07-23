#!/usr/bin/env bash
set -euo pipefail

PROMPT="${ARCAINE_QWEN35_DELTA_DECODE_PROMPT:-1}"
TOKENS="${ARCAINE_QWEN35_DELTA_DECODE_TOKENS:-32}"
WARMUP="${ARCAINE_QWEN35_DELTA_DECODE_WARMUP:-1}"
RUNS="${ARCAINE_QWEN35_DELTA_DECODE_RUNS:-5}"

/workspace/build/qwen35_delta_decode_fusion_bench \
  --p "${PROMPT}" --n "${TOKENS}" --w "${WARMUP}" --r "${RUNS}"
