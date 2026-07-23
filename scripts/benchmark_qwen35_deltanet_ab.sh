#!/usr/bin/env bash
set -euo pipefail

PREFILL="${ARCAINE_QWEN35_DELTA_PREFILL:-512,1024,2048,4096}"
TOKENS="${ARCAINE_QWEN35_DELTA_TOKENS:-32}"
WARMUP="${ARCAINE_QWEN35_DELTA_WARMUP:-1}"
RUNS="${ARCAINE_QWEN35_DELTA_RUNS:-5}"
KERNELS="${ARCAINE_QWEN35_DELTA_KERNELS:-baseline,esimd}"

/workspace/build/qwen35_deltanet_bench \
  --p "${PREFILL}" --n "${TOKENS}" --w "${WARMUP}" --r "${RUNS}" \
  --kernels "${KERNELS}"
