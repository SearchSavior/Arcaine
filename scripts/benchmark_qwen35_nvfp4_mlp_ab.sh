#!/usr/bin/env bash
set -euo pipefail

MODEL="${ARCAINE_QWEN35_MODEL_DIR:-/workspace/models/unsloth_Qwen3.6-27B-NVFP4}"
TOKENS="${ARCAINE_QWEN35_MLP_TOKENS:-1,128,512,1024}"
WARMUP="${ARCAINE_QWEN35_MLP_WARMUP:-1}"
RUNS="${ARCAINE_QWEN35_MLP_RUNS:-5}"
LAYER="${ARCAINE_QWEN35_MLP_LAYER:-0}"
KERNELS="${ARCAINE_QWEN35_MLP_KERNELS:-onednn,xe2-dpas}"

/workspace/build/qwen35_nvfp4_mlp_bench \
  --model "${MODEL}" --layer "${LAYER}" \
  -p "${TOKENS}" -w "${WARMUP}" -r "${RUNS}" \
  --kernels "${KERNELS}"
