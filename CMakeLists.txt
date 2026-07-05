cmake_minimum_required(VERSION 3.20)
project(myproject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # for clangd / IDE tooling
set(ARCAINE_SYCL_TARGETS "" CACHE STRING
    "Optional DPC++ SYCL AOT target list, e.g. intel_gpu_bmg_g31")
if(ARCAINE_SYCL_TARGETS)
    set(ARCAINE_SYCL_TARGET_FLAG "-fsycl-targets=${ARCAINE_SYCL_TARGETS}")
endif()

# Build with the Intel DPC++/SYCL compiler. Configure with:
#   cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx
#
# oneDNN is built from source and installed to /opt/onednn in the image,
# which is on CMAKE_PREFIX_PATH (set via ENV in the Dockerfile), so this
# find_package picks up its DNNLConfig.cmake automatically.
find_package(dnnl CONFIG REQUIRED)
include_directories(third_party third_party/minja/include)

# The dev image also ships older oneDNN builds through oneAPI/OpenVINO. Keep the
# source-built oneDNN selected by find_package() ahead of LD_LIBRARY_PATH entries
# so newer symbols such as dnnl_primitive_attr_set_scales_v3 resolve at runtime.
add_link_options(-Wl,--disable-new-dtags)

# libsoxr (static, built in-tree)
add_library(soxr STATIC IMPORTED)
set_target_properties(soxr PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/third_party/soxr/build/src/libsoxr.a"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/third_party/soxr/src"
)

# ONNX Runtime (for Silero VAD)
set(ORT_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_DIR}/lib/libonnxruntime.so.1.26.0"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include"
)

# ---------------------------------------------------------------------------
# Source groups
#   common/         — architecture-independent engine (registry, preprocessing, ...)
#   modeling/<arch> — one directory per model architecture; add a sibling to extend.
# ---------------------------------------------------------------------------
set(COMMON_SRCS
    src/register_builtins.cpp
    src/common/registry.cpp
    src/common/io/quant_loader.cpp
    src/common/preprocess/tokenizer.cpp
    src/common/preprocess/chat_template.cpp
    src/common/preprocess/image_proc.cpp
    src/common/preprocess/audio_proc.cpp
)

set(GEMMA4_SRCS
    src/modeling/gemma4_unified/loader.cpp
    src/modeling/gemma4_unified/attention.cpp
    src/modeling/gemma4_unified/ffn.cpp
    src/modeling/gemma4_unified/vision_embedder.cpp
    src/modeling/gemma4_unified/audio_embedder.cpp
    src/modeling/gemma4_unified/model.cpp
    src/modeling/gemma4_unified/register.cpp
)

set(QWEN_SRCS
    src/modeling/qwen3_5_moe/loader.cpp
    src/modeling/qwen3_5_moe/model.cpp
    src/modeling/qwen3_5_moe/register.cpp
)

# All built-in architectures linked into the inference/bench binaries.
set(ARCH_SRCS ${GEMMA4_SRCS} ${QWEN_SRCS})

# ---------------------------------------------------------------------------
add_executable(get_device_props src/utils/gpu_check.cpp)
target_compile_options(get_device_props PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(get_device_props PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_libraries(get_device_props PRIVATE DNNL::dnnl)
set_target_properties(get_device_props PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.skills/check-available-hardware")

# ---------------------------------------------------------------------------
add_executable(gemma4 src/main.cpp ${COMMON_SRCS} ${ARCH_SRCS})
target_compile_options(gemma4 PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(gemma4   PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
# ARCH_SRCS now pulls qwen3_5_moe (NVFP4 loader) + common/quant_loader.cpp, both
# of which include nvfp4.hpp -> __spirv_SubgroupMatrixMultiplyAccumulateINTEL.
target_link_options(gemma4 PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(gemma4 PRIVATE DNNL::dnnl soxr onnxruntime)
set_target_properties(gemma4 PROPERTIES
    BUILD_RPATH "${ORT_DIR}/lib"
    INSTALL_RPATH "${ORT_DIR}/lib")

# ---------------------------------------------------------------------------
add_executable(test_tokenizer src/common/test_tokenizer.cpp src/common/preprocess/tokenizer.cpp)
target_compile_options(test_tokenizer PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(test_tokenizer   PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_libraries(test_tokenizer PRIVATE DNNL::dnnl)

# ---------------------------------------------------------------------------
# --chat-template-kwargs parsing + chat-template merge. Logic is SYCL-free, but
# the global -fsycl-targets flag means every target must also pass -fsycl.
# Pass a model dir as argv[1] to exercise the render-merge assertions.
add_executable(test_chat_template_kwargs
    src/utils/test_chat_template_kwargs.cpp
    src/common/preprocess/chat_template.cpp
    src/common/preprocess/tokenizer.cpp)
target_compile_options(test_chat_template_kwargs PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(test_chat_template_kwargs    PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_libraries(test_chat_template_kwargs  PRIVATE DNNL::dnnl)

# ---------------------------------------------------------------------------
# Inference benchmark. Goes through the model registry, so it links the same
# architecture + common sources as the main binary.
add_executable(bench src/bench.cpp ${COMMON_SRCS} ${ARCH_SRCS})
target_compile_options(bench PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(bench   PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
# See gemma4: ARCH_SRCS / COMMON_SRCS include nvfp4.hpp-bearing loader TUs.
target_link_options(bench PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(bench PRIVATE DNNL::dnnl soxr onnxruntime)
set_target_properties(bench PROPERTIES
    BUILD_RPATH "${ORT_DIR}/lib"
    INSTALL_RPATH "${ORT_DIR}/lib")

# ---------------------------------------------------------------------------
# DiffusionGemma 26B-A4B (block-diffusion MoE) — hello-world driver.
# Text-only path: needs no soxr / onnxruntime preprocessing.
set(DIFFUSION_SRCS
    src/common/preprocess/tokenizer.cpp
    src/common/preprocess/chat_template.cpp
    src/common/io/quant_loader.cpp
    src/modeling/diffusion_gemma/loader.cpp
    src/modeling/diffusion_gemma/moe.cpp
    src/common/gpu/expert_parallel.cpp
    src/common/gpu/placement.cpp
    src/modeling/diffusion_gemma/activation_plan.cpp
    src/modeling/diffusion_gemma/attention.cpp
    src/modeling/diffusion_gemma/generate.cpp
    src/modeling/diffusion_gemma/model.cpp
    src/common/gpu/mem_planner.cpp
    src/utils/chat.cpp
    src/utils/viz.cpp
    src/utils/profile.cpp
)
add_executable(diffusion_gemma src/diffusion_main.cpp ${DIFFUSION_SRCS})
target_compile_options(diffusion_gemma PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(diffusion_gemma   PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(diffusion_gemma PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)

# OpenAI-compatible HTTP API server for DiffusionGemma.
add_executable(diffusion_server
    src/diffusion_server.cpp
    src/utils/gemma4_tool_call_parser.cpp
    ${DIFFUSION_SRCS})
target_compile_options(diffusion_server PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(diffusion_server   PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(diffusion_server PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)

# llama-bench style throughput benchmark (loads once, sweeps expert kernels).
add_executable(diffusion_bench src/diffusion_bench.cpp ${DIFFUSION_SRCS})
target_compile_options(diffusion_bench PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(diffusion_bench   PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(diffusion_bench PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(diffusion_bench PRIVATE DNNL::dnnl)
target_link_libraries(diffusion_gemma PRIVATE DNNL::dnnl)
target_link_libraries(diffusion_server PRIVATE DNNL::dnnl)

# ---------------------------------------------------------------------------
# Phase 5 — Qwen3.5-MoE NVFP4 loader smoke test. Loads real checkpoint weights
# (no kernels) and verifies tensor-name resolution, scale transpose
# [N,K/16]->[K/16,N], dst_scale folding, and per-layer struct population.
# Usage: qwen_load_smoke [model_dir] [max_layers]
# NOTE: the loader harness only needs loader.cpp + quant_loader.cpp; it does
# not construct QwenModel or register the arch, so model.cpp/register.cpp are
# deliberately excluded (they would otherwise pull build_chat_prompt/register_arch
# into the link line).
add_executable(qwen_load_smoke
    src/qwen_load_smoke.cpp
    src/common/io/quant_loader.cpp
    src/modeling/qwen3_5_moe/loader.cpp)
target_compile_options(qwen_load_smoke PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(qwen_load_smoke PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
# nvfp4.hpp (included transitively) pulls __spirv_SubgroupMatrixMultiplyAccumulateINTEL;
# the translator needs the extension even though this binary runs no DPAS kernel.
target_link_options(qwen_load_smoke PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(qwen_load_smoke PRIVATE DNNL::dnnl)

# qwen_kernels_selfcheck: build-verify + GPU smoke for qwen3_5_moe/kernels.hpp.
# An unused inline SYCL lambda is not device-codegen'd, so calling each kernel
# once (this TU) is what forces SPIR-V codegen and validates the device bodies.
# Compile in arcaine-dev-1 (SPIR-V codegen needs no physical GPU); run in
# arcaine-dev-run-* to exercise on the BMG G31. Numerical validation vs the HF
# reference is Phase 6; this is the milestone-2 build/run smoke.
add_executable(qwen_kernels_selfcheck src/qwen_kernels_selfcheck.cpp)
target_compile_options(qwen_kernels_selfcheck PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(qwen_kernels_selfcheck PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(qwen_kernels_selfcheck PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(qwen_kernels_selfcheck PRIVATE DNNL::dnnl)

# qwen_attn_selfcheck: milestone-3 build/run smoke for the Qwen full-attention
# forward (modeling/qwen3_5_moe/attention.hpp). Loads one full-attn layer's real
# NVFP4 + BF16(+1) weights straight from the checkpoint (no MoE / no QwenWeights)
# and runs qwen_full_attention_forward on a synthetic hidden, checking finiteness,
# determinism, and zero-input->zero-output; exercises both prefill and decode
# paths. Build in arcaine-dev-1 (SPIR-V codegen needs no GPU); run in
# arcaine-dev-run-* with ARCAINE_QWEN_MODEL_DIR=<qwen checkpoint dir> (and
# optional ARCAINE_QWEN_ATTN_LAYER=<full-attn idx, default 3>). Numerical
# validation vs HF is Phase 6 (deferred).
add_executable(qwen_attn_selfcheck
    src/qwen_attn_selfcheck.cpp
    src/common/io/quant_loader.cpp)
target_compile_options(qwen_attn_selfcheck PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(qwen_attn_selfcheck PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(qwen_attn_selfcheck PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(qwen_attn_selfcheck PRIVATE DNNL::dnnl)

# qwen_moe_selfcheck: milestone-4 build/run smoke for the Qwen MoE block forward
# (modeling/qwen3_5_moe/moe.hpp). Loads one layer's full MoE block (router BF16,
# 256 routed NVFP4 experts, shared NVFP4 expert, shared_expert_gate BF16) from the
# checkpoint and runs qwen_moe_forward on a synthetic hidden, checking finiteness,
# determinism, nonzeroness, and zero-input->zero-output; exercises prefill + decode.
# Build in arcaine-dev-1; run in arcaine-dev-run-* with
# ARCAINE_QWEN_MODEL_DIR=<qwen checkpoint dir> (optional ARCAINE_QWEN_MOE_LAYER=<idx>).
add_executable(qwen_moe_selfcheck
    src/qwen_moe_selfcheck.cpp
    src/common/io/quant_loader.cpp)
target_compile_options(qwen_moe_selfcheck PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(qwen_moe_selfcheck PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(qwen_moe_selfcheck PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(qwen_moe_selfcheck PRIVATE DNNL::dnnl)

# qwen_deltanet_selfcheck: milestone-5 build/run smoke for the Qwen Gated DeltaNet
# (modeling/qwen3_5_moe/gated_deltanet.hpp). Loads one linear-attn layer's real
# BF16 (in_proj_qkv/z/a/b, conv1d, A_log, dt_bias, norm) + NVFP4 (out_proj)
# weights from the checkpoint and runs qwen_linear_attn_forward on a synthetic
# hidden, checking prefill finite/deterministic/nonzero/zero-in->zero-out, decode
# finite/deterministic, AND prefill(N)[last] == prefill(N-1)+decode equivalence
# (cross-validates host chunked vs device recurrent core). Build in arcaine-dev-1;
# run in arcaine-dev-run-* with ARCAINE_QWEN_MODEL_DIR=<qwen checkpoint dir>
# (optional ARCAINE_QWEN_DELTANET_LAYER=<linear-attn idx, default 0>).
add_executable(qwen_deltanet_selfcheck
    src/qwen_deltanet_selfcheck.cpp
    src/common/io/quant_loader.cpp)
target_compile_options(qwen_deltanet_selfcheck PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(qwen_deltanet_selfcheck PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(qwen_deltanet_selfcheck PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(qwen_deltanet_selfcheck PRIVATE DNNL::dnnl)

# qwen_forward_selfcheck: milestone-6 end-to-end forward-loop smoke. Constructs a
# layer-capped QwenModel (ARCAINE_QWEN_MAX_LAYERS, default 4 here) and drives
# forward_tokens directly with synthetic token ids (bypassing the BPE tokenizer,
# which still has the null-bos_token bug). Checks logits size/finite/nonzero/
# not-degenerate/deterministic, decode finite, and reset_cache reproducibility.
# Links the full COMMON+ARCH sources (same as gemma4 minus main.cpp) so all
# transitive deps (chat_template/tokenizer/image_proc/audio_proc + registrars)
# resolve; needs the spirv-ext option (nvfp4.hpp) and onnxruntime/soxr (audio deps).
add_executable(qwen_forward_selfcheck src/qwen_forward_selfcheck.cpp ${COMMON_SRCS} ${ARCH_SRCS})
target_compile_options(qwen_forward_selfcheck PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(qwen_forward_selfcheck PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(qwen_forward_selfcheck PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(qwen_forward_selfcheck PRIVATE DNNL::dnnl soxr onnxruntime)
set_target_properties(qwen_forward_selfcheck PROPERTIES
    BUILD_RPATH "${ORT_DIR}/lib"
    INSTALL_RPATH "${ORT_DIR}/lib")

# ---------------------------------------------------------------------------
# NVFP4 expert-kernel roofline benchmark (random weights, no model). Header-only
# GPU deps (engine/buffer/nvfp4/device_select); links only oneDNN. The spirv-ext
# link option is required for __spirv_SubgroupMatrixMultiplyAccumulateINTEL
# (xe2/xe2v2 DPAS kernels), matching diffusion_gemma/diffusion_bench.
add_executable(nvfp4_roofline_bench src/nvfp4_roofline_bench.cpp)
target_compile_options(nvfp4_roofline_bench PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(nvfp4_roofline_bench PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(nvfp4_roofline_bench PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(nvfp4_roofline_bench PRIVATE DNNL::dnnl)

# ---------------------------------------------------------------------------
# Phase 0, Probe 1 — oneDNN batched f4_e2m1 matmul feasibility (host-side).
# SYCL + oneDNN only; needs the engine singleton. Mirrors get_device_props:70-73.
add_executable(nvfp4_batched_probe src/nvfp4_batched_probe.cpp)
target_compile_options(nvfp4_batched_probe PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(nvfp4_batched_probe PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_libraries(nvfp4_batched_probe PRIVATE DNNL::dnnl)

# ---------------------------------------------------------------------------
# Phase 1 numerical-correctness probe for matmul_nvfp4_packed_batched.
# Hand-verifiable deterministic case (all-1.0 A, K-contiguous 2.0/even-k W).
# Includes nvfp4.hpp (DPAS intrinsics) => needs the spirv-ext link option.
add_executable(nvfp4_batched_check src/nvfp4_batched_check.cpp)
target_compile_options(nvfp4_batched_check PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(nvfp4_batched_check PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(nvfp4_batched_check PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(nvfp4_batched_check PRIVATE DNNL::dnnl)

# ---------------------------------------------------------------------------
# Phase 0, Probe 2 — hand-rolled Intel block-2D-load primitive (device-side).
# Uses __builtin_IB_subgroup_createBlock2DAddressPayload + lsc_load_block2d.ugm
# inline-asm. The spirv-ext link option is required for the Xe subgroup builtins,
# matching nvfp4_roofline_bench / diffusion_bench above.
add_executable(nvfp4_blockload_probe src/nvfp4_blockload_probe.cpp)
target_compile_options(nvfp4_blockload_probe PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(nvfp4_blockload_probe PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(nvfp4_blockload_probe PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(nvfp4_blockload_probe PRIVATE DNNL::dnnl)

# Phase-2 probe: esimd::dpas operand/layout determination for the full-ESIMD
# xe2v3 kernel (lsc_load_2d + esimd::dpas). Mirrors the blockload probe.
add_executable(nvfp4_dpas_probe src/nvfp4_dpas_probe.cpp)
target_compile_options(nvfp4_dpas_probe PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(nvfp4_dpas_probe PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_libraries(nvfp4_dpas_probe PRIVATE DNNL::dnnl)

# Phase-2 probe: determine esimd::dpas operand-distribution semantics across the
# 16-lane subgroup (per-lane independent vs. N-distributed) by giving lanes
# DIFFERENT A/B. Decides the lane->work mapping for the full-ESIMD xe2v3 kernel.
add_executable(nvfp4_dpas_dist_probe src/nvfp4_dpas_dist_probe.cpp)
target_compile_options(nvfp4_dpas_dist_probe PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(nvfp4_dpas_dist_probe PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_libraries(nvfp4_dpas_dist_probe PRIVATE DNNL::dnnl)

# Phase-2 isolation probe: replicate xe2v3's single-tile data path with known
# data and dump lane-0's dequanted A / VNNI B / result c vs host ground-truth.
# Locates the v3 correctness bug (A-load / B-load / dpas stage).
add_executable(nvfp4_xe2v3_isolate src/nvfp4_xe2v3_isolate.cpp)
target_compile_options(nvfp4_xe2v3_isolate PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(nvfp4_xe2v3_isolate PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_libraries(nvfp4_xe2v3_isolate PRIVATE DNNL::dnnl)

# Phase-2 probe: validate vectorized (ESIMD simd) arithmetic e2m1 dequant vs the
# scalar nvfp4_e2m1_fast LUT, as the foundation for the v4 vectorized-dequant
# kernel. Uses only esimd::simd ops (convert/merge/operators), no SPIRV intrinsic.
add_executable(nvfp4_vec_dequant_probe src/nvfp4_vec_dequant_probe.cpp)
target_compile_options(nvfp4_vec_dequant_probe PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(nvfp4_vec_dequant_probe PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_libraries(nvfp4_vec_dequant_probe PRIVATE DNNL::dnnl)
