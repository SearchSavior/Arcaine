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
include_directories(src third_party third_party/minja/include)

# The dev image also ships older oneDNN builds through oneAPI/OpenVINO. Keep the
# source-built oneDNN selected by find_package() ahead of LD_LIBRARY_PATH entries
# so newer symbols such as dnnl_primitive_attr_set_scales_v3 resolve at runtime.
add_link_options(-Wl,--disable-new-dtags)

# libsoxr (static, vendored prebuilt — header + libsoxr.a only, like minja/nlohmann)
add_library(soxr STATIC IMPORTED)
set_target_properties(soxr PROPERTIES
    IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/third_party/soxr/libsoxr.a"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/third_party/soxr"
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
# Plain-C++ Unicode tables + helpers (ported from llama.cpp, MIT) used by
# tokenizer.cpp for Qwen2/GPT-2 byte-level BPE. Linked into every target that
# compiles tokenizer.cpp.
set(UNICODE_SRCS
    src/common/preprocess/unicode.cpp
    src/common/preprocess/unicode-data.cpp
)

set(COMMON_SRCS
    src/register_builtins.cpp
    src/common/registry.cpp
    src/common/io/quant_loader.cpp
    src/common/gpu/nvfp4_session.cpp
    src/common/preprocess/tokenizer.cpp
    src/common/preprocess/chat_template.cpp
    src/common/preprocess/image_proc.cpp
    src/common/preprocess/audio_proc.cpp
    ${UNICODE_SRCS}
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

set(QWEN35_SRCS
    src/modeling/qwen3_5/loader.cpp
    src/modeling/qwen3_5/vision.cpp
    src/modeling/qwen3_5/model.cpp
    src/modeling/qwen3_5/register.cpp
)

# All built-in architectures linked into the inference/bench binaries.
set(ARCH_SRCS ${GEMMA4_SRCS} ${QWEN_SRCS} ${QWEN35_SRCS})

# ---------------------------------------------------------------------------
add_executable(get_device_props src/utils/gpu_check.cpp)
target_compile_options(get_device_props PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(get_device_props PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_libraries(get_device_props PRIVATE DNNL::dnnl)
set_target_properties(get_device_props PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.agents/skills/check-available-hardware")

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
# --chat-template-kwargs parsing + chat-template merge. Logic is SYCL-free, but
# the global -fsycl-targets flag means every target must also pass -fsycl.
# Pass a model dir as argv[1] to exercise the render-merge assertions.
add_executable(test_chat_template_kwargs
    src/utils/test_chat_template_kwargs.cpp
    src/common/preprocess/chat_template.cpp
    src/common/preprocess/tokenizer.cpp
    ${UNICODE_SRCS})
target_compile_options(test_chat_template_kwargs PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(test_chat_template_kwargs    PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_libraries(test_chat_template_kwargs  PRIVATE DNNL::dnnl)

# ---------------------------------------------------------------------------
# Inference benchmark. Goes through the model registry, so it links the same
# architecture + common sources as the main binary.
add_executable(arcaine_mbench src/arcaine_mbench.cpp ${COMMON_SRCS} ${ARCH_SRCS})
target_compile_options(arcaine_mbench PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(arcaine_mbench   PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
# See gemma4: ARCH_SRCS / COMMON_SRCS include nvfp4.hpp-bearing loader TUs.
target_link_options(arcaine_mbench PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(arcaine_mbench PRIVATE DNNL::dnnl soxr onnxruntime)
set_target_properties(arcaine_mbench PROPERTIES
    BUILD_RPATH "${ORT_DIR}/lib"
    INSTALL_RPATH "${ORT_DIR}/lib")

# ---------------------------------------------------------------------------
# DiffusionGemma 26B-A4B (block-diffusion MoE) — hello-world driver.
# Text-only path: needs no soxr / onnxruntime preprocessing.
set(DIFFUSION_SRCS
    src/common/preprocess/tokenizer.cpp
    src/common/preprocess/chat_template.cpp
    src/common/io/quant_loader.cpp
    ${UNICODE_SRCS}
    src/common/gpu/nvfp4_session.cpp
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

# OpenAI-compatible HTTP API server. Single binary serving every registered
# architecture: dispatches by config.json::model_type at startup.
#   - "diffusion_gemma"   -> DiffusionGemmaModel (block-diffusion MoE)
#   - "gemma4_unified"    -> Gemma4Model (AR, multimodal, BF16)
#   - "qwen3_5_moe_text"  -> QwenModel (AR, NVFP4 MoE)
#   - "qwen3_5"           -> Qwen35Model (AR, FP8/NVFP4, multimodal)
# Links COMMON_SRCS + ARCH_SRCS (all arches, so the registry can construct any)
# plus DIFFUSION_SRCS (the block-diffusion path) and register_builtins.cpp
# (wires the AR arches into ModelRegistry). Needs soxr + onnxruntime for the
# Gemma4/Qwen3.5 audio + Silero VAD preprocessing paths.
add_executable(arcaine_server
    src/arcaine_server.cpp
    src/utils/gemma4_tool_call_parser.cpp
    ${DIFFUSION_SRCS}
    ${COMMON_SRCS}
    ${ARCH_SRCS}
    src/register_builtins.cpp)
target_compile_options(arcaine_server PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(arcaine_server   PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(arcaine_server PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)

# llama-bench style throughput benchmark (loads once, sweeps expert kernels).
add_executable(diffusion_bench src/diffusion_bench.cpp ${DIFFUSION_SRCS})
target_compile_options(diffusion_bench PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(diffusion_bench   PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(diffusion_bench PRIVATE -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(diffusion_bench PRIVATE DNNL::dnnl)
target_link_libraries(diffusion_gemma PRIVATE DNNL::dnnl)
target_link_libraries(arcaine_server PRIVATE DNNL::dnnl soxr onnxruntime)
set_target_properties(arcaine_server PROPERTIES
    BUILD_RPATH "${ORT_DIR}/lib"
    INSTALL_RPATH "${ORT_DIR}/lib")

# ---------------------------------------------------------------------------
# Unified kernel benchmark binary. Each kernel bench registers itself via
# REGISTER_BENCH() in its TU; a new kernel bench is added by dropping a new
# .cpp into src/bench/ and appending it to KBENCH_SRCS — no main dispatch
# table to edit. Each bench keeps its own env-var AB gating.
# Run:
#   ./build/arcaine_kbench --list
#   ./build/arcaine_kbench <name> [opts]
set(KBENCH_SRCS
    src/arcaine_kbench.cpp
    src/bench/qwen35_nvfp4_mlp_bench.cpp
    src/bench/qwen35_attention_bench.cpp
    src/bench/qwen35_deltanet_bench.cpp
    src/bench/qwen35_delta_decode_fusion_bench.cpp
    src/bench/nvfp4_roofline_bench.cpp
    src/bench/diffusion_int4_fusion_bench.cpp
    # qwen35_nvfp4_mlp_bench loads a real NVFP4 layer:
    src/common/io/quant_loader.cpp
    src/common/gpu/nvfp4_session.cpp)
add_executable(arcaine_kbench ${KBENCH_SRCS})
target_compile_options(arcaine_kbench PRIVATE
    -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(arcaine_kbench PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(arcaine_kbench PRIVATE
    -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(arcaine_kbench PRIVATE DNNL::dnnl)


