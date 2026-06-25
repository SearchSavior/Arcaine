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
#   common/   — architecture-independent engine (registry, preprocessing, ...)
#   <arch>/   — one directory per model architecture; add a sibling to extend.
# ---------------------------------------------------------------------------
set(COMMON_SRCS
    src/register_builtins.cpp
    src/common/registry.cpp
    src/common/preprocess/tokenizer.cpp
    src/common/preprocess/chat_template.cpp
    src/common/preprocess/image_proc.cpp
    src/common/preprocess/audio_proc.cpp
)

set(GEMMA4_SRCS
    src/gemma4_unified/loader.cpp
    src/gemma4_unified/attention.cpp
    src/gemma4_unified/ffn.cpp
    src/gemma4_unified/vision_embedder.cpp
    src/gemma4_unified/audio_embedder.cpp
    src/gemma4_unified/model.cpp
    src/gemma4_unified/register.cpp
)

# All built-in architectures linked into the inference/bench binaries.
set(ARCH_SRCS ${GEMMA4_SRCS})

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
    src/diffusion_gemma/loader.cpp
    src/diffusion_gemma/moe.cpp
    src/common/gpu/expert_parallel.cpp
    src/common/gpu/placement.cpp
    src/diffusion_gemma/activation_plan.cpp
    src/diffusion_gemma/attention.cpp
    src/diffusion_gemma/generate.cpp
    src/diffusion_gemma/model.cpp
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
