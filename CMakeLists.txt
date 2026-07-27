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
# Model selection. ARCAINE_MODELS selects which model implementations are
# compiled into this build (independent per-model CMake targets). A disabled
# model's sources are not compiled or linked; loading a disabled model at
# runtime yields a clear error from ModelRegistry.
set(ARCAINE_KNOWN_MODELS "gemma4_unified;qwen3_5;qwen3_5_moe;diffusion_gemma")
set(ARCAINE_MODELS "${ARCAINE_KNOWN_MODELS}" CACHE STRING
    "Model implementations compiled into Arcaine (semicolon-separated)")
foreach(m ${ARCAINE_MODELS})
    if(NOT m IN_LIST ARCAINE_KNOWN_MODELS)
        message(FATAL_ERROR
            "Unknown model '${m}' in ARCAINE_MODELS. Known: ${ARCAINE_KNOWN_MODELS}")
    endif()
endforeach()
# Global registration defines — visible to register_builtins.cpp (and, from
# Phase 6 on, builtin_models.cpp) so the #ifdef-gated registration calls match
# the set of models actually compiled into this build.
foreach(m ${ARCAINE_MODELS})
    string(TOUPPER "${m}" _M)
    add_compile_definitions(ARCAINE_MODEL_${_M})
endforeach()

include(${CMAKE_SOURCE_DIR}/cmake/ArcaineModel.cmake)

enable_testing()
add_subdirectory(src/inference)
add_subdirectory(src/preprocessing)   # tokenizer/chat-template/media/quant loader (no model logic)
add_subdirectory(src/modeling)        # adds only enabled model subdirs; sets ARCAINE_*_MODEL_LIBS
add_subdirectory(tests)

# ---------------------------------------------------------------------------
# Shared app infrastructure. Project preprocessing (tokenizer, chat template,
# media, quant loader) and the inference contracts live in the
# arcaine_preprocessing / arcaine_contracts libraries linked below; model
# implementations live in per-model static libraries under src/modeling/<m>/,
# linked via ARCAINE_*_MODEL_LIBS.
#
# AR apps still drive the legacy ModelRegistry (common/registry.cpp +
# register_builtins.cpp) until the Phase 7 server cutover.
set(ARCAINE_COMMON_SRCS
    src/register_builtins.cpp
    src/common/registry.cpp
)

set(ARCAINE_SYCL_LIB_FLAGS
    -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
set(ARCAINE_SYCL_LINK_FLAGS
    -fsycl ${ARCAINE_SYCL_TARGET_FLAG}
    -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)

# ---------------------------------------------------------------------------
add_executable(get_device_props src/utils/gpu_check.cpp)
target_compile_options(get_device_props PRIVATE -fsycl -fno-sycl-id-queries-fit-in-int ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(get_device_props PRIVATE -fsycl ${ARCAINE_SYCL_TARGET_FLAG})
target_link_libraries(get_device_props PRIVATE DNNL::dnnl)
set_target_properties(get_device_props PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.agents/skills/check-available-hardware")

# ---------------------------------------------------------------------------
# CLI (AR frontends, model-agnostic via the legacy ModelRegistry).
add_executable(gemma4 src/main.cpp ${ARCAINE_COMMON_SRCS})
target_compile_options(gemma4 PRIVATE ${ARCAINE_SYCL_LIB_FLAGS})
target_link_options(gemma4 PRIVATE ${ARCAINE_SYCL_LINK_FLAGS})
target_link_libraries(gemma4 PRIVATE
    arcaine_preprocessing ${ARCAINE_AR_MODEL_LIBS} DNNL::dnnl soxr onnxruntime)
set_target_properties(gemma4 PROPERTIES
    BUILD_RPATH "${ORT_DIR}/lib"
    INSTALL_RPATH "${ORT_DIR}/lib")

# ---------------------------------------------------------------------------
# --chat-template-kwargs parsing + chat-template merge. SYCL-free logic, but
# every target must pass -fsycl under the global sycl-targets flag. Pass a
# model dir as argv[1] to exercise the render-merge assertions.
add_executable(test_chat_template_kwargs
    src/utils/test_chat_template_kwargs.cpp)
target_compile_options(test_chat_template_kwargs PRIVATE ${ARCAINE_SYCL_LIB_FLAGS})
target_link_options(test_chat_template_kwargs PRIVATE ${ARCAINE_SYCL_LINK_FLAGS})
target_link_libraries(test_chat_template_kwargs PRIVATE arcaine_preprocessing DNNL::dnnl)

# ---------------------------------------------------------------------------
# Inference benchmark. Goes through the legacy ModelRegistry, so it links the
# same AR model libs + common infra as the CLI.
add_executable(arcaine_mbench src/arcaine_mbench.cpp ${ARCAINE_COMMON_SRCS})
target_compile_options(arcaine_mbench PRIVATE ${ARCAINE_SYCL_LIB_FLAGS})
target_link_options(arcaine_mbench PRIVATE ${ARCAINE_SYCL_LINK_FLAGS})
target_link_libraries(arcaine_mbench PRIVATE
    arcaine_preprocessing ${ARCAINE_AR_MODEL_LIBS} DNNL::dnnl soxr onnxruntime)
set_target_properties(arcaine_mbench PROPERTIES
    BUILD_RPATH "${ORT_DIR}/lib"
    INSTALL_RPATH "${ORT_DIR}/lib")

# ---------------------------------------------------------------------------
# DiffusionGemma driver + throughput bench. Gated on diffusion_gemma being
# enabled (the diffusion lib is self-contained for its GPU helpers).
if(ARCAINE_DIFFUSION_MODEL_LIB)
add_executable(diffusion_gemma src/diffusion_main.cpp src/utils/viz.cpp)
target_compile_options(diffusion_gemma PRIVATE ${ARCAINE_SYCL_LIB_FLAGS})
target_link_options(diffusion_gemma PRIVATE ${ARCAINE_SYCL_LINK_FLAGS})
target_link_libraries(diffusion_gemma PRIVATE
    arcaine_preprocessing ${ARCAINE_DIFFUSION_MODEL_LIB} DNNL::dnnl)

add_executable(diffusion_bench src/diffusion_bench.cpp)
target_compile_options(diffusion_bench PRIVATE ${ARCAINE_SYCL_LIB_FLAGS})
target_link_options(diffusion_bench PRIVATE ${ARCAINE_SYCL_LINK_FLAGS})
target_link_libraries(diffusion_bench PRIVATE
    arcaine_preprocessing ${ARCAINE_DIFFUSION_MODEL_LIB} DNNL::dnnl)
endif()

# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP API server. Talks only to ModelService; one execution
# path for every model (autoregressive or block-diffusion). Requires at least
# one enabled model (ARCAINE_MODEL_LIBS).
if(ARCAINE_MODEL_LIBS)
add_executable(arcaine_server
    src/apps/server/main.cpp
    src/apps/server/server_app.cpp
    src/apps/server/app_state.cpp
    src/apps/server/openai/error_encoder.cpp
    src/apps/server/openai/request_decoder.cpp
    src/apps/server/openai/response_encoder.cpp
    src/apps/server/openai/sse_event_sink.cpp
    src/apps/server/routes/chat_completions.cpp
    src/apps/server/routes/models.cpp)
target_compile_options(arcaine_server PRIVATE ${ARCAINE_SYCL_LIB_FLAGS})
target_link_options(arcaine_server PRIVATE ${ARCAINE_SYCL_LINK_FLAGS})
target_link_libraries(arcaine_server PRIVATE
    arcaine_contracts arcaine_preprocessing ${ARCAINE_MODEL_LIBS}
    DNNL::dnnl soxr onnxruntime)
set_target_properties(arcaine_server PROPERTIES
    BUILD_RPATH "${ORT_DIR}/lib"
    INSTALL_RPATH "${ORT_DIR}/lib")
endif()

# ---------------------------------------------------------------------------
# Unified kernel benchmark. Each kernel bench registers itself via
# REGISTER_BENCH() in its TU; a new kernel bench is added by dropping a .cpp
# into src/bench/ and appending it to KBENCH_SRCS. Each bench keeps its own
# env-var AB gating.
#   ./build/arcaine_kbench --list
#   ./build/arcaine_kbench <name> [opts]
set(KBENCH_SRCS
    src/arcaine_kbench.cpp
    src/bench/qwen35_nvfp4_mlp_bench.cpp
    src/bench/qwen35_attention_bench.cpp
    src/bench/qwen35_deltanet_bench.cpp
    src/bench/qwen35_delta_decode_fusion_bench.cpp
    src/bench/nvfp4_roofline_bench.cpp
    src/bench/diffusion_int4_fusion_bench.cpp)
add_executable(arcaine_kbench ${KBENCH_SRCS})
target_compile_options(arcaine_kbench PRIVATE ${ARCAINE_SYCL_LIB_FLAGS})
target_link_options(arcaine_kbench PRIVATE ${ARCAINE_SYCL_LINK_FLAGS})
target_link_libraries(arcaine_kbench PRIVATE arcaine_preprocessing DNNL::dnnl)
