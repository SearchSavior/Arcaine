# src/modeling/qwen3_5/CMakeLists.txt
# Qwen3.5 — autoregressive, DeltaNet, FP8/NVFP4, multimodal (vision).
# Owns its complete inference path: registration, service, session, request
# mapper, output parser, decode loop, sampling, cache, and kernels.
add_library(arcaine_model_qwen3_5 STATIC
    loader.cpp
    vision.cpp
    model.cpp
    registration.cpp
    service.cpp
    session.cpp
    request_mapper.cpp
    output_parser.cpp
    inference/decode_loop.cpp
    inference/sampling.cpp
    inference/cache.cpp
    kernels/manifest.cpp
)
target_include_directories(arcaine_model_qwen3_5 PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_options(arcaine_model_qwen3_5 PRIVATE
    -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(arcaine_model_qwen3_5 PRIVATE
    -fsycl ${ARCAINE_SYCL_TARGET_FLAG}
    -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(arcaine_model_qwen3_5 PUBLIC
    DNNL::dnnl arcaine_contracts arcaine_preprocessing)
arcaine_add_model(NAME qwen3_5
    TARGET arcaine_model_qwen3_5
    REGISTRATION_DEFINE ARCAINE_MODEL_QWEN3_5
    MODEL_BENCH_SOURCES
        benchmarks/model_bench.cpp
    KERNEL_BENCH_SOURCES
        benchmarks/attention_bench.cpp
        benchmarks/deltanet_bench.cpp
        benchmarks/delta_decode_fusion_bench.cpp
        benchmarks/int4_fusion_bench.cpp)
