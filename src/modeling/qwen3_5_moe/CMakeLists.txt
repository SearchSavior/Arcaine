# src/modeling/qwen3_5_moe/CMakeLists.txt
# Qwen3.5 MoE — autoregressive, NVFP4 MoE (model_type qwen3_5_moe_text).
# Owns its complete inference path: registration, service, session, request
# mapper, output parser, decode loop, sampling, cache, and kernels. MoE
# routing and expert execution live inside the architecture engine
# (model.cpp / moe.hpp); this module owns the generation orchestration.
add_library(arcaine_model_qwen3_5_moe STATIC
    loader.cpp
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
target_include_directories(arcaine_model_qwen3_5_moe PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_options(arcaine_model_qwen3_5_moe PRIVATE
    -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(arcaine_model_qwen3_5_moe PRIVATE
    -fsycl ${ARCAINE_SYCL_TARGET_FLAG}
    -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(arcaine_model_qwen3_5_moe PUBLIC
    DNNL::dnnl arcaine_contracts arcaine_preprocessing)
arcaine_add_model(NAME qwen3_5_moe
    TARGET arcaine_model_qwen3_5_moe
    REGISTRATION_DEFINE ARCAINE_MODEL_QWEN3_5_MOE
    MODEL_BENCH_SOURCES
        benchmarks/model_bench.cpp
    KERNEL_BENCH_SOURCES
        benchmarks/nvfp4_mlp_bench.cpp
        benchmarks/int4_moe_bench.cpp
        benchmarks/gdn_bench.cpp
        benchmarks/attention_bench.cpp)
