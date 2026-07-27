# src/modeling/gemma4_unified/CMakeLists.txt
# Gemma4 Unified — autoregressive, multimodal (image+audio), BF16/INT4-AWQ.
# Owns its complete inference path: registration, service, session, request
# mapper, output parser, decode loop, sampling, cache, and kernels.
add_library(arcaine_model_gemma4_unified STATIC
    loader.cpp
    attention.cpp
    ffn.cpp
    vision_embedder.cpp
    audio_embedder.cpp
    model.cpp
    register.cpp
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
target_include_directories(arcaine_model_gemma4_unified PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_options(arcaine_model_gemma4_unified PRIVATE
    -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(arcaine_model_gemma4_unified PRIVATE
    -fsycl ${ARCAINE_SYCL_TARGET_FLAG}
    -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
# arcaine_contracts: ModelService/InferenceSession/ModelRegistry (registration).
# arcaine_preprocessing: tokenizer, chat template, media, quant loader.
# soxr + onnxruntime come via arcaine_preprocessing (audio/VAD path).
target_link_libraries(arcaine_model_gemma4_unified PUBLIC
    DNNL::dnnl arcaine_contracts arcaine_preprocessing)
arcaine_add_model(NAME gemma4_unified
    TARGET arcaine_model_gemma4_unified
    REGISTRATION_DEFINE ARCAINE_MODEL_GEMMA4_UNIFIED)
