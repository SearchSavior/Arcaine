# Project-level preprocessing infrastructure: tokenizer, chat-template
# renderer, media loaders (image/audio), the TokenizerBridge wrapper, and the
# quantized weight loader. Pure infrastructure (no inference, no model logic);
# linked by every model target and the apps.
add_library(arcaine_preprocessing STATIC
    tokenizer.cpp
    chat_template.cpp
    unicode.cpp
    unicode-data.cpp
    image_processor.cpp
    audio_processor.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/chat.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/quantization/quant_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/gpu/nvfp4_session.cpp)
target_include_directories(arcaine_preprocessing PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_options(arcaine_preprocessing PRIVATE
    -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(arcaine_preprocessing PRIVATE
    -fsycl ${ARCAINE_SYCL_TARGET_FLAG}
    -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
# soxr + onnxruntime for the audio/VAD path; DNNL for nvfp4_session. Propagated
# to model libs and apps that link this target.
target_link_libraries(arcaine_preprocessing PUBLIC DNNL::dnnl soxr onnxruntime)
