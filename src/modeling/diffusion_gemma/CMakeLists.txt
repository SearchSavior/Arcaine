# src/modeling/diffusion_gemma/CMakeLists.txt
# DiffusionGemma 26B-A4B — block-diffusion MoE. Owns its complete inference
# path: registration, service, session, request mapper, output parser,
# denoising loop, and kernels. The diffusion-specific GPU helpers
# (placement / expert_execution / memory_plan) and profiling live inside this
# module's inference/ + the shared runtime/profiling/.
add_library(arcaine_model_diffusion_gemma STATIC
    loader.cpp
    moe.cpp
    activation_plan.cpp
    attention.cpp
    generate.cpp
    model.cpp
    registration.cpp
    service.cpp
    session.cpp
    request_mapper.cpp
    output_parser.cpp
    inference/denoising_loop.cpp
    inference/placement.cpp
    inference/expert_execution.cpp
    inference/memory_plan.cpp
    kernels/manifest.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/profiling/profile.cpp
)
target_include_directories(arcaine_model_diffusion_gemma PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_options(arcaine_model_diffusion_gemma PRIVATE
    -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
target_link_options(arcaine_model_diffusion_gemma PRIVATE
    -fsycl ${ARCAINE_SYCL_TARGET_FLAG}
    -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
target_link_libraries(arcaine_model_diffusion_gemma PUBLIC
    DNNL::dnnl arcaine_contracts arcaine_preprocessing)
arcaine_add_model(NAME diffusion_gemma
    TARGET arcaine_model_diffusion_gemma
    REGISTRATION_DEFINE ARCAINE_MODEL_DIFFUSION_GEMMA)
