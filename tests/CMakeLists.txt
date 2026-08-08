# Contract + registry + transport tests. Pure C++ (no SYCL, no GPU), so they
# build with the project compiler without -fsycl and run anywhere. GPU smoke
# tests are separate targets (built only when their model is enabled, run
# manually with a model dir).
include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party
                    ${CMAKE_CURRENT_SOURCE_DIR})

set(ARCAINE_TESTS
    contracts/test_capabilities
    contracts/test_descriptor
    registry/test_registry
)

foreach(test ${ARCAINE_TESTS})
    string(REPLACE "/" "_" target "arcaine_test_${test}")
    add_executable(${target} ${CMAKE_CURRENT_SOURCE_DIR}/${test}.cpp)
    target_link_libraries(${target} PRIVATE arcaine_contracts)
    add_test(NAME ${test} COMMAND ${target})
endforeach()

# Chat-template-kwargs parsing + chat-template render/merge assertions. Pure
# C++ (no SYCL); pass a model dir as argv[1] to exercise the render-merge. Runs
# against the bundled arcaine_preprocessing (tokenizer + chat template).
add_executable(arcaine_test_chat_template_kwargs
    ${CMAKE_CURRENT_SOURCE_DIR}/contracts/test_chat_template_kwargs.cpp)
target_link_libraries(arcaine_test_chat_template_kwargs PRIVATE
    arcaine_preprocessing DNNL::dnnl)
add_test(NAME contracts/chat_template_kwargs
         COMMAND arcaine_test_chat_template_kwargs)

# Pure-C++ output-parser unit tests (no GPU, no model load). Each links only
# the owning model target (which pulls in arcaine_contracts/preprocessing).
if(TARGET arcaine_model_qwen3_5)
    add_executable(arcaine_test_qwen3_5_output_parser
        ${CMAKE_CURRENT_SOURCE_DIR}/modeling/qwen3_5/test_output_parser.cpp)
    target_link_libraries(arcaine_test_qwen3_5_output_parser PRIVATE arcaine_model_qwen3_5)
    add_test(NAME modeling/qwen3_5/output_parser
             COMMAND arcaine_test_qwen3_5_output_parser)

    # Conv-state layout contract between the chunked and fused decode paths.
    # Needs a GPU but no model dir, so it runs under ctest.
    add_executable(arcaine_test_qwen3_5_conv_state_layout
        ${CMAKE_CURRENT_SOURCE_DIR}/modeling/qwen3_5/test_conv_state_layout.cpp)
    target_link_libraries(arcaine_test_qwen3_5_conv_state_layout PRIVATE
        arcaine_model_qwen3_5 DNNL::dnnl)
    target_compile_options(arcaine_test_qwen3_5_conv_state_layout PRIVATE
        -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
    target_link_options(arcaine_test_qwen3_5_conv_state_layout PRIVATE
        -fsycl ${ARCAINE_SYCL_TARGET_FLAG}
        -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
    add_test(NAME modeling/qwen3_5/conv_state_layout
             COMMAND arcaine_test_qwen3_5_conv_state_layout)

    add_executable(arcaine_smoke_qwen35
        ${CMAKE_CURRENT_SOURCE_DIR}/modeling/qwen3_5/smoke_service.cpp)
    target_link_libraries(arcaine_smoke_qwen35 PRIVATE arcaine_model_qwen3_5 DNNL::dnnl)
    target_compile_options(arcaine_smoke_qwen35 PRIVATE
        -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
    target_link_options(arcaine_smoke_qwen35 PRIVATE
        -fsycl ${ARCAINE_SYCL_TARGET_FLAG}
        -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
    set_target_properties(arcaine_smoke_qwen35 PROPERTIES
        BUILD_RPATH "${ORT_DIR}/lib" INSTALL_RPATH "${ORT_DIR}/lib")
endif()

if(TARGET arcaine_model_qwen3_5_moe)
    add_executable(arcaine_test_qwen3_5_moe_output_parser
        ${CMAKE_CURRENT_SOURCE_DIR}/modeling/qwen3_5_moe/test_output_parser.cpp)
    target_link_libraries(arcaine_test_qwen3_5_moe_output_parser PRIVATE arcaine_model_qwen3_5_moe)
    add_test(NAME modeling/qwen3_5_moe/output_parser
             COMMAND arcaine_test_qwen3_5_moe_output_parser)
endif()

if(TARGET arcaine_model_diffusion_gemma)
    add_executable(arcaine_smoke_diffusion
        ${CMAKE_CURRENT_SOURCE_DIR}/modeling/diffusion_gemma/smoke_service.cpp)
    target_link_libraries(arcaine_smoke_diffusion PRIVATE arcaine_model_diffusion_gemma DNNL::dnnl)
    target_compile_options(arcaine_smoke_diffusion PRIVATE
        -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
    target_link_options(arcaine_smoke_diffusion PRIVATE
        -fsycl ${ARCAINE_SYCL_TARGET_FLAG}
        -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
    set_target_properties(arcaine_smoke_diffusion PROPERTIES
        BUILD_RPATH "${ORT_DIR}/lib" INSTALL_RPATH "${ORT_DIR}/lib")
endif()

# Builtin-models wiring test: links all four model libs + arcaine_contracts and
# verifies register_builtin_models registers every model type through the same
# ModelRegistry. Only built when all four model targets exist (all-models build).
if(TARGET arcaine_model_gemma4_unified AND TARGET arcaine_model_qwen3_5
    AND TARGET arcaine_model_qwen3_5_moe AND TARGET arcaine_model_diffusion_gemma)
    add_executable(arcaine_test_builtin_models
        ${CMAKE_CURRENT_SOURCE_DIR}/registry/test_builtin_models.cpp)
    target_link_libraries(arcaine_test_builtin_models PRIVATE
        arcaine_contracts
        arcaine_model_gemma4_unified arcaine_model_qwen3_5
        arcaine_model_qwen3_5_moe arcaine_model_diffusion_gemma DNNL::dnnl)
    target_compile_options(arcaine_test_builtin_models PRIVATE
        -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
    target_link_options(arcaine_test_builtin_models PRIVATE
        -fsycl ${ARCAINE_SYCL_TARGET_FLAG}
        -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
    add_test(NAME registry/builtin_models COMMAND arcaine_test_builtin_models)
endif()

# GPU smoke test for the gemma4_unified ModelService (needs a GPU + model dir;
# run manually, not via ctest).
if(TARGET arcaine_model_gemma4_unified)
    add_executable(arcaine_smoke_gemma4
        ${CMAKE_CURRENT_SOURCE_DIR}/modeling/gemma4_unified/smoke_service.cpp)
    target_link_libraries(arcaine_smoke_gemma4 PRIVATE arcaine_model_gemma4_unified DNNL::dnnl)
    target_compile_options(arcaine_smoke_gemma4 PRIVATE
        -fsycl -fno-sycl-id-queries-fit-in-int -O2 ${ARCAINE_SYCL_TARGET_FLAG})
    target_link_options(arcaine_smoke_gemma4 PRIVATE
        -fsycl ${ARCAINE_SYCL_TARGET_FLAG}
        -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate)
    set_target_properties(arcaine_smoke_gemma4 PROPERTIES
        BUILD_RPATH "${ORT_DIR}/lib" INSTALL_RPATH "${ORT_DIR}/lib")
endif()
