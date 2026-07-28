# cmake/ArcaineModel.cmake
# Helper for registering a per-model build target. Called from each model's
# CMakeLists.txt after add_library().
#
#   arcaine_add_model(NAME <dir> TARGET <t> REGISTRATION_DEFINE <D>
#       [MODEL_BENCH_SOURCES <src> ...]     # modeling/<m>/benchmarks/model_bench.cpp ...
#       [KERNEL_BENCH_SOURCES <src> ...])   # modeling/<m>/benchmarks/<kernel>_bench.cpp ...
#
# Validates name uniqueness, records it, and propagates the registration define
# to the target's compile definitions. (The define is ALSO set globally via
# add_compile_definitions in the top-level CMakeLists so that builtin_models.cpp
# and the legacy register_builtins.cpp see it.)
#
# MODEL_BENCH_SOURCES / KERNEL_BENCH_SOURCES are collected (resolved against
# CMAKE_CURRENT_SOURCE_DIR) into the GLOBAL properties ARCAINE_MODEL_BENCH_SRCS
# / ARCAINE_KERNEL_BENCH_SRCS. The top-level CMake reads those properties after
# all model subdirectories are processed and builds the central arcaine_mbench
# / arcaine_kbench executables from the collected sources. Bench sources live
# beside their model but are compiled into the central bench binaries (their
# headers are model-local header-only kernels), so they do not link the model
# library.
set_property(GLOBAL PROPERTY ARCAINE_REGISTERED_MODEL_NAMES "")

function(arcaine_add_model)
    cmake_parse_arguments(ARG "" "NAME;TARGET;REGISTRATION_DEFINE"
        "MODEL_BENCH_SOURCES;KERNEL_BENCH_SOURCES" ${ARGN})
    if(NOT ARG_NAME OR NOT ARG_TARGET OR NOT ARG_REGISTRATION_DEFINE)
        message(FATAL_ERROR
            "arcaine_add_model requires NAME, TARGET, and REGISTRATION_DEFINE")
    endif()
    get_property(_known GLOBAL PROPERTY ARCAINE_REGISTERED_MODEL_NAMES)
    if(ARG_NAME IN_LIST _known)
        message(FATAL_ERROR "Model name '${ARG_NAME}' registered more than once")
    endif()
    set_property(GLOBAL APPEND PROPERTY ARCAINE_REGISTERED_MODEL_NAMES ${ARG_NAME})
    target_compile_definitions(${ARG_TARGET} PUBLIC ${ARG_REGISTRATION_DEFINE})

    # Collect per-model benchmark sources (resolved to absolute paths) into
    # global properties consumed by the central arcaine_mbench/arcaine_kbench.
    foreach(src ${ARG_MODEL_BENCH_SOURCES})
        set_property(GLOBAL APPEND PROPERTY ARCAINE_MODEL_BENCH_SRCS
            "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
    endforeach()
    foreach(src ${ARG_KERNEL_BENCH_SOURCES})
        set_property(GLOBAL APPEND PROPERTY ARCAINE_KERNEL_BENCH_SRCS
            "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
    endforeach()
endfunction()
