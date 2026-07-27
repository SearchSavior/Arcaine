# cmake/ArcaineModel.cmake
# Helper for registering a per-model build target. Called from each model's
# CMakeLists.txt after add_library().
#
#   arcaine_add_model(NAME <dir> TARGET <t> REGISTRATION_DEFINE <D>)
#
# Validates name uniqueness, records it, and propagates the registration define
# to the target's compile definitions. (The define is ALSO set globally via
# add_compile_definitions in the top-level CMakeLists so that builtin_models.cpp
# and the legacy register_builtins.cpp see it.)
set_property(GLOBAL PROPERTY ARCAINE_REGISTERED_MODEL_NAMES "")

function(arcaine_add_model)
    cmake_parse_arguments(ARG "" "NAME;TARGET;REGISTRATION_DEFINE" "" ${ARGN})
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
endfunction()
