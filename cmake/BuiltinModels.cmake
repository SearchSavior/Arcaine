# cmake/BuiltinModels.cmake
# Resolves ARCAINE_MODELS and adds only enabled model subdirectories. Sets
# ARCAINE_AR_MODEL_LIBS, ARCAINE_DIFFUSION_MODEL_LIB, and ARCAINE_MODEL_LIBS in
# the parent scope for the app targets to link.
#
# Included by src/modeling/CMakeLists.txt.
set(ARCAINE_AR_MODEL_LIBS "")
set(ARCAINE_DIFFUSION_MODEL_LIB "")
foreach(m ${ARCAINE_MODELS})
    add_subdirectory(${m})
    if(m STREQUAL "diffusion_gemma")
        list(APPEND ARCAINE_DIFFUSION_MODEL_LIB arcaine_model_diffusion_gemma)
    else()
        list(APPEND ARCAINE_AR_MODEL_LIBS arcaine_model_${m})
    endif()
endforeach()

set(ARCAINE_AR_MODEL_LIBS       ${ARCAINE_AR_MODEL_LIBS}       PARENT_SCOPE)
set(ARCAINE_DIFFUSION_MODEL_LIB ${ARCAINE_DIFFUSION_MODEL_LIB} PARENT_SCOPE)
set(ARCAINE_MODEL_LIBS          ${ARCAINE_AR_MODEL_LIBS} ${ARCAINE_DIFFUSION_MODEL_LIB} PARENT_SCOPE)
