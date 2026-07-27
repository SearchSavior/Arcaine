# src/modeling/CMakeLists.txt
# Adds only the model subdirectories selected by ARCAINE_MODELS and exports the
# enabled model library targets via ARCAINE_*_MODEL_LIBS (PARENT_SCOPE).
include(${CMAKE_SOURCE_DIR}/cmake/BuiltinModels.cmake)
