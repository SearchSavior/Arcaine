# Stable inference contracts, service interfaces, and registry.
# Pure C++ — no SYCL kernels, no model implementations. Linked by apps and by
# every model target.
add_library(arcaine_contracts STATIC
    registry/model_registry.cpp
    registry/builtin_models.cpp)
target_include_directories(arcaine_contracts PUBLIC ${CMAKE_SOURCE_DIR}/src)
# builtin_models.cpp references each model module's register_model(); those
# symbols are provided by the enabled per-model libraries (linked by apps).
