# src/benchmarks/CMakeLists.txt
# Central benchmark infrastructure: the REGISTER_BENCH kernel-bench registry +
# timing helpers (header-only) and the model-bench registry. Exposed as an
# INTERFACE library so arcaine_mbench / arcaine_kbench and per-model benchmark
# sources can include "benchmarks/registry.hpp" / "benchmarks/model_bench_registry.hpp"
# and link the headers' translation-unit dependencies.
add_library(arcaine_benchmarks INTERFACE)
target_include_directories(arcaine_benchmarks INTERFACE ${CMAKE_SOURCE_DIR}/src)
