// src/apps/kernel_bench/main.cpp
//
// Unified kernel-benchmark dispatcher. Each kernel bench registers itself via
// REGISTER_BENCH() in its own TU (now living beside its model under
// src/modeling/<m>/benchmarks/, collected into arcaine_kbench via
// arcaine_add_model's KERNEL_BENCH_SOURCES); this main only enumerates the
// registry and forwards argv (shifted past the bench name) to the selected
// bench's run().
//
//   ./build/arcaine_kbench --list           # list registered benches
//   ./build/arcaine_kbench --help            # list with descriptions
//   ./build/arcaine_kbench <name> [opts]     # run a bench; --help for per-bench opts

#include "benchmarks/registry.hpp"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    const auto& regs = arcaine::bench::Registry::get().all();
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: arcaine_kbench <bench> [opts]\n"
            "Run 'arcaine_kbench --list' to see registered benches.\n");
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help") {
        std::printf("Usage: arcaine_kbench <bench> [opts]\n\nRegistered benches:\n");
        for (const auto& b : regs)
            std::printf("  %-32s %s\n", b.name, b.description);
        return 0;
    }
    if (cmd == "--list") {
        for (const auto& b : regs) std::printf("%s\n", b.name);
        return 0;
    }
    for (const auto& b : regs) {
        if (b.name == cmd) {
            // argv[0] becomes the bench name so per-bench usage() prints cleanly,
            // e.g. "Usage: qwen35-attention [opts]" rather than "Usage: arcaine_kbench".
            return b.run(argc - 1, argv + 1);
        }
    }
    std::fprintf(stderr, "unknown bench '%s' (run 'arcaine_kbench --list')\n", cmd.c_str());
    return 2;
}
