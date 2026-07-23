// src/arcaine_kbench.cpp
//
// Unified kernel benchmark dispatcher. Each kernel bench registers itself via
// REGISTER_BENCH() in its own TU; this main only enumerates the registry and
// forwards argv (shifted past the bench name) to the selected bench's run().
//
// Run:
//   ./build/arcaine_kbench --list           # list registered benches
//   ./build/arcaine_kbench --help            # list with descriptions
//   ./build/arcaine_kbench <name> [opts]     # run a bench; --help for per-bench opts
//
// To add a new kernel bench: drop a .cpp in src/bench/ that defines a static
// `int run(int argc, char** argv)` and ends with REGISTER_BENCH("name", ..., run),
// then append it to KBENCH_SRCS in CMakeLists.txt. No dispatch table edit here.

#include "common/bench/registry.hpp"

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
