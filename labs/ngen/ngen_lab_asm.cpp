// Assembly-dump executable: compiled with NGEN_ASM so ngen emits textual Gen
// ISA instead of binaries. Kept as a separate binary from the harnesses
// because NGEN_ASM changes ngen class definitions (ODR).
//
// Dumps every fixed-shape kernel (bf16, s8, s4, w8a2, w4a8) as text.
//
// Usage: ngen_lab_asmdump [bf16_dpas.asm] [bf16_scalar.asm] [s8.asm] [s4.asm]
//                         [w8a2.asm] [w4a8.asm]

#include <fstream>
#include <iostream>

#ifndef NGEN_ASM
#define NGEN_ASM
#endif
#include "ngen.hpp"

#include "ngen_dpas_8x8_matmul_bf16_16x16x32.hpp"
#include "ngen_dpas_8x8_matmul_s8_16x16x32.hpp"
#include "ngen_dpas_8x8_matmul_s4_16x16x32.hpp"
#include "ngen_dpas_8x8_matmul_w8a2_16x16x32.hpp"
#include "ngen_dpas_8x8_matmul_w4a8_16x16x32.hpp"

// Symbolic payload registers, mirroring where the binary kernel's prologue
// places the cross-thread arguments (cosmetic; readability only).
static ngen::Subregister symArg(int uqIndex) {
    return ngen::GRF(1).uq(uqIndex);
}

template <typename Body>
static void dump(const char *path, Body body) {
    ngen::AsmCodeGenerator g(ngen::HW::Xe2, 0);
    body(g);
    std::ofstream os(path);
    g.getCode(os);
    std::cout << "wrote " << path << "\n";
}

// All fixed-shape kernels share the same argument signature
// (a, b, c, iters, gid), so one call adapter covers them.
template <typename Body>
static void dumpTile(const char *path, Body body) {
    dump(path, [body](ngen::AsmCodeGenerator &g) {
        body(g, symArg(0), symArg(1), symArg(2), ngen::GRF(1).ud(6),
                ngen::GRF(0).ud(1));
    });
}

int main(int argc, char **argv) {
    auto arg = [&](int i, const char *dflt) {
        return (argc > i) ? argv[i] : dflt;
    };

    // bf16: dpas + scalar bodies.
    dump(arg(1, "bf16_dpas_kernel.asm"), [](ngen::AsmCodeGenerator &g) {
        ngen_lab::dpasKernelBody(g, symArg(0), symArg(1), symArg(2),
                ngen::GRF(1).ud(6), ngen::GRF(0).ud(1));
    });
    dump(arg(2, "bf16_scalar_kernel.asm"), [](ngen::AsmCodeGenerator &g) {
        ngen_lab::scalarKernelBody(g, symArg(0), symArg(1), symArg(2),
                ngen::GRF(1).ud(6), ngen::GRF(0).ud(1));
    });

    // Integer kernels: single dpas body each.
    dumpTile(arg(3, "s8_kernel.asm"), [](auto &g, auto a, auto b, auto c,
                                          auto it, auto gid) {
        ngen_lab::s8::dpasKernelBody(g, a, b, c, it, gid);
    });
    dumpTile(arg(4, "s4_kernel.asm"), [](auto &g, auto a, auto b, auto c,
                                          auto it, auto gid) {
        ngen_lab::s4::dpasKernelBody(g, a, b, c, it, gid);
    });
    dumpTile(arg(5, "w8a2_kernel.asm"), [](auto &g, auto a, auto b, auto c,
                                            auto it, auto gid) {
        ngen_lab::w8a2::dpasKernelBody(g, a, b, c, it, gid);
    });
    dumpTile(arg(6, "w4a8_kernel.asm"), [](auto &g, auto a, auto b, auto c,
                                            auto it, auto gid) {
        ngen_lab::w4a8::dpasKernelBody(g, a, b, c, it, gid);
    });
    return 0;
}
