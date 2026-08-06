// Assembly-dump executable: compiled with NGEN_ASM so ngen emits textual Gen
// ISA instead of binaries. Kept as a separate binary from ngen_lab because
// NGEN_ASM changes ngen class definitions (ODR).
//
// Usage: ngen_lab_asmdump [dpas.asm] [scalar.asm]

#include <fstream>
#include <iostream>

#ifndef NGEN_ASM
#define NGEN_ASM
#endif
#include "ngen.hpp"

#include "ngen_dpas_8x8_matmul_bf16_16x16x32.hpp"

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

int main(int argc, char **argv) {
    const char *dpasPath = (argc > 1) ? argv[1] : "dpas_kernel.asm";
    const char *scalarPath = (argc > 2) ? argv[2] : "scalar_kernel.asm";

    dump(dpasPath, [](ngen::AsmCodeGenerator &g) {
        ngen_lab::dpasKernelBody(g, symArg(0), symArg(1), symArg(2),
                ngen::GRF(1).ud(6), ngen::GRF(0).ud(1));
    });
    dump(scalarPath, [](ngen::AsmCodeGenerator &g) {
        ngen_lab::scalarKernelBody(g, symArg(0), symArg(1), symArg(2),
                ngen::GRF(1).ud(6), ngen::GRF(0).ud(1));
    });
    return 0;
}
