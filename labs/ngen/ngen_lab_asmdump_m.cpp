// Assembly dump for the M-major W4A8 kernel (NGEN_ASM variant).
#include <fstream>
#include <iostream>
#include <cstdlib>
#ifndef NGEN_ASM
#define NGEN_ASM
#endif
#include "ngen.hpp"
#include "ngen_dpas_w4a8_gemm_m.hpp"

int main(int argc, char **argv) {
    using namespace ngen;
    // Env: NGEN_LAB_W4A8M_* knobs like the harness (only shape/mt/pipe affect
    // the body).
    auto env = [](const char *n, int d) {
        const char *v = std::getenv(n); return v ? std::atoi(v) : d;
    };
    static const int presetK[] = {5120, 6144, 5120, 5120, 17408};
    static const int presetN[] = {14336, 5120, 16384, 34816, 5120};
    int shape = env("NGEN_LAB_W4A8M_SHAPE", 1);
    ngen_lab::w4a8m::kN = presetN[shape - 1];
    ngen_lab::w4a8m::kK = presetK[shape - 1];
    ngen_lab::w4a8m::kKt = ngen_lab::w4a8m::kK / 32;
    ngen_lab::w4a8m::kMt = env("NGEN_LAB_W4A8M_MT", 2);
    ngen_lab::w4a8m::kPipeDepth = env("NGEN_LAB_W4A8M_PIPE", 0);
    ngen_lab::w4a8m::kWgY = 1;
    ngen_lab::w4a8m::kWgZ = 1;

    struct G : AsmCodeGenerator {
        using AsmCodeGenerator::AsmCodeGenerator;
        using AsmCodeGenerator::setDefaultAutoSWSB;
        using AsmCodeGenerator::setDefaultNoMask;
    } g(HW::Xe2, 0);
    g.setDefaultAutoSWSB(true);
    g.setDefaultNoMask(true);
    ngen_lab::w4a8m::gemmKernelBody(g, GRF(1).uq(0), GRF(1).uq(1),
            GRF(1).uq(2), GRF(1).uq(3), GRF(1).uq(4), GRF(0).ud(1),
            GRF(0).ud(2), GRF(2), GRF(3));
    const char *path = argc > 1 ? argv[1] : "w4a8m_kernel.asm";
    std::ofstream os(path);
    g.getCode(os);
    std::cout << "wrote " << path << "\n";
    return 0;
}
