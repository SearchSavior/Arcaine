// Generic textual Gen-ISA dump helper. Any ngen kernel body can be emitted as
// textual assembly by passing it to dumpAsm(); the AsmCodeGenerator prints the
// instruction stream instead of encoding a binary.
//
// Usage:
//   dumpAsm("my_kernel.asm", ngen::HW::Xe2, [](ngen::AsmCodeGenerator &g) {
//       myKernelBody(g, ...);
//   });
//
// Notes:
// - This TU must be compiled with NGEN_ASM defined (changes ngen class layout;
//   keep it out of the binary generator TU for ODR safety).
// - AsmCodeGenerator::fixup is a no-op, so printed operand regions can differ
//   from the real binary. Verify binary regions with IGA on the zebin.

#pragma once

#ifndef NGEN_ASM
#define NGEN_ASM
#endif

#include <fstream>
#include <iostream>

#include "ngen.hpp"

template <typename Body>
static void dumpAsm(const char *path, ngen::HW hw, Body body) {
    ngen::AsmCodeGenerator g(hw, 0);
    body(g);
    std::ofstream os(path);
    g.getCode(os);
    std::cout << "wrote " << path << "\n";
}
