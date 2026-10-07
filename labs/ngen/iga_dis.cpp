// Disassemble a raw Gen ISA binary with IGA (Xe2).
// Usage: iga_dis <raw-binary-file>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include <iga/iga.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: iga_dis <raw.bin>\n");
        return 1;
    }
    std::ifstream is(argv[1], std::ios::binary);
    std::vector<char> buf((std::istreambuf_iterator<char>(is)),
            std::istreambuf_iterator<char>());
    if (buf.empty()) {
        std::fprintf(stderr, "empty input\n");
        return 1;
    }

    iga_context_options_t opts = IGA_CONTEXT_OPTIONS_INIT(IGA_XE2);
    iga_context_t ctx = nullptr;
    if (iga_context_create(&opts, &ctx) != IGA_SUCCESS) {
        std::fprintf(stderr, "iga_context_create failed\n");
        return 1;
    }
    iga_disassemble_options_t dopts = IGA_DISASSEMBLE_OPTIONS_INIT();
    char *text = nullptr;
    auto st = iga_context_disassemble(ctx, &dopts, buf.data(),
            (uint32_t)buf.size(), nullptr, nullptr, &text);
    std::printf("%s\n", text ? text : "(no output)");
    if (st != IGA_SUCCESS) {
        uint32_t n = 0;
        const iga_diagnostic_t *diags = nullptr;
        iga_context_get_errors(ctx, &diags, &n);
        for (uint32_t i = 0; i < n; i++) {
            const char *m = nullptr;
            uint32_t off = 0;
            iga_diagnostic_get_message(&diags[i], &m);
            iga_diagnostic_get_offset(&diags[i], &off);
            std::fprintf(stderr, "decode error @%d: %s\n", off, m);
        }
    }
    iga_context_release(ctx);
    return st == IGA_SUCCESS ? 0 : 1;
}
