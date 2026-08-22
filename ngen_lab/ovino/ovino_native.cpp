// Dump the driver-compiled native binary for a SPIR-V module, to disassemble
// and inspect for DPAS/XMX instructions.
// Usage: ovino_native <kernel.spv> <kernel_name> [out.bin]
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define ZE_CHECK(x) do { ze_result_t r_=(x); if(r_!=ZE_RESULT_SUCCESS){std::fprintf(stderr,"ZE err 0x%x at %s:%d\n",(unsigned)r_,__FILE__,__LINE__);std::exit(1);} } while(0)

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s kernel.spv kernel_name [out.bin]\n", argv[0]); return 1; }
    const char* spv_path = argv[1];
    const char* kname = argv[2];
    const char* out_path = argc > 3 ? argv[3] : "native.bin";
    ZE_CHECK(zeInit(0));
    uint32_t ndrv = 0;
    ZE_CHECK(zeDriverGet(&ndrv, nullptr));
    std::vector<ze_driver_handle_t> drvs(ndrv);
    ZE_CHECK(zeDriverGet(&ndrv, drvs.data()));
    uint32_t ndev = 0;
    ZE_CHECK(zeDeviceGet(drvs[0], &ndev, nullptr));
    std::vector<ze_device_handle_t> devs(ndev);
    ZE_CHECK(zeDeviceGet(drvs[0], &ndev, devs.data()));
    int devIdx = std::atoi(getenv("L0_MIN_DEVIDX") ? getenv("L0_MIN_DEVIDX") : "0");
    int gpuSeen = -1;
    ze_device_handle_t device = nullptr;
    for (auto d : devs) {
        ze_device_properties_t p{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, nullptr};
        ZE_CHECK(zeDeviceGetProperties(d, &p));
        if (p.type != ZE_DEVICE_TYPE_GPU) continue;
        gpuSeen++;
        if (gpuSeen == devIdx) device = d;
    }
    if (!device) { std::fprintf(stderr, "no GPU\n"); return 1; }
    ze_context_desc_t cdesc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ze_context_handle_t context;
    ZE_CHECK(zeContextCreate(drvs[0], &cdesc, &context));

    std::ifstream f(spv_path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", spv_path); return 1; }
    std::vector<uint8_t> spv((std::istreambuf_iterator<char>(f)), {});
    ze_module_desc_t mdesc{ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr,
            ZE_MODULE_FORMAT_IL_SPIRV, spv.size(), spv.data(), nullptr, nullptr};
    ze_module_handle_t module = nullptr;
    ze_module_build_log_handle_t blog = nullptr;
    ze_result_t rc = zeModuleCreate(context, device, &mdesc, &module, &blog);
    if (blog) {
        size_t sz = 0;
        zeModuleBuildLogGetString(blog, &sz, nullptr);
        std::vector<char> text(sz + 1, 0);
        zeModuleBuildLogGetString(blog, &sz, text.data());
        if (sz > 1) std::printf("[build log]\n%s\n", text.data());
        zeModuleBuildLogDestroy(blog);
    }
    if (rc != ZE_RESULT_SUCCESS) { std::fprintf(stderr, "module fail 0x%x\n", (unsigned)rc); return 1; }
    std::printf("module OK\n");

    // force kernel specialization by creating the kernel
    ze_kernel_desc_t kdesc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, kname};
    ze_kernel_handle_t kernel = nullptr;
    ze_result_t krc = zeKernelCreate(module, &kdesc, &kernel);
    std::printf("kernel create -> 0x%x\n", (unsigned)krc);

    size_t nbsz = 0;
    ze_result_t nrc = zeModuleGetNativeBinary(module, &nbsz, nullptr);
    std::printf("native size query -> 0x%x size=%zu\n", (unsigned)nrc, nbsz);
    if (nrc != ZE_RESULT_SUCCESS || nbsz == 0) {
        std::fprintf(stderr, "no native binary available\n");
        return 1;
    }
    std::vector<uint8_t> nb(nbsz);
    ZE_CHECK(zeModuleGetNativeBinary(module, &nbsz, nb.data()));
    std::ofstream os(out_path, std::ios::binary);
    os.write((const char*)nb.data(), (std::streamsize)nbsz);
    os.close();
    std::printf("native binary: %zu bytes -> %s\n", nbsz, out_path);
    return 0;
}
