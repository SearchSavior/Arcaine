// Diagnostic host: grid {1,1,512} local {1,1,16} -- dump actual group mapping.
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define ZE_CHECK(x) do { ze_result_t r_=(x); if(r_!=ZE_RESULT_SUCCESS){std::fprintf(stderr,"ZE err 0x%x at %s:%d\n",(unsigned)r_,__FILE__,__LINE__);std::exit(1);} } while(0)

int main(int argc, char** argv) {
    const char* spv_path = argc > 1 ? argv[1] : "ngen_lab/ovino/diag.spv";
    std::fprintf(stderr, "tick: start\n");
    std::fflush(stderr);
    std::fprintf(stderr, "tick zeInit\n");
    ZE_CHECK(zeInit(0));
    uint32_t ndrv = 0;
    std::fprintf(stderr, "tick drvget1\n");
    ZE_CHECK(zeDriverGet(&ndrv, nullptr));
    std::vector<ze_driver_handle_t> drvs(ndrv);
    ZE_CHECK(zeDriverGet(&ndrv, drvs.data()));
    uint32_t ndev = 0;
    std::fprintf(stderr, "tick devget1\n");
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
    std::fprintf(stderr, "tick ctxcreate\n");
    ZE_CHECK(zeContextCreate(drvs[0], &cdesc, &context));
    ze_command_queue_desc_t qdesc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
            ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t list;
    std::fprintf(stderr, "tick cqcreate\n");
    ZE_CHECK(zeCommandListCreateImmediate(context, device, &qdesc, &list));

    std::ifstream f(spv_path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", spv_path); return 1; }
    std::vector<uint8_t> spv((std::istreambuf_iterator<char>(f)), {});
    ze_module_desc_t mdesc{ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr,
            ZE_MODULE_FORMAT_IL_SPIRV, spv.size(), spv.data(), nullptr, nullptr};
    ze_module_handle_t module = nullptr;
    ze_module_build_log_handle_t blog = nullptr;
    std::fprintf(stderr, "tick modulecreate\n");
    ze_result_t rc = zeModuleCreate(context, device, &mdesc, &module, &blog);
    if (rc != ZE_RESULT_SUCCESS) { std::fprintf(stderr, "module fail 0x%x\n", (unsigned)rc); return 1; }
    if (blog) {
        size_t sz = 0;
        zeModuleBuildLogGetString(blog, &sz, nullptr);
        std::vector<char> text(sz + 1, 0);
        zeModuleBuildLogGetString(blog, &sz, text.data());
        if (sz > 1) std::printf("[diag build log]\n%s\n", text.data());
        zeModuleBuildLogDestroy(blog);
    }
    ze_kernel_desc_t kdesc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "diag"};
    ze_kernel_handle_t kernel = nullptr;
    std::fprintf(stderr, "tick kernelcreate\n");
    if (zeKernelCreate(module, &kdesc, &kernel) != ZE_RESULT_SUCCESS) {
        std::fprintf(stderr, "kernel create fail\n"); return 1;
    }
    // Try group size (1,1,16)
    ze_result_t gs = zeKernelSetGroupSize(kernel, 1, 1, 16);
    std::printf("zeKernelSetGroupSize(1,1,16) -> 0x%x\n", (unsigned)gs);
    std::fflush(stdout);

    void *out = nullptr, *sizes = nullptr;
    ze_device_mem_alloc_desc_t dd{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    std::fprintf(stderr, "tick alloc\n");
    ZE_CHECK(zeMemAllocDevice(context, &dd, 64 * 3 * 4, 64, device, &out));
    ZE_CHECK(zeMemAllocDevice(context, &dd, 4 * 4, 64, device, &sizes));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &out));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &sizes));

    // grid z = 512, i.e. 32 groups of 16 -- same as probe2 ({1,32,512} shape)
    // grid z = 512, i.e. 32 groups of 16 -- same as probe2 ({1,32,512} shape)
    ze_group_count_t gc{1, 1, 512};
    std::fprintf(stderr, "tick launch\n");
    ze_result_t lrc = zeCommandListAppendLaunchKernel(list, kernel, &gc, nullptr, 0, nullptr);
    std::printf("launch -> 0x%x\n", (unsigned)lrc);
    std::fflush(stdout);
    if (lrc != ZE_RESULT_SUCCESS) return 1;
    ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));
    std::vector<uint32_t> outh(64 * 3), sizesh(4);
    ZE_CHECK(zeCommandListAppendMemoryCopy(list, outh.data(), out, 64 * 3 * 4, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListAppendMemoryCopy(list, sizesh.data(), sizes, 4 * 4, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));
    std::printf("first 12 groups (g2, gl2, sg_size):\n");
    for (int g = 0; g < 12; ++g)
        std::printf("  g2=%3u gl2=%3u sg=%u\n", outh[g*3], outh[g*3+1], outh[g*3+2]);
    std::printf("  ... last group:\n");
    for (int g = 30; g < 33; ++g)
        std::printf("  g2=%3u gl2=%3u sg=%u\n", outh[g*3], outh[g*3+1], outh[g*3+2]);
    return 0;
}
