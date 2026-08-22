// red_test host: grid {512,32,1} local (16,1,1).
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define ZE_CHECK(x) do { ze_result_t r_=(x); if(r_!=ZE_RESULT_SUCCESS){std::fprintf(stderr,"ZE err 0x%x at %s:%d\n",(unsigned)r_,__FILE__,__LINE__);std::exit(1);} } while(0)

int main(int argc, char** argv) {
    const char* spv_path = argc > 1 ? argv[1] : "ngen_lab/ovino/red_test.spv";
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
    ze_command_queue_desc_t qdesc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
            ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t list;
    ZE_CHECK(zeCommandListCreateImmediate(context, device, &qdesc, &list));

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
    ze_kernel_desc_t kdesc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "red_test"};
    ze_kernel_handle_t kernel = nullptr;
    if (zeKernelCreate(module, &kdesc, &kernel) != ZE_RESULT_SUCCESS) {
        std::fprintf(stderr, "kernel create fail\n"); return 1;
    }
    ZE_CHECK(zeKernelSetGroupSize(kernel, 16, 1, 1));

    void *d_red = nullptr, *d_size = nullptr;
    ze_device_mem_alloc_desc_t dd{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    ZE_CHECK(zeMemAllocDevice(context, &dd, 32 * 32 * 4, 64, device, &d_red));
    ZE_CHECK(zeMemAllocDevice(context, &dd, 4, 64, device, &d_size));
    ZE_CHECK(zeCommandListAppendMemset(list, d_red, 0, 32 * 32 * 4, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &d_red));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &d_size));

    ze_group_count_t gc{512, 32, 1};
    ze_result_t lrc = zeCommandListAppendLaunchKernel(list, kernel, &gc, nullptr, 0, nullptr);
    std::printf("launch -> 0x%x\n", (unsigned)lrc); std::fflush(stdout);
    if (lrc != ZE_RESULT_SUCCESS) return 1;
    ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));

    std::vector<float> redh(32 * 32), sizeh(1);
    ZE_CHECK(zeCommandListAppendMemoryCopy(list, redh.data(), d_red, 32 * 32 * 4, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListAppendMemoryCopy(list, sizeh.data(), d_size, 4, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));

    bool ok = true;
    std::printf("sg_size=%d\n", (int)sizeh[0]);
    for (int h = 0; h < 32 && ok; ++h)
        for (int g = 0; g < 32; ++g) {
            float expect = (float)(256 * g + 136);
            if (redh[h * 32 + g] != expect) {
                std::printf("  MISMATCH h=%d g=%d got=%.1f expect=%.1f\n",
                            h, g, redh[h * 32 + g], expect);
                ok = false;
            }
        }
    std::printf("reduce test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
