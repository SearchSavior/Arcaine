// Probe2 host: 3D grid, scalar args, vload8, reduce_add with float4 pattern.
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define ZE_CHECK(x) do { ze_result_t r_=(x); if(r_!=ZE_RESULT_SUCCESS){std::fprintf(stderr,"ZE err 0x%x at %s:%d\n",(unsigned)r_,__FILE__,__LINE__);std::exit(1);} } while(0)

int main(int argc, char** argv) {
    const char* spv_path = argc > 1 ? argv[1] : "ngen_lab/ovino/probe2.spv";
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
    std::printf("module OK\n");
    ze_kernel_desc_t kdesc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "probe2"};
    ze_kernel_handle_t kernel = nullptr;
    if (zeKernelCreate(module, &kdesc, &kernel) != ZE_RESULT_SUCCESS) {
        std::fprintf(stderr, "kernel create fail\n"); return 1;
    }
    std::printf("kernel OK\n");
    ZE_CHECK(zeKernelSetGroupSize(kernel, 1, 1, 16));

    void *v = nullptr, *out = nullptr, *so = nullptr, *red = nullptr;
    ze_device_mem_alloc_desc_t dd{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    ZE_CHECK(zeMemAllocDevice(context, &dd, 32 * 128 * 2, 64, device, &v));
    ZE_CHECK(zeMemAllocDevice(context, &dd, 32 * 512 * 4, 64, device, &out));
    ZE_CHECK(zeMemAllocDevice(context, &dd, 32 * 32 * 4, 64, device, &so));
    ZE_CHECK(zeMemAllocDevice(context, &dd, 32 * 32 * 4, 64, device, &red));

    std::vector<uint16_t> vh(32 * 128);
    for (int i = 0; i < 32 * 128; ++i) vh[i] = (uint16_t)(i % 1000);
    ZE_CHECK(zeCommandListAppendMemoryCopy(list, v, vh.data(), 32 * 128 * 2, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));

    ZE_CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &v));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &out));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 2, sizeof(void*), &so));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 3, sizeof(void*), &red));
    int32_t seq_len = 7, qt = 4096;
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 4, sizeof(seq_len), &seq_len));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 5, sizeof(qt), &qt));

    // grid {1, 32, 32*16}
    ze_group_count_t gc{1, 32, 512};
    ZE_CHECK(zeCommandListAppendLaunchKernel(list, kernel, &gc, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));

    std::vector<float> outh(32 * 512), soh(32 * 32), redh(32 * 32);
    ZE_CHECK(zeCommandListAppendMemoryCopy(list, outh.data(), out, 32 * 512 * 4, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListAppendMemoryCopy(list, soh.data(), so, 32 * 32 * 4, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListAppendMemoryCopy(list, redh.data(), red, 32 * 32 * 4, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));

    bool ok = true;
    // check a few scattered out entries: out[g1*512 + g2 + lid] = g1*100 + g2 + lid
    for (int g1 : {0, 7, 31})
        for (int g2 : {0, 255, 511})
            if (outh[g1 * 512 + g2] != (float)(g1 * 100 + g2 + 0)) ok = false;
    for (int g1 : {3, 17})
        for (int g2 : {0, 16, 496})
            if (soh[g1 * 32 + g2 / 16] != (float)(seq_len + qt)) ok = false;
    // red[g1*32 + g2/16] = sum over lanes of vload8(v + g1*128 + lid*8)
    // lane lid reads v[g1*128 + lid*8 + 0..8) = (g1*128 + lid*8 + 0..7)
    // sum = 8*(g1*128) + 8*8*sum_{l=0..15} l + 28
    for (int g1 : {0, 5}) {
        float expect = 0;
        for (int l = 0; l < 16; ++l)
            for (int j = 0; j < 8; ++j)
                expect += (float)((g1 * 128 + l * 8 + j) % 1000);
        if (redh[g1 * 32] != expect) ok = false;
    }
    std::printf("out[0]=%.0f out[7*512+255]=%.0f out[31*512+511]=%.0f\n",
                outh[0], outh[7 * 512 + 255], outh[31 * 512 + 511]);
    std::printf("scalar_out[0]=%.0f (expect %d) scalar_out[17*32+1]=%.0f\n",
                soh[0], seq_len + qt, soh[17 * 32 + 1]);
    std::printf("red[0]=%.0f red[5*32]=%.0f\n", redh[0], redh[5 * 32]);
    std::printf("probe2: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
