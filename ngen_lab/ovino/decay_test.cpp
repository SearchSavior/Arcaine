// Decay test host: grid {1,32,512} local (1,1,16).
// state' = state * exp(g[h]) with g = bf16(log(2)) for every head.
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>

#define ZE_CHECK(x) do { ze_result_t r_=(x); if(r_!=ZE_RESULT_SUCCESS){std::fprintf(stderr,"ZE err 0x%x at %s:%d\n",(unsigned)r_,__FILE__,__LINE__);std::exit(1);} } while(0)

int main(int argc, char** argv) {
    const char* spv_path = argc > 1 ? argv[1] : "ngen_lab/ovino/decay_test.spv";
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
    ze_kernel_desc_t kdesc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "decay_test"};
    ze_kernel_handle_t kernel = nullptr;
    if (zeKernelCreate(module, &kdesc, &kernel) != ZE_RESULT_SUCCESS) {
        std::fprintf(stderr, "kernel create fail\n"); return 1;
    }
    ZE_CHECK(zeKernelSetGroupSize(kernel, 1, 1, 16));

    constexpr int nv = 32, dk = 128, dv = 128;
    constexpr size_t state_elems = (size_t)nv * dk * dv;
    void *d_s = nullptr, *d_g = nullptr;
    ze_device_mem_alloc_desc_t dd{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    ZE_CHECK(zeMemAllocDevice(context, &dd, state_elems * 4, 64, device, &d_s));
    ZE_CHECK(zeMemAllocDevice(context, &dd, nv * 2, 64, device, &d_g));

    std::vector<float> s0(state_elems);
    for (size_t i = 0; i < state_elems; ++i) s0[i] = (float)(i % 97) / 97.0f - 0.5f;
    auto f2b = [](float f) {
        uint32_t u; std::memcpy(&u, &f, 4);
        uint32_t bias = ((u >> 16) & 1) + 0x7FFFu;
        return (uint16_t)((u + bias) >> 16);
    };
    std::vector<uint16_t> gh(nv);
    for (int h = 0; h < nv; ++h) gh[h] = f2b(std::log(2.0f));
    auto bf2f = [](uint16_t u) { uint32_t x = (uint32_t)u << 16; float f; std::memcpy(&f, &x, 4); return f; };
    const float eg_exact = std::exp((double)bf2f(gh[0]));
    ZE_CHECK(zeCommandListAppendMemoryCopy(list, d_s, s0.data(), state_elems * 4, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListAppendMemoryCopy(list, d_g, gh.data(), nv * 2, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));

    ZE_CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &d_s));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &d_g));
    ze_group_count_t gc{1, nv, 32 * 16};
    ze_result_t lrc = zeCommandListAppendLaunchKernel(list, kernel, &gc, nullptr, 0, nullptr);
    std::printf("launch -> 0x%x\n", (unsigned)lrc); std::fflush(stdout);
    if (lrc != ZE_RESULT_SUCCESS) return 1;
    ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));

    std::vector<float> s1(state_elems);
    ZE_CHECK(zeCommandListAppendMemoryCopy(list, s1.data(), d_s, state_elems * 4, nullptr, 0, nullptr));
    ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));

    size_t mism = 0; double maxerr = 0; size_t first = state_elems;
    for (size_t i = 0; i < state_elems; ++i) {
        float expect = s0[i] * eg_exact;
        if (std::fabs(s1[i] - expect) > 1e-3f * std::max(1.0f, std::fabs(expect))) {
            if (first == state_elems) first = i;
            mism++;
            maxerr = std::max(maxerr, (double)std::fabs(s1[i] - expect));
        }
    }
    std::printf("decay test: mismatches=%zu/%zu maxerr=%.6f\n", mism, state_elems, maxerr);
    // Find worst elements and report (h,k,v) + ratio to localize mapping errors.
    {
        size_t worst[5] = {0,0,0,0,0};
        double worstv[5] = {0,0,0,0,0};
        for (size_t i = 0; i < state_elems; ++i) {
            double e = std::fabs(s1[i] - s0[i]*eg_exact);
            for (int j = 0; j < 5; ++j)
                if (e > worstv[j]) { for (int k2 = 4; k2 > j; --k2) { worst[k2]=worst[k2-1]; worstv[k2]=worstv[k2-1]; }
                                      worst[j]=i; worstv[j]=e; break; }
        }
        for (int j = 0; j < 5; ++j)
            std::printf("  worst[%d] idx %zu: in=%.4f out=%.4f ratio=%.4f (h=%zu k=%zu v=%zu)\n",
                        j, worst[j], s0[worst[j]], s1[worst[j]],
                        s0[worst[j]] != 0 ? s1[worst[j]]/s0[worst[j]] : 0.0f,
                        worst[j] / (dk*dv), (worst[j]/dv) % dk, worst[j] % dv);
    }
    if (first < state_elems)
        std::printf("  first mismatch at idx %zu: in=%.6f out=%.6f expect=%.6f (h=%zu k=%zu v=%zu)\n",
                    first, s0[first], s1[first], s0[first]*2.0f,
                    first / (dk * dv), (first / dv) % dk, first % dv);
    return mism == 0 ? 0 : 1;
}
