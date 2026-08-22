// Scalar sequential GDN harness. grid {nv, vblocks}, local {1,1}.
// Mirrors the qwen35-gdn bench synthetic data + host reference exactly.
#include <level_zero/ze_api.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define ZE_CHECK(x) do { ze_result_t r_=(x); if(r_!=ZE_RESULT_SUCCESS){std::fprintf(stderr,"ZE err 0x%x at %s:%d\n",(unsigned)r_,__FILE__,__LINE__);std::exit(1);} } while(0)

static uint32_t rng_state = 42u;
static float frand(float lo, float hi) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return lo + (hi - lo) * (float)((rng_state >> 8) & 0xFFFFFF) / 16777216.0f;
}
static float bf2f(uint16_t u) { uint32_t x = (uint32_t)u << 16; float f; std::memcpy(&f, &x, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); uint32_t b = ((u >> 16) & 1) + 0x7FFFu; return (uint16_t)((u + b) >> 16); }

int main(int argc, char** argv) {
    std::string p_csv = "512,1024,2048,4096";
    std::string spv_path = "ngen_lab/ovino/gdn_seq_vec.spv";
    int iters = 5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-p" && i + 1 < argc) p_csv = argv[++i];
        else if (a == "-n" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--spv" && i + 1 < argc) spv_path = argv[++i];
    }

    ZE_CHECK(zeInit(0));
    uint32_t ndrv = 0;
    ZE_CHECK(zeDriverGet(&ndrv, nullptr));
    std::vector<ze_driver_handle_t> drvs(ndrv);
    ZE_CHECK(zeDriverGet(&ndrv, drvs.data()));
    uint32_t ndev = 0;
    ZE_CHECK(zeDeviceGet(drvs[0], &ndev, nullptr));
    std::vector<ze_device_handle_t> devs(ndev);
    ZE_CHECK(zeDeviceGet(drvs[0], &ndev, devs.data()));
    ze_device_handle_t device = nullptr;
    int gpuSeen = -1, devIdx = std::atoi(getenv("L0_MIN_DEVIDX") ? getenv("L0_MIN_DEVIDX") : "0");
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
    if (!f) { std::fprintf(stderr, "cannot open %s\n", spv_path.c_str()); return 1; }
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
    ze_kernel_desc_t kdesc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "gdn_seq_vec"};
    ze_kernel_handle_t kernel = nullptr;
    if (zeKernelCreate(module, &kdesc, &kernel) != ZE_RESULT_SUCCESS) {
        std::fprintf(stderr, "kernel create fail\n"); return 1;
    }
    ZE_CHECK(zeKernelSetGroupSize(kernel, 16, 1, 1));

    constexpr int nv = 32, dk = 128, dv = 128, vblocks = dv / 4;
    constexpr size_t state_elems = (size_t)nv * dk * dv;
    auto alloc = [&](size_t n) -> void* {
        void* p = nullptr;
        ze_device_mem_alloc_desc_t dd{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
        ZE_CHECK(zeMemAllocDevice(context, &dd, n, 64, device, &p));
        return p;
    };

    std::vector<int> tokens;
    { std::string s = p_csv + ","; size_t pos = 0;
      while ((pos = s.find(',')) != std::string::npos) {
          tokens.push_back(std::atoi(s.substr(0, pos).c_str())); s.erase(0, pos + 1); } }

    for (int T : tokens) {
        const size_t ntok = (size_t)T;
        std::vector<uint16_t> q_h(ntok * nv * dk), k_h(ntok * nv * dk),
                              v_h(ntok * nv * dv), beta_h(ntok * nv), g_h(ntok * nv);
        const float scale = 1.0f / std::sqrt((float)dk);
        for (size_t row = 0; row < ntok * nv; ++row) {
            float nq = 0, nk = 0;
            std::vector<float> tq(dk), tk(dk);
            for (int d = 0; d < dk; ++d) {
                tq[d] = frand(-1.f, 1.f); nq += tq[d] * tq[d];
                tk[d] = frand(-1.f, 1.f); nk += tk[d] * tk[d];
            }
            nq = 1.f / std::sqrt(std::max(nq, 1e-12f));
            nk = 1.f / std::sqrt(std::max(nk, 1e-12f));
            for (int d = 0; d < dk; ++d) {
                q_h[row * dk + d] = f2bf(tq[d] * nq * scale);
                k_h[row * dk + d] = f2bf(tk[d] * nk);
            }
            for (int d = 0; d < dv; ++d) v_h[row * dv + d] = f2bf(frand(-1.f, 1.f));
            beta_h[row] = f2bf(frand(0.05f, 0.95f));
            g_h[row]    = f2bf(-frand(0.0005f, 0.05f));
        }
        std::vector<float> s0(state_elems, 0.0f);
        std::vector<uint16_t> core_h(ntok * nv * dv);
        std::vector<float> s1(state_elems);
        { // host reference
            std::vector<float> Sh = s0;
            for (int h = 0; h < nv; ++h) {
                float* S = Sh.data() + (size_t)h * dk * dv;
                for (int t = 0; t < T; ++t) {
                    float eg = std::exp(bf2f(g_h[(size_t)t * nv + h]));
                    float bb = bf2f(beta_h[(size_t)t * nv + h]);
                    const uint16_t* qrow = q_h.data() + ((size_t)t * nv + h) * dk;
                    const uint16_t* krow = k_h.data() + ((size_t)t * nv + h) * dk;
                    const uint16_t* vrow = v_h.data() + ((size_t)t * nv + h) * dv;
                    for (int dv_i = 0; dv_i < dv; ++dv_i) {
                        float kv_mem = 0;
                        for (int dk_i = 0; dk_i < dk; ++dk_i)
                            kv_mem += S[(size_t)dk_i * dv + dv_i] * eg * bf2f(krow[dk_i]);
                        float delta = (bf2f(vrow[dv_i]) - kv_mem) * bb;
                        float o = 0;
                        for (int dk_i = 0; dk_i < dk; ++dk_i) {
                            float s = S[(size_t)dk_i * dv + dv_i] * eg + bf2f(krow[dk_i]) * delta;
                            S[(size_t)dk_i * dv + dv_i] = s;
                            o += s * bf2f(qrow[dk_i]);
                        }
                        core_h[((size_t)t * nv + h) * dv + dv_i] = f2bf(o);
                    }
                }
            }
            s1 = Sh;
        }

        void *d_q = alloc(q_h.size() * 2), *d_k = alloc(k_h.size() * 2),
             *d_v = alloc(v_h.size() * 2), *d_beta = alloc(beta_h.size() * 2),
             *d_g = alloc(g_h.size() * 2), *d_s = alloc(state_elems * 4),
             *d_core = alloc(core_h.size() * 2);
        ZE_CHECK(zeCommandListAppendMemoryCopy(list, d_q, q_h.data(), q_h.size() * 2, nullptr, 0, nullptr));
        ZE_CHECK(zeCommandListAppendMemoryCopy(list, d_k, k_h.data(), k_h.size() * 2, nullptr, 0, nullptr));
        ZE_CHECK(zeCommandListAppendMemoryCopy(list, d_v, v_h.data(), v_h.size() * 2, nullptr, 0, nullptr));
        ZE_CHECK(zeCommandListAppendMemoryCopy(list, d_beta, beta_h.data(), beta_h.size() * 2, nullptr, 0, nullptr));
        ZE_CHECK(zeCommandListAppendMemoryCopy(list, d_g, g_h.data(), g_h.size() * 2, nullptr, 0, nullptr));
        ZE_CHECK(zeCommandListAppendMemoryCopy(list, d_s, s0.data(), state_elems * 4, nullptr, 0, nullptr));
        ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));

        auto setp = [&](uint32_t i, void* p) { ZE_CHECK(zeKernelSetArgumentValue(kernel, i, sizeof(void*), &p)); };
        setp(0, d_q); setp(1, d_k); setp(2, d_v); setp(3, d_s); setp(4, d_beta); setp(5, d_g); setp(6, d_core);
        int32_t seq_len = T, qt = nv * dk, kt = nv * dk, vt = nv * dv;
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 7, sizeof(seq_len), &seq_len));
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 8, sizeof(qt), &qt));
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 9, sizeof(kt), &kt));
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 10, sizeof(vt), &vt));

        ze_group_count_t gc{vblocks * 16, nv, 1};
        for (int i = 0; i < 2; ++i) {
            ZE_CHECK(zeCommandListAppendLaunchKernel(list, kernel, &gc, nullptr, 0, nullptr));
            ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));
        }
        std::vector<double> samples;
        for (int i = 0; i < iters; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            ZE_CHECK(zeCommandListAppendLaunchKernel(list, kernel, &gc, nullptr, 0, nullptr));
            ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));
            auto t1 = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        double mean = 0; for (double s : samples) mean += s; mean /= samples.size();

        // correctness run: re-upload cold state so comparison is valid
        ZE_CHECK(zeCommandListAppendMemoryCopy(list, d_s, s0.data(), state_elems * 4, nullptr, 0, nullptr));
        ZE_CHECK(zeCommandListAppendLaunchKernel(list, kernel, &gc, nullptr, 0, nullptr));
        ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));
        std::vector<uint16_t> core_gpu(core_h.size());
        std::vector<float> s_gpu(state_elems);
        ZE_CHECK(zeCommandListAppendMemoryCopy(list, core_gpu.data(), d_core, core_h.size() * 2, nullptr, 0, nullptr));
        ZE_CHECK(zeCommandListAppendMemoryCopy(list, s_gpu.data(), d_s, state_elems * 4, nullptr, 0, nullptr));
        ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));

        float max_abs = 0; double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < core_h.size(); ++i) {
            float e = bf2f(core_h[i]), o = bf2f(core_gpu[i]);
            max_abs = std::max(max_abs, std::fabs(o - e));
            dot += (double)e * o; na += (double)e * e; nb += (double)o * o;
        }
        double cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0;
        float state_rel = 0;
        for (size_t i = 0; i < state_elems; ++i)
            state_rel = std::max(state_rel, std::fabs(s_gpu[i] - s1[i]) / std::max(1e-3f, std::fabs(s1[i])));
        std::printf("ovino_vec p=%d runs=%d mean_ms=%.3f tok_per_s=%.1f "
                    "core_max_abs=%.6f core_cos=%.6f state_max_rel=%.6f\n",
                    T, iters, mean, T * 1000.0 / mean, max_abs, cos, state_rel);
        if (T == 8) {
            std::printf("  [dump] core h=0..3 t=0: ref vs gpu\n");
            for (int i = 0; i < 4; ++i)
                std::printf("    h%d ref=%8.4f gpu=%8.4f\n", i, bf2f(core_h[i]), bf2f(core_gpu[i]));
            std::printf("  [dump] state h=0 k=0..3 v=0: ref vs gpu\n");
            for (int k = 0; k < 4; ++k)
                std::printf("    k%d ref=%9.5f gpu=%9.5f\n", k, s1[k*128], s_gpu[k*128]);
            int bad = 0;
            for (size_t i = 0; i < state_elems && bad < 6; ++i) {
                float err = std::fabs(s_gpu[i] - s1[i]) / std::max(1e-3f, std::fabs(s1[i]));
                if (err > 1e-3f) {
                    std::printf("  STATEBAD idx %zu: ref=%.5f gpu=%.5f (h=%zu k=%zu v=%zu)\n",
                                i, s1[i], s_gpu[i], i/(dk*dv), (i/dv)%dk, i%dv);
                    bad++;
                }
            }
        }
        zeMemFree(context, d_q); zeMemFree(context, d_k); zeMemFree(context, d_v);
        zeMemFree(context, d_beta); zeMemFree(context, d_g);
        zeMemFree(context, d_s); zeMemFree(context, d_core);
    }
    std::printf("done\n");
    return 0;
}
