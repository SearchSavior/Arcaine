// M-major W4A8 harness (oneDNN-gemmstone-style rewrite), driving
// ngen_dpas_w4a8_gemm_m.hpp. Same math/layouts as ngen_dpas_w4a8_gemm.cpp
// (per-token act_s, bf16 out, D16U32 store, bf16-ulp check, 5 shape presets),
// with the M-major tile + software-pipeline knobs:
//   NGEN_LAB_W4A8M_SHAPE      1..5 production (K,N) presets
//   NGEN_LAB_W4A8M_M          GEMM rows (default 512)
//   NGEN_LAB_W4A8M_MT         m-tiles per thread, 1..4 (default 2; 16 rows ea)
//   NGEN_LAB_W4A8M_PIPE       software pipeline depth D (0 = serial, default 0)
//   NGEN_LAB_W4A8M_WGY/_WGZ   WG cooperation threads along M / N (default 1)
//   NGEN_LAB_W4A8M_MODE       0 full, 1 no-epilogue, 2 dpas-only, 3 loads-only
//   NGEN_LAB_W4A8M_REPS/_CHECK  as the N-major harness
#include <level_zero/ze_api.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ngen.hpp"
#include "ngen_level_zero.hpp"

#include "ngen_dpas_w4a8_gemm_m.hpp"

using namespace ngen_lab::w4a8m;

struct LabGen : ngen::LevelZeroCodeGenerator<ngen::HW::Xe2> {
    using ngen::LevelZeroCodeGenerator<ngen::HW::Xe2>::LevelZeroCodeGenerator;
    using ngen::BinaryCodeGenerator<ngen::HW::Xe2>::setDefaultAutoSWSB;
    using ngen::BinaryCodeGenerator<ngen::HW::Xe2>::setDefaultNoMask;
};

#define ZE_CHECK(x)                                                        \
    do {                                                                   \
        ze_result_t r_ = (x);                                              \
        if (r_ != ZE_RESULT_SUCCESS) {                                     \
            std::fprintf(stderr, "ZE error 0x%x at %s:%d\n", (unsigned)r_, \
                    __FILE__, __LINE__);                                   \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

static void packA(const int8_t *a, uint8_t *out, int M, int K) {
    const int mtiles = M / 16, ktiles = K / 32;
    for (int mt = 0; mt < mtiles; mt++)
        for (int kt = 0; kt < ktiles; kt++)
            for (int j = 0; j < 8; j++)
                for (int m = 0; m < 16; m++)
                    for (int e = 0; e < 4; e++)
                        out[(((size_t)mt * ktiles + kt) * 8 + j) * 64 + m * 4
                                + e] = uint8_t(a[(size_t)(mt * 16 + m) * K
                                        + kt * 32 + 4 * j + e]);
}

static void packB(const int8_t *w, uint32_t *out, int N, int K) {
    const int ktiles = K / 32, ntiles = N / 16;
    for (int nt = 0; nt < ntiles; nt++)
        for (int kt = 0; kt < ktiles; kt++)
            for (int h = 0; h < 2; h++)
                for (int r = 0; r < 8; r++) {
                    const int n = nt * 16 + 8 * h + r;
                    for (int j = 0; j < 4; j++) {
                        uint32_t d = 0;
                        for (int e = 0; e < 8; e++)
                            d |= (uint32_t)(uint8_t)(
                                         w[(size_t)n * K + kt * 32 + 8 * j + e]
                                         & 0xF)
                                    << (4 * e);
                        out[((size_t)nt * ktiles + kt) * 64 + h * 32 + r * 4
                                + j] = d;
                    }
                }
}

static double bf16ToDouble(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return double(f);
}

static void cpuReference(const std::vector<int8_t> &A,
        const std::vector<int8_t> &W, const std::vector<float> &as,
        const std::vector<float> &ws, int M, int N, int K,
        std::vector<uint16_t> &refBf16) {
    const int Kt = K / 32;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int t = 0; t < Kt; t++) {
                int32_t s = 0;
                for (int k = 0; k < 32; k++)
                    s += int32_t(A[(size_t)m * K + t * 32 + k])
                            * int32_t(W[(size_t)n * K + t * 32 + k]);
                acc += double(ws[(size_t)t * N + n]) * double(s);
            }
            acc *= double(as[m]);
            float f = (float)acc;
            uint32_t bits;
            std::memcpy(&bits, &f, 4);
            uint32_t rounded = (bits + 0x7FFFu + ((bits >> 16) & 1)) & 0xFFFF0000u;
            refBf16[(size_t)m * N + n] = (uint16_t)(rounded >> 16);
        }
}

struct Kernel {
    ze_module_handle_t module = nullptr;
    ze_kernel_handle_t kernel = nullptr;
};

struct L0 {
    ze_driver_handle_t driver;
    ze_device_handle_t device;
    ze_context_handle_t context;
    ze_command_list_handle_t list;
    L0() {
        ZE_CHECK(zeInit(0));
        uint32_t ndrv = 0;
        ZE_CHECK(zeDriverGet(&ndrv, nullptr));
        if (ndrv == 0) throw std::runtime_error("no L0 driver");
        std::vector<ze_driver_handle_t> drvs(ndrv);
        ZE_CHECK(zeDriverGet(&ndrv, drvs.data()));
        driver = drvs[0];
        uint32_t ndev = 0;
        ZE_CHECK(zeDeviceGet(driver, &ndev, nullptr));
        std::vector<ze_device_handle_t> devs(ndev);
        ZE_CHECK(zeDeviceGet(driver, &ndev, devs.data()));
        device = nullptr;
        int devIdx = std::atoi(
                getenv("NGEN_LAB_DEVIDX") ? getenv("NGEN_LAB_DEVIDX") : "0");
        int gpuSeen = -1;
        for (auto d : devs) {
            ze_device_properties_t p{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,
                    nullptr};
            ZE_CHECK(zeDeviceGetProperties(d, &p));
            if (p.type != ZE_DEVICE_TYPE_GPU) continue;
            gpuSeen++;
            std::printf("gpu %d: %s%s\n", gpuSeen, p.name,
                    gpuSeen == devIdx ? "  [selected]" : "");
            if (gpuSeen == devIdx) device = d;
        }
        if (!device) throw std::runtime_error("no GPU at NGEN_LAB_DEVIDX");
        uint32_t ngroups = 0;
        ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(
                device, &ngroups, nullptr));
        std::vector<ze_command_queue_group_properties_t> groups(ngroups,
                {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr});
        ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(
                device, &ngroups, groups.data()));
        uint32_t computeOrd = UINT32_MAX;
        for (uint32_t i = 0; i < ngroups; i++)
            if (computeOrd == UINT32_MAX
                    && (groups[i].flags
                            & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE))
                computeOrd = i;
        ze_context_desc_t cdesc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
        ZE_CHECK(zeContextCreate(driver, &cdesc, &context));
        ze_command_queue_desc_t qdesc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                nullptr, computeOrd, 0, 0, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
                ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
        ZE_CHECK(zeCommandListCreateImmediate(context, device, &qdesc, &list));
    }
    void *alloc(size_t bytes) {
        void *p = nullptr;
        ze_device_mem_alloc_desc_t ddesc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
                nullptr, 0, 0};
        ze_host_mem_alloc_desc_t hdesc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                nullptr, 0};
        ZE_CHECK(zeMemAllocShared(
                context, &ddesc, &hdesc, bytes, 64, device, &p));
        return p;
    }
    void sync() {
        ze_result_t r = zeCommandListHostSynchronize(list, 10000000000ull);
        if (r != ZE_RESULT_SUCCESS) {
            ze_result_t st = zeDeviceGetStatus(device);
            std::fprintf(stderr, "sync failed: 0x%x, zeDeviceGetStatus=0x%x\n",
                    (unsigned)r, (unsigned)st);
            std::exit(1);
        }
    }
};

static Kernel buildKernel(const L0 &l0, const ngen::Product &product) {
    ngen::InterfaceHandler iface(ngen::HW::Xe2);
    iface.externalName("ngen_lab_w4a8m_gemm");
    iface.requireGRF(256);
    iface.requireSIMD(16);
    iface.requireDPAS();
    iface.requireWorkgroup(16, kWgY, kWgZ);
    const bool needLid = kWgY > 1 || kWgZ > 1;
    if (needLid) iface.requireLocalID(3);
    iface.newArgument("a", ngen::ExternalArgumentType::GlobalPtr,
            ngen::GlobalAccessType::Stateless);
    iface.newArgument("b", ngen::ExternalArgumentType::GlobalPtr,
            ngen::GlobalAccessType::Stateless);
    iface.newArgument("ws", ngen::ExternalArgumentType::GlobalPtr,
            ngen::GlobalAccessType::Stateless);
    iface.newArgument("as", ngen::ExternalArgumentType::GlobalPtr,
            ngen::GlobalAccessType::Stateless);
    iface.newArgument("c", ngen::ExternalArgumentType::GlobalPtr,
            ngen::GlobalAccessType::Stateless);
    iface.finalize();

    LabGen gen(product);
    gen.setInterface(iface);
    gen.setDefaultAutoSWSB(true);
    gen.prologue();
    gen.setDefaultNoMask(true);
    gemmKernelBody(gen, gen.getArgument("a"), gen.getArgument("b"),
            gen.getArgument("ws"), gen.getArgument("as"), gen.getArgument("c"),
            gen.getGroupID(0), gen.getGroupID(1),
            needLid ? gen.getLocalID(1) : ngen::GRF(250),
            needLid ? gen.getLocalID(2) : ngen::GRF(251));
    gen.epilogue();

    auto mk = gen.getModuleAndKernel(l0.context, l0.device);
    ZE_CHECK(zeKernelSetGroupSize(mk.second, 16, kWgY, kWgZ));
    return {mk.first, mk.second};
}

static uint32_t envU32(const char *name, uint32_t dflt) {
    const char *v = std::getenv(name);
    return v ? uint32_t(std::atoi(v)) : dflt;
}

int main() {
    static const int presetK[] = {5120, 6144, 5120, 5120, 17408};
    static const int presetN[] = {14336, 5120, 16384, 34816, 5120};
    const uint32_t shape = envU32("NGEN_LAB_W4A8M_SHAPE", 0);
    const int M = int(envU32("NGEN_LAB_W4A8M_M", 512));
    int N = int(envU32("NGEN_LAB_W4A8M_N", 0));
    int K = int(envU32("NGEN_LAB_W4A8M_K", 0));
    if (shape >= 1 && shape <= 5) {
        if (N == 0) N = presetN[shape - 1];
        if (K == 0) K = presetK[shape - 1];
    } else if (N == 0 || K == 0) {
        N = 16384; K = 5120;
    }
    kMt = int(envU32("NGEN_LAB_W4A8M_MT", 2));
    kPipeDepth = int(envU32("NGEN_LAB_W4A8M_PIPE", 0));
    kWgY = int(envU32("NGEN_LAB_W4A8M_WGY", 1));
    kWgZ = int(envU32("NGEN_LAB_W4A8M_WGZ", 1));
    kMode = int(envU32("NGEN_LAB_W4A8M_MODE", 0));
    kLdWS = envU32("NGEN_LAB_W4A8M_LDWS", 1) != 0;
    const int reps = int(envU32("NGEN_LAB_W4A8M_REPS", 20));
    const char *checkEnv = std::getenv("NGEN_LAB_W4A8M_CHECK");
    switch (envU32("NGEN_LAB_W4A8M_CACHE", 0)) {
    case 2: kCacheAB = ngen::CacheSettingsLSC::L1C_L3C; break;
    case 3: kCacheAB = ngen::CacheSettingsLSC::L1S_L3C; break;
    default: break;
    }

    if (M % (16 * kMt * kWgY) || N % (16 * kWgZ) || K % 32 || kMt < 1
            || kMt > 4 || kWgY < 1 || kWgY > 8 || kWgZ < 1 || kWgZ > 8) {
        std::fprintf(stderr, "need M%%(16*MT*WGY)==0, N%%(16*WGZ)==0, "
                "K%%32==0, MT in [1,4], WGY/WGZ in [1,8]\n");
        return 1;
    }
    // Register budget check: highest GRF = kF()+3*Mt+8 (see the .hpp).
    {
        int nb = kPipeDepth + 1;
        int hi = 26 + 20 * kMt * nb + 4 * nb + 6 * kMt;   // rough bound
        if (hi > 256) {
            std::fprintf(stderr, "PIPE=%d MT=%d exceeds the 256-GRF budget\n",
                    kPipeDepth, kMt);
            return 1;
        }
        (void)nb; (void)hi;
    }
    const int Kt = K / 32;
    kN = N; kK = K; kKt = Kt;
    const bool doCheck = (kMode == 0)
            && (checkEnv ? std::atoi(checkEnv) != 0
                         : (double)M * N * K <= 5e9);

    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> distA(-128, 127);
    std::uniform_int_distribution<int> distW(-8, 7);
    std::uniform_real_distribution<float> distAS(0.005f, 0.02f);
    std::uniform_real_distribution<float> distWS(5e-4f, 2e-3f);

    std::vector<int8_t> hA((size_t)M * K);
    std::vector<int8_t> hW((size_t)N * K);
    for (auto &v : hA) v = int8_t(distA(rng));
    for (auto &v : hW) v = int8_t(distW(rng));
    std::vector<float> hAS((size_t)M);
    for (auto &v : hAS) v = distAS(rng);
    std::vector<float> hWS((size_t)Kt * N);
    for (auto &v : hWS) v = distWS(rng);

    std::vector<uint32_t> hBpacked((size_t)N * K / 8);
    packB(hW.data(), hBpacked.data(), N, K);
    std::vector<uint8_t> hApacked((size_t)M * K);
    packA(hA.data(), hApacked.data(), M, K);

    L0 l0;
    // Tail pad: the pipelined loads read `ahead` tiles past the end on the
    // final iterations (never consumed; the last loads are skipped at u+ahead
    // >= Kt, so only the prologue + in-range loads run — pad for safety).
    const size_t wsPad = (size_t)N * 4 + 1024;
    constexpr size_t abPad = 16384;
    void *dA = l0.alloc((size_t)M * K + abPad);
    void *dB = l0.alloc((size_t)N * K / 2 + abPad);
    void *dWS = l0.alloc((size_t)Kt * N * 4 + wsPad);
    void *dAS = l0.alloc((size_t)M * 4);
    void *dC = l0.alloc((size_t)M * N * 2);
    std::memcpy(dA, hApacked.data(), (size_t)M * K);
    std::memcpy(dB, hBpacked.data(), (size_t)N * K / 2);
    std::memcpy(dWS, hWS.data(), (size_t)Kt * N * 4);
    std::memcpy(dAS, hAS.data(), (size_t)M * 4);
    std::memset(dC, 0, (size_t)M * N * 2);

    auto product =
            ngen::LevelZeroCodeGenerator<ngen::HW::Xe2>::detectHWInfo(
                    l0.context, l0.device);
    Kernel kern = buildKernel(l0, product);

    ZE_CHECK(zeKernelSetArgumentValue(kern.kernel, 0, sizeof(void *), &dA));
    ZE_CHECK(zeKernelSetArgumentValue(kern.kernel, 1, sizeof(void *), &dB));
    ZE_CHECK(zeKernelSetArgumentValue(kern.kernel, 2, sizeof(void *), &dWS));
    ZE_CHECK(zeKernelSetArgumentValue(kern.kernel, 3, sizeof(void *), &dAS));
    ZE_CHECK(zeKernelSetArgumentValue(kern.kernel, 4, sizeof(void *), &dC));

    ze_group_count_t gc{uint32_t(N / 16 / kWgZ),
            uint32_t(M / 16 / kMt / kWgY), 1};
    ZE_CHECK(zeCommandListAppendLaunchKernel(
            l0.list, kern.kernel, &gc, nullptr, 0, nullptr));
    l0.sync();

    if (doCheck) {
        std::vector<uint16_t> ref((size_t)M * N);
        cpuReference(hA, hW, hAS, hWS, M, N, K, ref);
        std::vector<uint16_t> got((size_t)M * N);
        std::memcpy(got.data(), dC, got.size() * 2);
        double maxAbs = 0.0, maxRel = 0.0, refAbs = 0.0;
        for (size_t i = 0; i < got.size(); i++) {
            double gd = bf16ToDouble(got[i]);
            double rd = bf16ToDouble(ref[i]);
            double d = std::fabs(gd - rd);
            maxAbs = std::max(maxAbs, d);
            refAbs = std::max(refAbs, std::fabs(rd));
            maxRel = std::max(maxRel, d / (std::fabs(rd) + 1e-6 * refAbs));
        }
        bool ok = maxAbs <= (1.0 / 256.0) * refAbs;
        std::printf("[correctness] w4a8m M=%d N=%d K=%d MT=%d PIPE=%d  "
                    "max_abs=%.4g (ref_abs=%.4g, glob_rel=%.4g) -> %s\n",
                M, N, K, kMt, kPipeDepth, maxAbs, refAbs, maxAbs / refAbs,
                ok ? "PASS" : "FAIL");
        if (!ok) return 1;
    } else {
        std::printf("[correctness] skipped\n");
    }

    double best = 1e30;
    for (int r = 0; r < reps + 2; r++) {
        auto t0 = std::chrono::steady_clock::now();
        ZE_CHECK(zeCommandListAppendLaunchKernel(
                l0.list, kern.kernel, &gc, nullptr, 0, nullptr));
        l0.sync();
        auto t1 = std::chrono::steady_clock::now();
        if (r >= 2)
            best = std::min(best,
                    std::chrono::duration<double>(t1 - t0).count());
    }
    const double flops = 2.0 * M * N * K;
    const double bytes = double(M) * K + double(N) * K / 2
            + double(Kt) * N * 4 + double(M) * 4 + double(M) * N * 2;
    std::printf("[perf] w4a8m M=%d N=%d K=%d MT=%d PIPE=%d WGY=%d WGZ=%d "
                "mode=%d  %8.3f ms  %7.2f TFLOP/s  %7.1f GB/s\n",
            M, N, K, kMt, kPipeDepth, kWgY, kWgZ, kMode, best * 1e3,
            flops / best / 1e12, bytes / best / 1e9);
    return 0;
}
