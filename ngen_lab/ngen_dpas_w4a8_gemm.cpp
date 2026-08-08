// Full-GEMM W4A8 prototype harness (s8 activations x s4 weights -> fp32 C
// with per-32-K-tile fp rescale), driving ngen_dpas_w4a8_gemm.hpp.
//
//   C[M,N] (f32) = sum_t as32[m,t] * ws[t,n] * (Aq[M,K] s8 @ W[N,K] s4 tile t)
//
// Shapes are runtime-parametric (lab perf sweeps):
//   NGEN_LAB_W4A8_M / _N / _K   GEMM shape (M%16==0, K%32==0, N%(16*T)==0)
//   NGEN_LAB_W4A8_T             N-tiles per thread, 1..2 (default 1)
//   NGEN_LAB_W4A8_REPS          timed launches (default 20; best of reps+2)
//   NGEN_LAB_W4A8_CHECK         1 = force CPU-reference check,
//                               0 = force off, unset = auto (M*N*K <= 1e9)
//
// Host-side layouts (see the .hpp for the kernel's view):
//   A:    s8 (M,K) row-major on the host; swizzled by packA() into DPAS
//         src1 block order [m_tile][k_tile][j][lane m] (512B per block).
//   W:    s4 natural (N,K) row-major on the host; packed by packB() into the
//         DPAS src2 stream [n_tile][k_tile][h][r][j] (64 dwords per (nt,kt)).
//   ws:   f32 (K/32, N) row-major.
//   as32: f32 (M, K/32) row-major, with 16 extra pad floats after the last
//         row — per-32-tile activation scales expanded on the host from
//         per-128-group scales (as32[m,t] = act_s[m, t/4]). The kernel
//         preloads one tile ahead, so the last row's preload-ahead reads the
//         pad (in-bounds, unused).
//   C:    f32 (M,N) row-major.

#include <level_zero/ze_api.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "ngen.hpp"
#include "ngen_level_zero.hpp"

#include "ngen_dpas_w4a8_gemm.hpp"

using namespace ngen_lab::w4a8gemm;

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

// ---- A packer -------------------------------------------------------------
// A natural s8 (M,K) row-major -> DPAS src1 block order [m_tile][k_tile]
// [j][lane m]: dword (j,m) = {A[mt*16+m][kt*32+4j .. +3]}. 512B per block.
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

// ---- B packer -------------------------------------------------------------
// W natural s4 (N,K) row-major -> DPAS src2 stream [n_tile][k_tile][h][r][j]:
//   dword (nt,kt,h,r,j) = 8 K-nibbles {W[n][kt*32 + 8j + e], e=0..7},
//   column n = nt*16 + 8h + r. 64 dwords (256B) per (nt,kt).
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

// ---- CPU reference (fp64 over the quantized operands) ---------------------
static void cpuReference(const std::vector<int8_t> &A,
        const std::vector<int8_t> &W, const std::vector<float> &as32,
        const std::vector<float> &ws, int M, int N, int K,
        std::vector<double> &ref) {
    const int Kt = K / 32;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int t = 0; t < Kt; t++) {
                int32_t s = 0;
                for (int k = 0; k < 32; k++)
                    s += int32_t(A[(size_t)m * K + t * 32 + k])
                            * int32_t(W[(size_t)n * K + t * 32 + k]);
                acc += double(as32[(size_t)m * Kt + t])
                        * double(ws[(size_t)t * N + n]) * double(s);
            }
            ref[(size_t)m * N + n] = acc;
        }
}

// ---- Level Zero (same idiom as the s4 tile harness) -----------------------
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
        if (computeOrd == UINT32_MAX)
            throw std::runtime_error("no compute queue group");
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

struct Kernel {
    ze_module_handle_t module = nullptr;
    ze_kernel_handle_t kernel = nullptr;
};

static Kernel buildKernel(const L0 &l0, const ngen::Product &product) {
    ngen::InterfaceHandler iface(ngen::HW::Xe2);
    iface.externalName("ngen_lab_w4a8_gemm");
    iface.requireGRF(128);
    iface.requireSIMD(16);
    iface.requireDPAS();
    iface.requireWorkgroup(16, kWgY, kWgZ);
    // Requiring local IDs perturbs the thread-dispatch payload and costs
    // ~2.5x perf by itself -- only request them when WG cooperation is on.
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
            gen.getGroupID(kSwz ? 1 : 0), gen.getGroupID(kSwz ? 0 : 1),
            needLid ? gen.getLocalID(1) : ngen::GRF(124),
            needLid ? gen.getLocalID(2) : ngen::GRF(125));
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
    const int M = int(envU32("NGEN_LAB_W4A8_M", 1024));
    const int N = int(envU32("NGEN_LAB_W4A8_N", 16384));
    const int K = int(envU32("NGEN_LAB_W4A8_K", 5120));
    kTiles = int(envU32("NGEN_LAB_W4A8_T", 1));
    kMode = int(envU32("NGEN_LAB_W4A8_MODE", 0));
    kWgY = int(envU32("NGEN_LAB_W4A8_WGY", 1));
    kWgZ = int(envU32("NGEN_LAB_W4A8_WGZ", 1));
    kBarrier = envU32("NGEN_LAB_W4A8_BAR", 0) != 0;
    kSwz = envU32("NGEN_LAB_W4A8_SWZ", 0) != 0;
    kP2 = envU32("NGEN_LAB_W4A8_P2", 0) != 0;
    const int reps = int(envU32("NGEN_LAB_W4A8_REPS", 20));
    const char *checkEnv = std::getenv("NGEN_LAB_W4A8_CHECK");
    // NGEN_LAB_W4A8_CACHE: 0=default, 1=L1UC_L3C, 2=L1C_L3C, 3=L1S_L3C.
    switch (envU32("NGEN_LAB_W4A8_CACHE", 0)) {
    case 1: kCacheAB = ngen::CacheSettingsLSC::L1UC_L3C; break;
    case 2: kCacheAB = ngen::CacheSettingsLSC::L1C_L3C; break;
    case 3: kCacheAB = ngen::CacheSettingsLSC::L1S_L3C; break;
    default: break;
    }

    if (M % (16 * kWgY) || K % 32 || N % (16 * kTiles * kWgZ)
            || kTiles < 1 || kTiles > 2) {
        std::fprintf(stderr,
                "need M%%(16*WGY)==0, K%%32==0, N%%(16*T*WGZ)==0, T in [1,2]\n");
        return 1;
    }
    if (kP2 && (kTiles != 1 || (kMode != 0 && kMode != 4 && kMode != 5))) {
        std::fprintf(stderr,
                "P2 requires T=1 and MODE=0 (4: ablation, 5: mad epilogue)\n");
        return 1;
    }
    const int Kt = K / 32;               // 32-deep K tiles
    const int G128 = (K + 127) / 128;    // per-128 activation scale groups
    kN = N; kK = K; kKt = Kt;
    kUnroll = int(envU32("NGEN_LAB_W4A8_U", 4));

    // ws ping-pong requires even kUnroll (or 1 = serial, no rotation). P2
    // needs U%4==0 so the g128 act_s boundary stays inside the unroll body.
    if (kP2) {
        while (kUnroll > 4 && (Kt % kUnroll || (kUnroll & 3))) kUnroll--;
        if (Kt % kUnroll || (kUnroll & 3)) {
            std::fprintf(stderr,
                    "P2: no kUnroll (multiple of 4) divides Kt=%d\n", Kt);
            return 1;
        }
    } else {
        while (kUnroll > 1 && (Kt % kUnroll || (kUnroll & 1))) kUnroll--;
    }
    kPrefetch = int(envU32("NGEN_LAB_W4A8_PF", 0));
    // Mode 4 is an ablation (epilogue sources garbage); mode 5 is exact.
    const bool doCheck = (kMode == 0 || kMode == 5)
            && (checkEnv ? std::atoi(checkEnv) != 0
                         : (double)M * N * K <= 1e9);

    // ---- host data ---------------------------------------------------------
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> distA(-128, 127); // activations s8
    std::uniform_int_distribution<int> distW(-8, 7);     // weights s4
    std::uniform_real_distribution<float> distAS(0.005f, 0.02f);
    std::uniform_real_distribution<float> distWS(5e-4f, 2e-3f);

    std::vector<int8_t> hA((size_t)M * K);
    std::vector<int8_t> hW((size_t)N * K);
    for (auto &v : hA) v = int8_t(distA(rng));
    for (auto &v : hW) v = int8_t(distW(rng));

    // Per-128-group activation scales, expanded per 32-tile (row stride Kt,
    // 64-float tail pad for the kernel's preload-ahead).
    std::vector<float> hAS((size_t)M * Kt + 64, 0.0f);
    {
        std::vector<float> g128((size_t)M * G128);
        for (auto &v : g128) v = envU32("NGEN_LAB_W4A8_ASCONST", 0) ? 0.01f : distAS(rng);
        for (int m = 0; m < M; m++)
            for (int t = 0; t < Kt; t++)
                hAS[(size_t)m * Kt + t] = g128[(size_t)m * G128 + t / 4];
    }
    std::vector<float> hWS((size_t)Kt * N);
    for (auto &v : hWS) v = distWS(rng);

    std::vector<uint32_t> hBpacked((size_t)N * K / 8); // 8 s4 per dword
    packB(hW.data(), hBpacked.data(), N, K);

    std::vector<uint8_t> hApacked((size_t)M * K);
    packA(hA.data(), hApacked.data(), M, K);

    L0 l0;
    // Tail pad for the rotated pipeline's never-consumed u+1 loads past the
    // last unroll group (A: +512B, B: +256B, ws: one WG row, as: +kUnroll).
    constexpr size_t kPad = 16384; // covers prefetch-ahead over-reads too
    void *dA = l0.alloc((size_t)M * K + kPad);
    void *dB = l0.alloc((size_t)N * K / 2 + kPad);
    void *dWS = l0.alloc((size_t)Kt * N * 4 + kPad);
    void *dAS = l0.alloc(((size_t)M * Kt + 64) * 4);
    void *dC = l0.alloc((size_t)M * N * 4);
    std::memcpy(dA, hApacked.data(), (size_t)M * K);
    std::memcpy(dB, hBpacked.data(), (size_t)N * K / 2);
    std::memcpy(dWS, hWS.data(), (size_t)Kt * N * 4);
    std::memcpy(dAS, hAS.data(), hAS.size() * 4);
    std::memset(dC, 0, (size_t)M * N * 4);

    auto product =
            ngen::LevelZeroCodeGenerator<ngen::HW::Xe2>::detectHWInfo(
                    l0.context, l0.device);
    Kernel kern = buildKernel(l0, product);

    ZE_CHECK(zeKernelSetArgumentValue(kern.kernel, 0, sizeof(void *), &dA));
    ZE_CHECK(zeKernelSetArgumentValue(kern.kernel, 1, sizeof(void *), &dB));
    ZE_CHECK(zeKernelSetArgumentValue(kern.kernel, 2, sizeof(void *), &dWS));
    ZE_CHECK(zeKernelSetArgumentValue(kern.kernel, 3, sizeof(void *), &dAS));
    ZE_CHECK(zeKernelSetArgumentValue(kern.kernel, 4, sizeof(void *), &dC));

    // SWZ=1 transposes the launch order (X = m-tiles, Y = n-groups) to probe
    // L2 slice-camping / broadcast-hotspot sensitivity of the load path.
    const bool swz = kSwz;
    ze_group_count_t gc{uint32_t(N / 16 / kTiles / kWgZ),
            uint32_t(M / 16 / kWgY), 1};
    if (swz) std::swap(gc.groupCountX, gc.groupCountY);
    ZE_CHECK(zeCommandListAppendLaunchKernel(
            l0.list, kern.kernel, &gc, nullptr, 0, nullptr));
    l0.sync();

    // ---- correctness ---------------------------------------------------------
    if (doCheck) {
        std::vector<double> ref((size_t)M * N);
        cpuReference(hA, hW, hAS, hWS, M, N, K, ref);
        std::vector<float> got((size_t)M * N);
        std::memcpy(got.data(), dC, got.size() * 4);
        double maxAbs = 0.0, maxRel = 0.0, refAbs = 0.0;
        for (size_t i = 0; i < got.size(); i++) {
            double d = std::fabs(got[i] - ref[i]);
            maxAbs = std::max(maxAbs, d);
            refAbs = std::max(refAbs, std::fabs(ref[i]));
            maxRel = std::max(maxRel, d / (std::fabs(ref[i]) + 1e-6 * refAbs));
        }
        // Global-relative criterion: near-zero refs (catastrophic cancellation
        // across K tiles) make pointwise relative error meaningless.
        bool ok = maxAbs <= 1e-4 * refAbs;
        std::printf("[correctness] w4a8 gemm M=%d N=%d K=%d T=%d  "
                    "max_abs=%.4g (ref_abs=%.4g, glob_rel=%.4g) -> %s\n",
                M, N, K, kTiles, maxAbs, refAbs, maxAbs / refAbs,
                ok ? "PASS" : "FAIL");
        if (!ok) {
            int shown = 0;
            for (size_t i = 0; i < got.size() && shown < 8; i++) {
                double d = std::fabs(got[i] - ref[i]);
                if (d > 1e-4 * refAbs) {
                    std::printf("  mismatch m=%zu n=%zu got=%.6g ref=%.6g\n",
                            i / N, i % N, got[i], ref[i]);
                    shown++;
                }
            }
            return 1;
        }
    } else {
        std::printf("[correctness] skipped (NGEN_LAB_W4A8_CHECK=0 or shape "
                    "too large)\n");
    }

    // ---- performance ---------------------------------------------------------
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
            + double(Kt) * N * 4 + double(M) * Kt * 4
            + double(M) * N * 4;
    std::printf("[perf] w4a8 gemm M=%d N=%d K=%d T=%d U=%d mode=%d p2=%d  %8.3f ms"
                "  %7.2f TFLOP/s  %7.1f GB/s (min traffic)\n",
            M, N, K, kTiles, kUnroll, kMode, int(kP2), best * 1e3,
            flops / best / 1e12, bytes / best / 1e9);
    return 0;
}
