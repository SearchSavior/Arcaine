// ngen_lab: standalone Level Zero executable for a fixed-shape BF16 16x16x32
// matmul on Intel Arc BMG (Xe2), with an nGEN-authored DPAS kernel, an
// nGEN-authored scalar (fp32 mad) kernel, a CPU reference, a B-weight packer,
// and correctness/performance reporting.
//
// Environment knobs (A/B testing):
//   NGEN_LAB_DEVIDX       GPU index among L0 GPU devices (default 0)
//   NGEN_LAB_KERNELS      dpas|scalar|both   (default both)
//   NGEN_LAB_GROUPS       work-groups for the perf run   (default 8192)
//   NGEN_LAB_DPAS_ITERS   accumulation iters, DPAS perf  (default 4096)
//   NGEN_LAB_SCALAR_ITERS accumulation iters, scalar perf (default 16)
//   NGEN_LAB_REPS         timed repetitions              (default 10)
//
// Correctness always runs first (64 tiles, 3 iterations) against the CPU
// reference and gates the perf run.

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

#include "ngen_dpas_8x8_matmul_bf16_16x16x32.hpp"

#define ZE_CHECK(x)                                                        \
    do {                                                                   \
        ze_result_t r_ = (x);                                              \
        if (r_ != ZE_RESULT_SUCCESS) {                                     \
            std::fprintf(stderr, "ZE error 0x%x at %s:%d\n", (unsigned)r_, \
                    __FILE__, __LINE__);                                   \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

namespace ngen_lab {

// ---- bf16 helpers ----------------------------------------------------------
static uint16_t f32ToBf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u); // round-to-nearest-even
    return uint16_t(u >> 16);
}
static float bf16ToF32(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ---- B-weight packer -------------------------------------------------------
// Packs natural row-major B[32][16] (bf16) into the DPAS src2 broadcast
// layout: 4 pieces (k-chunk c in {0,1} major, then n-half h in {0,1}),
// each 256 bytes, dword (r*8 + j) = {B[16c+2j][8h+r] (lo), B[16c+2j+1][8h+r] (hi)}.
static void packB(const uint16_t *bNat, uint32_t *bPacked) {
    for (int c = 0; c < 2; c++)
        for (int h = 0; h < 2; h++)
            for (int r = 0; r < 8; r++)
                for (int j = 0; j < 8; j++) {
                    uint32_t lo = bNat[(16 * c + 2 * j) * kN + 8 * h + r];
                    uint32_t hi = bNat[(16 * c + 2 * j + 1) * kN + 8 * h + r];
                    bPacked[((c * 2 + h) * 64) + r * 8 + j] = lo | (hi << 16);
                }
}

// ---- CPU reference ---------------------------------------------------------
static void cpuReference(const std::vector<uint16_t> &a,
        const std::vector<uint16_t> &bNat, int tiles, int iters,
        std::vector<float> &cRef) {
    for (int t = 0; t < tiles; t++)
        for (int m = 0; m < kM; m++)
            for (int n = 0; n < kN; n++) {
                float acc = 0.f;
                for (int k = 0; k < kK; k++)
                    acc += bf16ToF32(a[(size_t)t * kM * kK + m * kK + k])
                            * bf16ToF32(bNat[k * kN + n]);
                cRef[(size_t)t * kM * kN + m * kN + n] = acc * float(iters);
            }
}

struct ErrorStats {
    double maxAbs = 0, maxRel = 0, cos = 0;
};

static ErrorStats compare(const std::vector<float> &got,
        const std::vector<float> &ref) {
    ErrorStats s;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        double d = std::fabs(got[i] - ref[i]);
        s.maxAbs = std::max(s.maxAbs, d);
        s.maxRel = std::max(s.maxRel, d / std::max(1e-6, std::fabs((double)ref[i])));
        dot += double(got[i]) * ref[i];
        na += double(got[i]) * got[i];
        nb += double(ref[i]) * ref[i];
    }
    s.cos = dot / std::sqrt(na * nb + 1e-30);
    return s;
}

// ---- Level Zero setup ------------------------------------------------------
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
        int devIdx = std::atoi(getenv("NGEN_LAB_DEVIDX") ? getenv("NGEN_LAB_DEVIDX")
                                                         : "0");
        int gpuSeen = -1;
        for (auto d : devs) {
            ze_device_properties_t p{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, nullptr};
            ZE_CHECK(zeDeviceGetProperties(d, &p));
            if (p.type != ZE_DEVICE_TYPE_GPU) continue;
            gpuSeen++;
            std::printf("gpu %d: %s%s\n", gpuSeen, p.name,
                    gpuSeen == devIdx ? "  [selected]" : "");
            if (gpuSeen == devIdx) device = d;
        }
        if (!device) throw std::runtime_error("no GPU at NGEN_LAB_DEVIDX");
        // Enumerate queue groups; ordinal 0 is not guaranteed to be compute.
        uint32_t ngroups = 0;
        ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device, &ngroups, nullptr));
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
        ZE_CHECK(zeMemAllocShared(context, &ddesc, &hdesc, bytes, 64, device, &p));
        return p;
    }
    void sync() {
        ze_result_t r = zeCommandListHostSynchronize(list, 10000000000ull);
        if (r != ZE_RESULT_SUCCESS) {
            ze_result_t st = zeDeviceGetStatus(device);
            std::fprintf(stderr,
                    "sync failed: 0x%x, zeDeviceGetStatus=0x%x (%s)\n",
                    (unsigned)r, (unsigned)st,
                    st == ZE_RESULT_SUCCESS ? "available" : "lost/unavailable");
            std::exit(1);
        }
    }
};

struct Kernel {
    ze_module_handle_t module = nullptr;
    ze_kernel_handle_t kernel = nullptr;
};

// Derived generator to expose the protected default-modifier setters, matching
// oneDNN's generator pattern (setDefaultAutoSWSB *before* prologue,
// setDefaultNoMask after). Without SWSB annotations, memory ops race ahead of
// the instructions producing their addresses/data (l0_min stage 3/5 finding).
struct LabGen : ngen::LevelZeroCodeGenerator<ngen::HW::Xe2> {
    using ngen::LevelZeroCodeGenerator<ngen::HW::Xe2>::LevelZeroCodeGenerator;
    using ngen::BinaryCodeGenerator<ngen::HW::Xe2>::setDefaultAutoSWSB;
    using ngen::BinaryCodeGenerator<ngen::HW::Xe2>::setDefaultNoMask;
};

template <typename Body>
static Kernel buildKernel(const L0 &l0, const ngen::Product &product,
        const char *name, Body body) {
    ngen::InterfaceHandler iface(ngen::HW::Xe2);
    iface.externalName(name);
    iface.requireGRF(128);
    iface.requireSIMD(16);
    iface.requireDPAS();
    iface.requireWorkgroup(16, 1, 1);
    iface.newArgument("a", ngen::ExternalArgumentType::GlobalPtr,
            ngen::GlobalAccessType::Stateless);
    iface.newArgument("b", ngen::ExternalArgumentType::GlobalPtr,
            ngen::GlobalAccessType::Stateless);
    iface.newArgument("c", ngen::ExternalArgumentType::GlobalPtr,
            ngen::GlobalAccessType::Stateless);
    iface.newArgument("iters", ngen::DataType::ud);
    iface.finalize();

    LabGen gen(product);
    gen.setInterface(iface);
    gen.setDefaultAutoSWSB(true);
    gen.prologue();
    gen.setDefaultNoMask(true);
    body(gen, gen.getArgument("a"), gen.getArgument("b"), gen.getArgument("c"),
            gen.getArgument("iters"), gen.getGroupID(0));
    gen.epilogue();

    auto mk = gen.getModuleAndKernel(l0.context, l0.device);
    ZE_CHECK(zeKernelSetGroupSize(mk.second, 16, 1, 1));
    return {mk.first, mk.second};
}

struct RunConfig {
    uint32_t groups, iters;
};

static double runKernel(const L0 &l0, const Kernel &k, void *a, void *b, void *c,
        RunConfig cfg, int reps) {
    ZE_CHECK(zeKernelSetArgumentValue(k.kernel, 0, sizeof(void *), &a));
    ZE_CHECK(zeKernelSetArgumentValue(k.kernel, 1, sizeof(void *), &b));
    ZE_CHECK(zeKernelSetArgumentValue(k.kernel, 2, sizeof(void *), &c));
    ZE_CHECK(zeKernelSetArgumentValue(k.kernel, 3, sizeof(uint32_t), &cfg.iters));
    ze_group_count_t gc{cfg.groups, 1, 1};

    double best = 1e30;
    for (int r = 0; r < reps + 2; r++) { // 2 warmup
        auto t0 = std::chrono::steady_clock::now();
        ZE_CHECK(zeCommandListAppendLaunchKernel(
                l0.list, k.kernel, &gc, nullptr, 0, nullptr));
        const_cast<L0 &>(l0).sync();
        auto t1 = std::chrono::steady_clock::now();
        if (r >= 2)
            best = std::min(best,
                    std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

static uint32_t envU32(const char *name, uint32_t dflt) {
    const char *v = std::getenv(name);
    return v ? uint32_t(std::atoi(v)) : dflt;
}

static std::string envStr(const char *name, const char *dflt) {
    const char *v = std::getenv(name);
    return v ? v : dflt;
}

} // namespace ngen_lab

int main() {
    using namespace ngen_lab;

    const std::string which = envStr("NGEN_LAB_KERNELS", "both");
    const uint32_t perfGroups = envU32("NGEN_LAB_GROUPS", 8192);
    const uint32_t dpasIters = envU32("NGEN_LAB_DPAS_ITERS", 4096);
    const uint32_t scalarIters = envU32("NGEN_LAB_SCALAR_ITERS", 16);
    const int reps = int(envU32("NGEN_LAB_REPS", 10));
    const bool runDpas = which == "both" || which == "dpas";
    const bool runScalar = which == "both" || which == "scalar";

    const uint32_t chkGroups = 64, chkIters = 3;
    const uint32_t maxGroups = std::max(perfGroups, chkGroups);

    // Host data: random bf16 in [-1,1).
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<uint16_t> hA((size_t)maxGroups * kM * kK);
    std::vector<uint16_t> hBnat(kK * kN);
    for (auto &v : hA) v = f32ToBf16(dist(rng));
    for (auto &v : hBnat) v = f32ToBf16(dist(rng));

    std::vector<uint32_t> hBpacked(kBBytes / 4);
    packB(hBnat.data(), hBpacked.data());

    L0 l0;
    void *dA = l0.alloc((size_t)maxGroups * kABytes);
    void *dBpacked = l0.alloc(kBBytes);
    void *dBnat = l0.alloc(kBBytes);
    void *dC = l0.alloc((size_t)maxGroups * kCBytes);
    std::memcpy(dA, hA.data(), (size_t)maxGroups * kABytes);
    std::memcpy(dBpacked, hBpacked.data(), kBBytes);
    std::memcpy(dBnat, hBnat.data(), kBBytes);

    auto product = ngen::LevelZeroCodeGenerator<ngen::HW::Xe2>::detectHWInfo(
            l0.context, l0.device);

    Kernel kDpas, kScalar;
    if (runDpas)
        kDpas = buildKernel(l0, product, "ngen_lab_dpas",
                [](auto &g, auto a, auto b, auto c, auto it, auto gid) {
                    dpasKernelBody(g, a, b, c, it, gid);
                });
    if (runScalar)
        kScalar = buildKernel(l0, product, "ngen_lab_scalar",
                [](auto &g, auto a, auto b, auto c, auto it, auto gid) {
                    scalarKernelBody(g, a, b, c, it, gid);
                });

    // ---- correctness (64 tiles, 3 iterations) ------------------------------
    std::vector<float> cRef((size_t)chkGroups * kM * kN);
    cpuReference(hA, hBnat, chkGroups, chkIters, cRef);
    bool ok = true;

    auto check = [&](const char *name, Kernel &k, void *bPtr) {
        std::memset(dC, 0, (size_t)chkGroups * kCBytes);
        runKernel(l0, k, dA, bPtr, dC, {chkGroups, chkIters}, 1);
        std::vector<float> got((size_t)chkGroups * kM * kN);
        std::memcpy(got.data(), dC, got.size() * sizeof(float));
        auto s = compare(got, cRef);
        bool pass = s.cos > 0.9999 && s.maxRel < 1e-3;
        std::printf("[correctness] %-6s cos=%.8f max_abs=%.4g max_rel=%.4g -> %s\n",
                name, s.cos, s.maxAbs, s.maxRel, pass ? "PASS" : "FAIL");
        ok = ok && pass;
    };
    if (runDpas) check("dpas", kDpas, dBpacked);
    if (runScalar) check("scalar", kScalar, dBnat);
    if (!ok) {
        std::fprintf(stderr, "correctness failed; skipping perf\n");
        return 1;
    }

    // ---- performance -------------------------------------------------------
    const double flopsPerTile = 2.0 * kM * kN * kK;
    auto perf = [&](const char *name, Kernel &k, void *bPtr, uint32_t iters) {
        double t = runKernel(l0, k, dA, bPtr, dC, {perfGroups, iters}, reps);
        double flops = flopsPerTile * perfGroups * iters;
        std::printf("[perf] %-6s groups=%u iters=%u  %8.3f ms  %8.2f GFLOP/s\n",
                name, perfGroups, iters, t * 1e3, flops / t / 1e9);
    };
    if (runDpas) perf("dpas", kDpas, dBpacked, dpasIters);
    if (runScalar) perf("scalar", kScalar, dBnat, scalarIters);

    std::printf("note: textual assembly for both kernels is produced by the "
                "companion binary: ./ngen_lab_asmdump [dpas.asm] [scalar.asm]\n");
    return 0;
}
