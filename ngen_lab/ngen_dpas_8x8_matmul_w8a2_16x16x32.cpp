// nGEN-authored fixed-shape s8x2->s32 (test: s2 in src2) 16x16x32 DPAS matmul microkernel.
//
//   C[16x16] (s32) += A[16x32] (s8, row-major) * B[32x16] (s8)
//
// Same structure as the BF16 variant (ngen_dpas_8x8_matmul_bf16_16x16x32):
// SIMD16, lane m = row m of C. Tunable tiles-per-thread: NGEN_LAB_S2_TILES
// selects how many independent 16x16x32 tiles one thread accumulates (more
// tiles = more independent dpas chains for latency hiding; B is shared).
//
// DPAS register layout (Xe2, 64B GRFs, SIMD16, lanes = M):
//   A operand (src1, per-lane):   GRF j, lane m = row m's dword j =
//                                 {A[m][4j], A[m][4j+1], A[m][4j+2], A[m][4j+3]}
//                                 (4 s8 per dword along K). Loaded with 8
//                                 scattered D32x1 loads (one per A GRF).
//   B operand (src2, broadcast):  dword (j, r) = {B[4j][r], B[4j+1][r],
//                                 B[4j+2][r], B[4j+3][r]} (VNNI). packB
//                                 produces 2 N-halves of 256B; loaded as
//                                 contiguous 64B block chunks.
//   C accumulator (dst/src0):     GRF n, lane m = C[m][n] (s32).
// One dpas(16 lanes, sdepth 8, rcount 8) covers C[16 lanes x 8 cols] += K=32;
// the full tile needs 2 dpas (2 N halves).
//
// Integer widening and address-vector idioms follow the BF16 variant (see
// docs/findings.md): same-type strided movs, GRF(base+1) for high halves.

#include <level_zero/ze_api.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ngen.hpp"
#include "ngen_level_zero.hpp"

#include "ngen_dpas_8x8_matmul_w8a2_16x16x32.hpp"

using namespace ngen_lab::w8a2;
// Derived generator exposing the protected default-modifier setters (SWSB +
// NoMask), matching the BF16 harness and oneDNN's pattern.
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

// ---- B packer -------------------------------------------------------------
// Packs natural row-major B[32][16] (s8) into the DPAS src2 VNNI layout:
//   piece h (N half 8h..8h+7), dword (j, r) =
//       {B[4j][8h+r], B[4j+1][8h+r], B[4j+2][8h+r], B[4j+3][8h+r]}
//   where j = K dword (0..7), r = column within half (0..7).
// Each half is 8x8 dwords = 256B.
static void packB(const int8_t *bNat, uint32_t *bPacked) {
    for (int h = 0; h < 2; h++)
        for (int r = 0; r < 8; r++)
            for (int j = 0; j < 2; j++) {
                uint32_t d = 0;
                for (int e = 0; e < 16; e++) {
                    int8_t v = bNat[(16 * j + e) * kN + 8 * h + r];
                    d |= (uint32_t)(v & 0x3) << (2 * e);
                }
                bPacked[(h * 16) + r * 2 + j] = d;
            }
}

// ---- CPU reference --------------------------------------------------------
static void cpuReference(const std::vector<int8_t> &a,
        const std::vector<int8_t> &bNat, int tiles, int iters,
        std::vector<int32_t> &cRef) {
    for (int t = 0; t < tiles; t++)
        for (int m = 0; m < kM; m++)
            for (int n = 0; n < kN; n++) {
                int32_t acc = 0;
                for (int k = 0; k < kK; k++)
                    acc += int32_t(a[(size_t)t * kM * kK + m * kK + k])
                            * int32_t(bNat[k * kN + n]);
                cRef[(size_t)t * kM * kN + m * kN + n] = acc * iters;
            }
}

// ---- Level Zero -----------------------------------------------------------
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
            ze_device_properties_t p{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, nullptr};
            ZE_CHECK(zeDeviceGetProperties(d, &p));
            if (p.type != ZE_DEVICE_TYPE_GPU) continue;
            gpuSeen++;
            std::printf("gpu %d: %s%s\n", gpuSeen, p.name,
                    gpuSeen == devIdx ? "  [selected]" : "");
            if (gpuSeen == devIdx) device = d;
        }
        if (!device) throw std::runtime_error("no GPU at NGEN_LAB_DEVIDX");
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
    iface.externalName("ngen_lab_int8");
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
    dpasKernelBody(gen, gen.getArgument("a"), gen.getArgument("b"),
            gen.getArgument("c"), gen.getArgument("iters"),
            gen.getGroupID(0));
    gen.epilogue();

    auto mk = gen.getModuleAndKernel(l0.context, l0.device);
    ZE_CHECK(zeKernelSetGroupSize(mk.second, 16, 1, 1));
    return {mk.first, mk.second};
}

static uint32_t envU32(const char *name, uint32_t dflt) {
    const char *v = std::getenv(name);
    return v ? uint32_t(std::atoi(v)) : dflt;
}

static std::string envStr(const char *name, const char *dflt) {
    const char *v = std::getenv(name);
    return v ? v : dflt;
}

int main() {
    const uint32_t perfGroups = envU32("NGEN_LAB_GROUPS", 8192);
    const uint32_t iters = envU32("NGEN_LAB_DPAS_ITERS", 4096);
    const int reps = int(envU32("NGEN_LAB_REPS", 10));
    kTiles = int(envU32("NGEN_LAB_S2_TILES", 1));
    // T=3/4 overflow the 128-GRF map (pointer vectors exceed GRF 127) and,
    // per measurement, provide no benefit anyway (T=2 is already slower than
    // T=1 at this shape — 2048 HW threads supply the independent dpas work).
    if (kTiles < 1 || kTiles > 2) {
        std::fprintf(stderr, "NGEN_LAB_S2_TILES must be in [1,2]\n");
        return 1;
    }
    const uint32_t chkGroups = 64, chkIters = 3;
    const uint32_t maxGroups = std::max(perfGroups, chkGroups);

    // Host data: random s8 in [-128, 127].
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> distA(-128, 127), distB(-2, 1);
    std::vector<int8_t> hA((size_t)maxGroups * kM * kK);
    std::vector<int8_t> hBnat(kK * kN);
    for (auto &v : hA) v = int8_t(distA(rng));
    for (auto &v : hBnat) v = int8_t(distB(rng));

    std::vector<uint32_t> hBpacked(kBBytes / 4);
    packB(hBnat.data(), hBpacked.data());

    L0 l0;
    void *dA = l0.alloc((size_t)maxGroups * kABytes);
    void *dBpacked = l0.alloc(kBBytes);
    void *dC = l0.alloc((size_t)maxGroups * kCBytes);
    std::memcpy(dA, hA.data(), (size_t)maxGroups * kABytes);
    std::memcpy(dBpacked, hBpacked.data(), kBBytes);

    auto product = ngen::LevelZeroCodeGenerator<ngen::HW::Xe2>::detectHWInfo(
            l0.context, l0.device);
    Kernel k = buildKernel(l0, product);

    // ---- correctness (64 tiles, 3 iterations) ------------------------------
    std::vector<int32_t> cRef((size_t)chkGroups * kM * kN);
    cpuReference(hA, hBnat, chkGroups, chkIters, cRef);

    std::memset(dC, 0, (size_t)chkGroups * kCBytes);
    ZE_CHECK(zeKernelSetArgumentValue(k.kernel, 0, sizeof(void *), &dA));
    ZE_CHECK(zeKernelSetArgumentValue(k.kernel, 1, sizeof(void *), &dBpacked));
    ZE_CHECK(zeKernelSetArgumentValue(k.kernel, 2, sizeof(void *), &dC));
    uint32_t chkItersU = chkIters;
    ZE_CHECK(zeKernelSetArgumentValue(k.kernel, 3, sizeof(uint32_t), &chkItersU));
    ze_group_count_t gc{chkGroups / kTiles, 1, 1};
    ZE_CHECK(zeCommandListAppendLaunchKernel(l0.list, k.kernel, &gc, nullptr, 0,
            nullptr));
    l0.sync();

    std::vector<int32_t> got((size_t)chkGroups * kM * kN);
    std::memcpy(got.data(), dC, got.size() * sizeof(int32_t));
    size_t mismatches = 0;
    for (size_t i = 0; i < got.size(); i++)
        if (got[i] != cRef[i]) mismatches++;
    bool ok = mismatches == 0;
    std::printf("[correctness] w8a2 dpas (tiles=%d) mismatches=%zu/%zu -> %s\n",
            kTiles, mismatches, got.size(), ok ? "PASS" : "FAIL");
    if (!ok) {
        for (size_t i = 0; i < got.size(); i++)
            if (got[i] != cRef[i]) {
                std::printf("  first mismatch at tile %zu: got=%d ref=%d\n",
                        i / (kM * kN), got[i], cRef[i]);
                break;
            }
        return 1;
    }

    // ---- performance -------------------------------------------------------
    uint32_t perfIters = iters;
    ZE_CHECK(zeKernelSetArgumentValue(k.kernel, 3, sizeof(uint32_t), &perfIters));
    ze_group_count_t pgc{perfGroups / kTiles, 1, 1};
    double best = 1e30;
    for (int r = 0; r < reps + 2; r++) {
        auto t0 = std::chrono::steady_clock::now();
        ZE_CHECK(zeCommandListAppendLaunchKernel(
                l0.list, k.kernel, &pgc, nullptr, 0, nullptr));
        l0.sync();
        auto t1 = std::chrono::steady_clock::now();
        if (r >= 2)
            best = std::min(best,
                    std::chrono::duration<double>(t1 - t0).count());
    }
    const double flops = 2.0 * kM * kN * kK * perfGroups * iters;
    std::printf("[perf] w8a2 dpas (tiles=%d) groups=%u iters=%u  %8.3f ms  "
                "%8.2f GFLOP/s\n",
            kTiles, perfGroups, iters, best * 1e3, flops / best / 1e9);
    return 0;
}
