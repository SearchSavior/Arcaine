// Minimal Level Zero + ngen bring-up ladder. Stage selected at compile time:
//   0: no-argument kernel, single mov (tests zebin/EOT only)
//   1: one ud scalar arg copied to a GRF (tests cross-thread payload)
//   2: one global pointer arg copied to a GRF, no deref (tests pointer ABI)
//   3: ptr arg + SIMD16 scattered UGM store of zeros (oneDNN zero_out pattern)
//   4: ptr arg + SIMD1 block store of zeros
//   5: baked-address store of the inline payload registers (payload dump)
//   6: full DPAS kernel body from ngen_dpas_8x8_matmul_bf16_16x16x32.hpp (probe)
//   7: full scalar kernel body from ngen_dpas_8x8_matmul_bf16_16x16x32.hpp (probe)
//
// Harness deliberately: enumerates queue groups for a COMPUTE queue, dumps the
// zebin to disk, creates the module explicitly and prints the build log, and
// synchronizes with a finite timeout + zeDeviceGetStatus.

#ifndef TEST_STAGE
#define TEST_STAGE 0
#endif

#include <level_zero/ze_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

#define TICK(msg) std::printf("[tick] %s\n", msg)

static constexpr size_t kBufBytes = 4096;
static constexpr int kSimd = 16;

int main() {
    std::printf("stage %d\n", TEST_STAGE);

    ZE_CHECK(zeInit(0));
    uint32_t ndrv = 0;
    ZE_CHECK(zeDriverGet(&ndrv, nullptr));
    std::vector<ze_driver_handle_t> drvs(ndrv);
    ZE_CHECK(zeDriverGet(&ndrv, drvs.data()));
    uint32_t ndev = 0;
    ZE_CHECK(zeDeviceGet(drvs[0], &ndev, nullptr));
    std::vector<ze_device_handle_t> devs(ndev);
    ZE_CHECK(zeDeviceGet(drvs[0], &ndev, devs.data()));
    int devIdx = std::atoi(getenv("L0_MIN_DEVIDX") ? getenv("L0_MIN_DEVIDX")
                                                   : "0");
    int gpuSeen = -1;
    ze_device_handle_t device = nullptr;
    for (auto d : devs) {
        ze_device_properties_t p{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, nullptr};
        ZE_CHECK(zeDeviceGetProperties(d, &p));
        if (p.type != ZE_DEVICE_TYPE_GPU) continue;
        gpuSeen++;
        std::printf("gpu %d: %s%s\n", gpuSeen, p.name,
                gpuSeen == devIdx ? "  [selected]" : "");
        if (gpuSeen == devIdx) device = d;
    }
    if (!device) {
        std::fprintf(stderr, "no GPU at index %d\n", devIdx);
        return 1;
    }

    // Pick a queue group with the COMPUTE flag; ordinal 0 is not guaranteed.
    uint32_t ngroups = 0;
    ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device, &ngroups, nullptr));
    std::vector<ze_command_queue_group_properties_t> groups(ngroups,
            {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr});
    ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(
            device, &ngroups, groups.data()));
    uint32_t computeOrd = UINT32_MAX;
    for (uint32_t i = 0; i < ngroups; i++) {
        std::printf("queue group %u: flags=0x%x queues=%u\n", i,
                groups[i].flags, groups[i].numQueues);
        if (computeOrd == UINT32_MAX
                && (groups[i].flags
                        & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE))
            computeOrd = i;
    }
    if (computeOrd == UINT32_MAX) {
        std::fprintf(stderr, "no compute queue group\n");
        return 1;
    }

    ze_context_handle_t context;
    ze_context_desc_t cdesc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ZE_CHECK(zeContextCreate(drvs[0], &cdesc, &context));
    ze_command_queue_desc_t qdesc{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
            nullptr, computeOrd, 0, 0, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
            ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t list;
    ZE_CHECK(zeCommandListCreateImmediate(context, device, &qdesc, &list));
    TICK("harness up (compute queue)");

#if TEST_STAGE >= 2
    float *buf = nullptr;
    ze_device_mem_alloc_desc_t ddesc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            nullptr, 0, 0};
    ze_host_mem_alloc_desc_t hdesc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
            nullptr, 0};
    ZE_CHECK(zeMemAllocShared(context, &ddesc, &hdesc, kBufBytes, 64, device,
            (void **)&buf));
    std::memset(buf, 0xFF, kBufBytes);
    TICK("USM allocated");
#endif

    // ---- interface ---------------------------------------------------------
    ngen::InterfaceHandler iface(ngen::HW::Xe2);
    iface.externalName("l0_min");
    iface.requireGRF(128);
    iface.requireSIMD(kSimd);
    iface.requireWorkgroup(kSimd, 1, 1);
    // NOTE: setInlineGRFCount(1) declared inline delivery but the driver left
    // r1-r2 all zero on BMG (stage 5 dump); the r0.0-indirect loadargs path
    // (ngen default on Xe2) is what oneDNN relies on. Keep default (0).
#if TEST_STAGE == 1
    iface.newArgument("x", ngen::DataType::ud);
#endif
#if TEST_STAGE >= 2
    iface.newArgument("c", ngen::ExternalArgumentType::GlobalPtr,
            ngen::GlobalAccessType::Stateless);
#endif
#if TEST_STAGE == 6 || TEST_STAGE == 7
    iface.newArgument("a", ngen::ExternalArgumentType::GlobalPtr,
            ngen::GlobalAccessType::Stateless);
    iface.newArgument("b", ngen::ExternalArgumentType::GlobalPtr,
            ngen::GlobalAccessType::Stateless);
    iface.newArgument("iters", ngen::DataType::ud);
#endif
    iface.finalize();

    auto product = ngen::LevelZeroCodeGenerator<
            ngen::HW::Xe2>::detectHWInfo(context, device);
    if (ngen::getCore(product.family) != ngen::HW::Xe2) {
        std::fprintf(stderr, "detected product core is not Xe2\n");
        return 1;
    }
    TICK("detectHWInfo ok (Xe2)");

    // Derived generator to expose the protected default-modifier setters,
    // matching oneDNN's generator pattern (setDefaultAutoSWSB *before*
    // prologue, setDefaultNoMask after).
    struct MinGen : ngen::LevelZeroCodeGenerator<ngen::HW::Xe2> {
        using ngen::LevelZeroCodeGenerator<ngen::HW::Xe2>::LevelZeroCodeGenerator;
        using ngen::BinaryCodeGenerator<ngen::HW::Xe2>::setDefaultAutoSWSB;
        using ngen::BinaryCodeGenerator<ngen::HW::Xe2>::setDefaultNoMask;
    };
    MinGen gen(product);
    gen.setInterface(iface);
    gen.setDefaultAutoSWSB(true);
    gen.prologue();
    gen.setDefaultNoMask(true);
    {
        using namespace ngen;
#if TEST_STAGE == 0
        gen.mov(1, GRF(10).ud(0), 0);
#elif TEST_STAGE == 1
        gen.mov(1, GRF(10).ud(0), gen.getArgument("x"));
#elif TEST_STAGE == 2
        gen.mov(1, GRF(10).uq(0), gen.getArgument("c"));
#elif TEST_STAGE == 3 || TEST_STAGE == 4
        // OneDNN zero_out pattern: scattered SIMD16 UGM store of zeros with a
        // per-lane address vector built from an uv immediate.
        GRF rBase(10), rIdx(12), rOff(13);
        GRF rPtrLo(14), rPtrHi(15); // 16-lane A64 address vector: 2 GRFs
        GRF rZero(16);
        gen.mov(1, rBase.uq(0), gen.getArgument("c"));
        gen.mov(16, rZero.ud(), 0u);
        gen.mov(8, rIdx.uw(0)(1), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
        gen.mov(8, rIdx.uw(8)(1),
                Immediate::uv(8, 9, 10, 11, 12, 13, 14, 15));
        // Widening idiom from ngen_emulation.hpp (emov ground truth): never
        // mix integer widths in one mov/add. Write the low subwords with a
        // same-type strided mov, then zero the high subwords.
        gen.mov(16, rOff.uw(0)(2), rIdx.uw(0)(1)); // uw->ud: low words
        gen.mov(16, rOff.uw(1)(2), uint16_t(0));   //         zero high words
        gen.shl(16, rOff.ud(0)(1), rOff.ud(0)(1), 2); // byte offsets
        gen.mov(8, rPtrLo.ud(0)(2), rOff.ud(0)(1)); // ud->uq: low dwords
        gen.mov(8, rPtrLo.ud(1)(2), 0u);            //         zero high
        gen.mov(8, rPtrHi.ud(0)(2), rOff.ud(8)(1));
        gen.mov(8, rPtrHi.ud(1)(2), 0u);
        gen.add(8, rPtrLo.uq(0)(1), rPtrLo.uq(0)(1), rBase.uq(0)(0));
        gen.add(8, rPtrHi.uq(0)(1), rPtrHi.uq(0)(1), rBase.uq(0)(0));
#if TEST_STAGE == 3
        gen.store.ugm(16, scattered(DataSizeLSC::D32, 1), gen.A64, rPtrLo,
                rZero);
#else
        gen.store.ugm(1, block(DataSizeLSC::D32, 4), gen.A64, rBase, rZero);
#endif
#elif TEST_STAGE == 6 || TEST_STAGE == 7
        // Full DPAS/scalar kernel body (probe): A/B/C pointers + iters=1.
        if (TEST_STAGE == 6)
            ngen_lab::dpasKernelBody(gen, gen.getArgument("a"),
                    gen.getArgument("b"), gen.getArgument("c"),
                    gen.getArgument("iters"), gen.getGroupID(0));
        else
            ngen_lab::scalarKernelBody(gen, gen.getArgument("a"),
                    gen.getArgument("b"), gen.getArgument("c"),
                    gen.getArgument("iters"), gen.getGroupID(0));
#elif TEST_STAGE == 5
        // Baked-address store: no reliance on argument delivery. Dumps the
        // inline payload registers r1-r2 (128B) to the USM buffer so the host
        // can see exactly what the driver delivered and where.
        const uint64_t baked = (uint64_t)(uintptr_t)buf;
        GRF rAddr(10);
        for (int i = 0; i < 4; i++) {
            uint64_t a = baked + 32 * i;
            gen.mov(1, rAddr.ud(0), uint32_t(a & 0xFFFFFFFFu));
            gen.mov(1, rAddr.ud(1), uint32_t(a >> 32));
            gen.store.ugm(1, block(DataSizeLSC::D32, 8), gen.A64, rAddr,
                    GRF(1 + i));
        }
#endif
    }
    gen.epilogue();
    TICK("kernel generated");

    // ---- dump zebin ---------------------------------------------------------
    const auto zebin = gen.getBinary();
    {
        char path[64];
        std::snprintf(path, sizeof(path), "l0_min_s%d.zebin", TEST_STAGE);
        std::ofstream os(path, std::ios::binary);
        os.write((const char *)zebin.data(), (std::streamsize)zebin.size());
        std::printf("[tick] zebin dumped: %s (%zu bytes)\n", path,
                zebin.size());
    }

    // ---- explicit module create with build log ------------------------------
    ze_module_desc_t mdesc{ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr,
            ZE_MODULE_FORMAT_NATIVE, zebin.size(), zebin.data(), nullptr,
            nullptr};
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
    ZE_CHECK(rc);
    TICK("module created");

    ze_kernel_desc_t kdesc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
            "l0_min"};
    ze_kernel_handle_t kernel = nullptr;
    ZE_CHECK(zeKernelCreate(module, &kdesc, &kernel));
    ZE_CHECK(zeKernelSetGroupSize(kernel, kSimd, 1, 1));
#if TEST_STAGE == 1
    uint32_t xval = 42;
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(xval), &xval));
#elif TEST_STAGE == 6 || TEST_STAGE == 7
    // Stage 6/7: A/B/C shared buffers (buf holds C) + iters=1.
    void *a = buf, *b = buf + ngen_lab::kBBytes, *c = buf;
    uint32_t iters = 1;
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(void *), &a));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(void *), &b));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 2, sizeof(void *), &c));
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 3, sizeof(uint32_t), &iters));
#elif TEST_STAGE >= 2
    ZE_CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(void *), &buf));
#endif

    ze_group_count_t gc{1, 1, 1};
    ZE_CHECK(zeCommandListAppendLaunchKernel(
            list, kernel, &gc, nullptr, 0, nullptr));
    TICK("launched");

    // Finite sync; report device status on failure.
    ze_result_t src_ = zeCommandListHostSynchronize(list, 10000000000ull);
    if (src_ != ZE_RESULT_SUCCESS) {
        ze_result_t st = zeDeviceGetStatus(device);
        std::fprintf(stderr,
                "sync failed: 0x%x, zeDeviceGetStatus=0x%x (%s)\n",
                (unsigned)src_, (unsigned)st,
                st == ZE_RESULT_SUCCESS ? "available" : "lost/unavailable");
        return 1;
    }
    TICK("sync done");

#if TEST_STAGE == 3
    int good = 0;
    for (int i = 0; i < int(kBufBytes / 4); i++) {
        uint32_t u;
        std::memcpy(&u, &buf[i], 4);
        if (i < 16 && u == 0u) good++;           // zeroed by the store
        if (i >= 16 && u != 0xFFFFFFFFu) good--; // clobbered outside
    }
    std::printf("[check] zeroed lanes: %d/16 (no out-of-range writes) -> %s\n",
            good, good == 16 ? "PASS" : "FAIL");
#elif TEST_STAGE == 4
    int expectZero = 4;
    int zeroed = 0, changed = 0;
    for (int i = 0; i < int(kBufBytes / 4); i++) {
        if (buf[i] == 0.f) zeroed++;
        uint32_t u;
        std::memcpy(&u, &buf[i], 4);
        if (u != 0xFF00FF00u) { // memset(0xFF) bit pattern
            if (u != 0xFFFFFFFFu) changed++;
        }
    }
    int nonFF = 0;
    for (int i = 0; i < int(kBufBytes / 4); i++) {
        uint32_t u;
        std::memcpy(&u, &buf[i], 4);
        if (u != 0xFFFFFFFFu) nonFF++;
    }
    std::printf("[check] zeroed floats: %d, non-0xFF dwords: %d "
                "(expect >= %d zeroed) -> %s\n",
            zeroed, nonFF, expectZero,
            zeroed >= expectZero ? "PASS" : "FAIL");
#elif TEST_STAGE == 5
    std::printf("[dump] buf=%p; inline payload qwords (r1-r2):\n",
            (void *)buf);
    for (int i = 0; i < 16; i++) {
        uint64_t q;
        std::memcpy(&q, (const char *)buf + 8 * i, 8);
        std::printf("  q[%2d] = 0x%016llx%s\n", i, (unsigned long long)q,
                q == (uint64_t)(uintptr_t)buf ? "  <-- == buf" : "");
    }
#endif
    std::printf("stage %d complete\n", TEST_STAGE);
    return 0;
}
