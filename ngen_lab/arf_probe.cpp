// Minimal ARF dpas probe: does s8 x s4 dpas into an accumulator register work
// on Xe2 (L0), and how must the acc be read back? Variants driven by
// ARF_PROBE_SRC0 (0=GRF zero, 1=chained same-acc, 2=pre-zeroed other acc) and
// ARF_PROBE_DST (0=acc0/acc1 even, 1=acc0/acc2, 2=acc1/acc3 odd).
#include <level_zero/ze_api.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "ngen.hpp"
#include "ngen_level_zero.hpp"

using namespace ngen;

struct LabGen : LevelZeroCodeGenerator<HW::Xe2> {
    using LevelZeroCodeGenerator<HW::Xe2>::LevelZeroCodeGenerator;
    using BinaryCodeGenerator<HW::Xe2>::setDefaultAutoSWSB;
    using BinaryCodeGenerator<HW::Xe2>::setDefaultNoMask;
};

#define ZE_CHECK(x) do { ze_result_t r_=(x); if(r_!=ZE_RESULT_SUCCESS){ \
    std::fprintf(stderr,"ZE error 0x%x at %s:%d\n",(unsigned)r_,__FILE__,__LINE__); std::exit(1);} } while(0)

static uint32_t envU32(const char *n, uint32_t d) {
    const char *v = std::getenv(n); return v ? uint32_t(std::atoi(v)) : d;
}

int main() {
    const int src0mode = int(envU32("ARF_PROBE_SRC0", 0));
    const int dstmode  = int(envU32("ARF_PROBE_DST", 0));
    const int nodpas   = int(envU32("ARF_PROBE_NODPAS", 0));
    // A: 16x32 s8; B: 16x32 s4 packed; expect C = A@B^T (16x16 s32).
    int8_t hA[16 * 32], hBp[16 * 16];
    for (int i = 0; i < 16 * 32; i++) hA[i] = (int8_t)((i * 7) % 32 - 16);
    for (int i = 0; i < 16 * 16; i++) hBp[i] = 0;
    for (int n = 0; n < 16; n++)
        for (int k = 0; k < 32; k++) {
            int v = (n * 3 + k * 5) % 16 - 8;
            int b = n * 16 + k / 2;
            hBp[b] |= (uint8_t)(v & 0xF) << (4 * (k & 1));
        }
    // Pack B for dpas src2 (packB from the w4a8 harness): dword (h,r,j).
    uint32_t Bpk[64];
    for (int h = 0; h < 2; h++)
        for (int r = 0; r < 8; r++) {
            int n = 8 * h + r;
            for (int j = 0; j < 4; j++) {
                uint32_t d = 0;
                for (int e = 0; e < 8; e++)
                    d |= (uint32_t)((hBp[n * 16 + (8 * j + e) / 2] >> (4 * ((8 * j + e) & 1))) & 0xF) << (4 * e);
                Bpk[h * 32 + r * 4 + j] = d;
            }
        }

    zeInit(0);
    uint32_t nd = 0; zeDriverGet(&nd, nullptr);
    std::vector<ze_driver_handle_t> drvs(nd); zeDriverGet(&nd, drvs.data());
    ze_driver_handle_t drv = drvs[0];
    uint32_t ndev = 0; zeDeviceGet(drv, &ndev, nullptr);
    std::vector<ze_device_handle_t> devs(ndev); zeDeviceGet(drv, &ndev, devs.data());
    ze_device_handle_t dev = nullptr;
    for (auto d : devs) {
        ze_device_properties_t p{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, nullptr};
        zeDeviceGetProperties(d, &p);
        if (p.type == ZE_DEVICE_TYPE_GPU) { printf("gpu: %s\n", p.name); dev = d; break; }
    }
    ze_context_desc_t cd{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ze_context_handle_t ctx; zeContextCreate(drv, &cd, &ctx);
    uint32_t ng = 0; zeDeviceGetCommandQueueGroupProperties(dev, &ng, nullptr);
    std::vector<ze_command_queue_group_properties_t> gs(ng, {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr});
    zeDeviceGetCommandQueueGroupProperties(dev, &ng, gs.data());
    uint32_t cq = 0;
    for (uint32_t i = 0; i < ng; i++) if (gs[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) { cq = i; break; }
    ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, cq, 0, 0,
        ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
    ze_command_list_handle_t list; zeCommandListCreateImmediate(ctx, dev, &qd, &list);
    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dZ = nullptr;
    ze_device_mem_alloc_desc_t dd{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    ze_host_mem_alloc_desc_t hd{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
    zeMemAllocShared(ctx, &dd, &hd, 16 * 32, 64, dev, &dA);
    zeMemAllocShared(ctx, &dd, &hd, 256, 64, dev, &dB);
    zeMemAllocShared(ctx, &dd, &hd, 16 * 16 * 4, 64, dev, &dC);
    zeMemAllocShared(ctx, &dd, &hd, 16 * 16 * 4, 64, dev, &dZ);
    memcpy(dA, hA, 16 * 32); memcpy(dB, Bpk, 256); memset(dC, 0, 16 * 16 * 4); memset(dZ, 0, 16 * 16 * 4);

    InterfaceHandler iface(HW::Xe2);
    iface.externalName("arf_probe");
    iface.requireGRF(envU32("ARF_PROBE_GRF", 256));
    iface.requireSIMD(16);
    iface.requireDPAS();
    iface.requireWorkgroup(16, 1, 1);
    iface.newArgument("a", ExternalArgumentType::GlobalPtr, GlobalAccessType::Stateless);
    iface.newArgument("b", ExternalArgumentType::GlobalPtr, GlobalAccessType::Stateless);
    iface.newArgument("c", ExternalArgumentType::GlobalPtr, GlobalAccessType::Stateless);
    iface.newArgument("z", ExternalArgumentType::GlobalPtr, GlobalAccessType::Stateless);
    iface.finalize();
    auto product = LevelZeroCodeGenerator<HW::Xe2>::detectHWInfo(ctx, dev);
    LabGen gen(product);
    gen.setInterface(iface);
    gen.setDefaultAutoSWSB(true);
    gen.prologue();
    gen.setDefaultNoMask(true);

    GRF rPtrA(46), rPtrB(47), rPtrC(48);
    gen.mov(1, rPtrA.uq(0), gen.getArgument("a"));
    gen.mov(1, rPtrB.uq(0), gen.getArgument("b"));
    gen.mov(1, rPtrC.uq(0), gen.getArgument("c"));
    // Load A (8 GRFs), B (4 GRFs) block loads; zero block into rZ..rZ+7.
    for (int j = 0; j < 4; j++)
        gen.load(1, GRF(16 + 2 * j), block(DataSizeLSC::D32, 32), gen.A64,
                rPtrA + 128 * j);
    for (int j = 0; j < 4; j++)
        gen.load(1, GRF(24 + j), block(DataSizeLSC::D32, 16), gen.A64,
                rPtrB + 64 * j);
    for (int i = 0; i < 8; i += 2) gen.mov(32, GRF(28 + i).d(), 0);

    // Pre-zero the destination accs (for the chained/other-acc src0 variants).
    for (int a = 0; a < 4; a++)
        gen.mov(16, AccumulatorRegister(a).sub(0, DataType::d), 0);

    // The dpas for the two halves. DST mapping depends on dstmode.
    // 0=acc0/1 (plain FPU accs), 1=acc2/3, 2=mme0/1, 3=acc4/5, 4=acc8/9.
    int accIdx[2];
    switch (dstmode) {
    case 1: accIdx[0] = 2; accIdx[1] = 3; break;
    case 2: accIdx[0] = 4; accIdx[1] = 5; break;
    case 3: accIdx[0] = 6; accIdx[1] = 7; break;
    case 4: accIdx[0] = 8; accIdx[1] = 9; break;
    default: accIdx[0] = 0; accIdx[1] = 1; break;
    }
    auto dstAcc = [&](int h) { return AccumulatorRegister(accIdx[h]); };
    const int grfdst = int(envU32("ARF_PROBE_GRFDST", 0));
    if (!nodpas)
    for (int h = 0; h < 2; h++) {
        RegData dst, src0;
        if (grfdst) { dst = GRF(60 + 8 * h).d(); src0 = GRF(60 + 8 * h).d(); }
        else {
            dst = dstAcc(h).sub(0, DataType::d);
            if (src0mode == 0)      src0 = GRF(28).d();                    // GRF zero
            else if (src0mode == 1) src0 = dstAcc(h).sub(0, DataType::d);  // chain
            else                    src0 = AccumulatorRegister(6 + h).sub(0, DataType::d); // other acc
        }
        const bool fwd = envU32("ARF_PROBE_FWD", 0) != 0;
        const InstructionModifier mod = fwd ? InstructionModifier::createFwd()
                                            : InstructionModifier();
        gen.dpas(mod, 8, 8, dst, src0, GRF(16).b(), GRF(24 + 2 * h).s4());
    }
    // Per-lane C address vector (2 GRFs, 16 uq): lane m -> C + m*4 bytes.
    gen.mov(8, GRF(44).uw(0)(1), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
    gen.mov(8, GRF(44).uw(8)(1), Immediate::uv(8, 9, 10, 11, 12, 13, 14, 15));
    gen.mov(16, GRF(44).ud(0)(2), GRF(44).uw(0)(1));
    gen.mov(16, GRF(45).ud(0)(2), 0u);
    gen.mul(16, GRF(44).ud(0)(2), GRF(44).ud(0)(2), 4u);
    gen.add(8, GRF(44).uq(0)(1), GRF(44).uq(0)(1), rPtrC.uq(0));
    gen.add(8, GRF(45).uq(0)(1), GRF(45).uq(0)(1), rPtrC.uq(0));
    // Read accs back: cvt whole-halves to f32 GRFs, store per-lane.
    for (int h = 0; h < 2; h++)
        for (int c = 0; c < 4; c++) {
            RegData src;
            if (grfdst) src = GRF(60 + 8 * h + 2 * c).d();
            else src = dstAcc(h).sub(32 * c, DataType::d);
            gen.mov(32, GRF(36 + 2 * c).f(), src);
            for (int n = 0; n < 2; n++) {
                const int col = 8 * h + 2 * c + n;
                gen.store(16, scattered(DataSizeLSC::D32, 1), gen.A64,
                        GRF(44) + 4 * (col * 16), GRF(36 + 2 * c + n));
            }
        }
    gen.epilogue();
    auto mk = gen.getModuleAndKernel(ctx, dev);
    zeKernelSetGroupSize(mk.second, 16, 1, 1);
    zeKernelSetArgumentValue(mk.second, 0, sizeof(void*), &dA);
    zeKernelSetArgumentValue(mk.second, 1, sizeof(void*), &dB);
    zeKernelSetArgumentValue(mk.second, 2, sizeof(void*), &dC);
    zeKernelSetArgumentValue(mk.second, 3, sizeof(void*), &dZ);
    ze_group_count_t gc{1, 1, 1};
    zeCommandListAppendLaunchKernel(list, mk.second, &gc, nullptr, 0, nullptr);
    auto t0 = std::chrono::steady_clock::now();
    ze_result_t r = zeCommandListHostSynchronize(list, 10000000000ull);
    auto t1 = std::chrono::steady_clock::now();
    if (r != ZE_RESULT_SUCCESS) {
        ze_result_t st = zeDeviceGetStatus(dev);
        std::fprintf(stderr, "sync failed: 0x%x, zeDeviceGetStatus=0x%x\n", (unsigned)r, (unsigned)st);
        return 1;
    }
    std::printf("sync ok in %.1f ms\n",
            std::chrono::duration<double>(t1 - t0).count() * 1e3);
    // CPU ref.
    int mism = 0;
    for (int n = 0; n < 16; n++) {
        int sum = 0;
        for (int k = 0; k < 32; k++)
            sum += (int)hA[k] * (int)((hBp[n * 16 + k / 2] >> (4 * (k & 1))) & 0xF);
        int got = ((int32_t*)dC)[n * 16];
        if (got != sum) {
            if (mism < 4) std::printf("  col %d got=%d ref=%d\n", n, got, sum);
            mism++;
        }
    }
    std::printf("[ARF probe] src0=%d dst=%d -> %s (%d/16 mismatches)\n",
            src0mode, dstmode, mism ? "FAIL" : "PASS", mism);
    return mism ? 1 : 0;
}
