// De-risk the fused-MoE thesis: does ONE grouped DPAS launch over all 128
// experts reach a meaningful fraction of VRAM bandwidth, vs the per-expert
// loop (128 separate launches) the production path uses?
//
// Models the gate/up expert matmul at the real shape: E=128 experts, M=16
// tokens/expert (padded), K=H=2816, N=2*moe_inter=1408. Weight-only (bf16
// activations, FP4 weights). Both variants use K-split for occupancy.
//
// Build:
//   icpx -fsycl -O2 -Xspirv-translator \
//     -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate \
//     tools/dpas_moe_grouped_bench.cpp -o dpas_moe_grouped_bench
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>

using v8s = short __attribute__((ext_vector_type(8)));
using v8i = int   __attribute__((ext_vector_type(8)));
using v8f = float __attribute__((ext_vector_type(8)));
SYCL_EXTERNAL v8f __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
    int KDim, v8s A, v8i B, v8f C, int Operands)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
    { return v8f{}; }
#endif
static constexpr int kBF16Ops = 0x3000;

static inline uint16_t f2bf(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)((u + ((u >> 16) & 1) + 0x7FFFu) >> 16);
}
static inline float e4m3_fast(uint8_t b) {
    uint32_t exp = (b >> 3) & 0x0f, mant = b & 0x07;
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    if (exp == 0) return sign * (float)mant * (1.0f / 512.0f);
    uint32_t bits = ((uint32_t)(b & 0x80) << 24) | ((exp - 7 + 127) << 23) | (mant << 20);
    float out; __builtin_memcpy(&out, &bits, 4); return out;
}
static inline float e2m1_fast(uint8_t bits) {
    const float mag[8] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
    return (bits & 8) ? -mag[bits & 7] : mag[bits & 7];
}
// Dequant FP4 weight column n, K-tile k0 -> VNNI bf16 B operand. Weight base
// pointers are per-expert; w_packed is [N,K/2], w_scale is [K/16,N].
static inline v8i dqb(const uint8_t* wp, const uint8_t* ws, int n, int k0, int K, int N) {
    float s = e4m3_fast(ws[(size_t)(k0/16)*N + n]);
    const uint8_t* row = wp + (size_t)n*(K/2) + k0/2;
    v8i b;
    for (int j=0;j<8;++j){ uint8_t by=row[j];
        b[j]=(int)((uint32_t)f2bf(e2m1_fast(by&0xf)*s)|((uint32_t)f2bf(e2m1_fast(by>>4)*s)<<16)); }
    return b;
}

// Coalesced/blocked dequant: weights repacked so the 16 lanes of a sub-group
// (columns n0..n0+15) read 128 contiguous bytes per K-tile. Layout per expert:
//   wc[ n_tile*(K/16)*128 + k_tile*128 + lane*8 + j ]  (16 cols x 8 bytes).
static inline v8i dqb_coal(const uint8_t* wc, const uint8_t* ws, int n, int lane,
                           int k0, int K, int N) {
    float s = e4m3_fast(ws[(size_t)(k0/16)*N + n]);
    const uint8_t* row = wc + (size_t)(n/16)*(K/16)*128 + (size_t)(k0/16)*128 + lane*8;
    v8i b;
    for (int j=0;j<8;++j){ uint8_t by=row[j];
        b[j]=(int)((uint32_t)f2bf(e2m1_fast(by&0xf)*s)|((uint32_t)f2bf(e2m1_fast(by>>4)*s)<<16)); }
    return b;
}

constexpr int E=128, M=16, K=2816, N=1408;   // gate/up shape
constexpr int TR = E*M;                       // total rows

static int ksplit(int rows,int n){ int base=(rows/8)*(n/16); int ks=(2048+base-1)/std::max(base,1); return std::clamp(ks,1,std::min(K/16,32)); }

int main(){
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(7); std::uniform_int_distribution<int> nib(0,15), sc(40,110);
    std::normal_distribution<float> an(0,1);

    size_t wpN=(size_t)E*N*(K/2), wsN=(size_t)E*(K/16)*N, aN=(size_t)TR*K, cN=(size_t)TR*N;
    uint8_t* wp=sycl::malloc_device<uint8_t>(wpN,q);
    uint8_t* ws=sycl::malloc_device<uint8_t>(wsN,q);
    uint16_t* A=sycl::malloc_device<uint16_t>(aN,q);
    uint16_t* C=sycl::malloc_device<uint16_t>(cN,q);
    { std::vector<uint8_t> h(wpN); for(auto&v:h)v=(uint8_t)(nib(rng)|(nib(rng)<<4)); q.memcpy(wp,h.data(),wpN).wait(); }
    { std::vector<uint8_t> h(wsN); for(auto&v:h)v=(uint8_t)sc(rng); q.memcpy(ws,h.data(),wsN).wait(); }
    { std::vector<uint16_t> h(aN); for(auto&v:h)v=f2bf(an(rng)); q.memcpy(A,h.data(),aN).wait(); }

    // Repack weights into the coalesced/blocked layout for the third variant.
    uint8_t* wc=sycl::malloc_device<uint8_t>(wpN,q);
    {
        std::vector<uint8_t> src(wpN), dst(wpN);
        q.memcpy(src.data(),wp,wpN).wait();
        for(int e=0;e<E;++e){ const uint8_t* se=src.data()+(size_t)e*N*(K/2); uint8_t* de=dst.data()+(size_t)e*N*(K/2);
            for(int n=0;n<N;++n) for(int kt=0;kt<K/16;++kt) for(int j=0;j<8;++j)
                de[(size_t)(n/16)*(K/16)*128 + (size_t)kt*128 + (n%16)*8 + j] = se[(size_t)n*(K/2) + kt*8 + j]; }
        q.memcpy(wc,dst.data(),wpN).wait();
    }

    double wbytes=(double)wpN + (double)wsN;   // FP4 weights + f8 scales read once
    int iters=50;

    // ---- (a) per-expert loop: E separate launches ----
    auto loop_once=[&]{
        int KS=ksplit(M,N);
        for(int e=0;e<E;++e){
            const uint8_t* wpe=wp+(size_t)e*N*(K/2); const uint8_t* wse=ws+(size_t)e*(K/16)*N;
            const uint16_t* Ae=A+(size_t)e*M*K; uint16_t* Ce=C+(size_t)e*M*N;
            q.submit([&](sycl::handler& h){
                sycl::local_accessor<float,1> slm((size_t)KS*8*16,h);
                h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)(M/8)*KS,N),sycl::range<2>(KS,16)),
                    [=](sycl::nd_item<2> it)[[sycl::reqd_sub_group_size(16)]]{
                        int m0=(int)it.get_group(0)*8, s=(int)it.get_local_id(0), lane=(int)it.get_local_id(1);
                        int n=(int)it.get_group(1)*16+lane; v8f c={0,0,0,0,0,0,0,0};
                        for(int kt=s;kt<K/16;kt+=KS){ int k0=kt*16; v8i b=dqb(wpe,wse,n,k0,K,N);
                            v8s a; for(int m=0;m<8;++m)a[m]=(short)Ae[(size_t)(m0+m)*K+k0+lane];
                            c=__spirv_SubgroupMatrixMultiplyAccumulateINTEL(16,a,b,c,kBF16Ops); }
                        for(int m=0;m<8;++m) slm[(size_t)(s*8+m)*16+lane]=c[m];
                        it.barrier(sycl::access::fence_space::local_space);
                        if(s==0)for(int m=0;m<8;++m){ float sum=0; for(int ss=0;ss<KS;++ss)sum+=slm[(size_t)(ss*8+m)*16+lane];
                            Ce[(size_t)(m0+m)*N+lane]=f2bf(sum); } });
            });
        }
    };

    // ---- (b) grouped: ONE launch over all tiles; tile -> expert via m0/M ----
    auto grouped_once=[&]{
        int KS=ksplit(TR,N);
        q.submit([&](sycl::handler& h){
            sycl::local_accessor<float,1> slm((size_t)KS*8*16,h);
            h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)(TR/8)*KS,N),sycl::range<2>(KS,16)),
                [=](sycl::nd_item<2> it)[[sycl::reqd_sub_group_size(16)]]{
                    int m0=(int)it.get_group(0)*8, s=(int)it.get_local_id(0), lane=(int)it.get_local_id(1);
                    int n=(int)it.get_group(1)*16+lane; int e=m0/M;             // expert for this tile
                    const uint8_t* wpe=wp+(size_t)e*N*(K/2); const uint8_t* wse=ws+(size_t)e*(K/16)*N;
                    v8f c={0,0,0,0,0,0,0,0};
                    for(int kt=s;kt<K/16;kt+=KS){ int k0=kt*16; v8i b=dqb(wpe,wse,n,k0,K,N);
                        v8s a; for(int m=0;m<8;++m)a[m]=(short)A[(size_t)(m0+m)*K+k0+lane];
                        c=__spirv_SubgroupMatrixMultiplyAccumulateINTEL(16,a,b,c,kBF16Ops); }
                    for(int m=0;m<8;++m) slm[(size_t)(s*8+m)*16+lane]=c[m];
                    it.barrier(sycl::access::fence_space::local_space);
                    if(s==0)for(int m=0;m<8;++m){ float sum=0; for(int ss=0;ss<KS;++ss)sum+=slm[(size_t)(ss*8+m)*16+lane];
                        C[(size_t)(m0+m)*N+lane]=f2bf(sum); } });
        });
    };

    auto bench=[&](const char* name, auto fn){
        fn(); q.wait();   // warmup
        auto t0=std::chrono::steady_clock::now();
        for(int i=0;i<iters;++i) fn();
        q.wait();
        double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        double gbps=wbytes*iters/sec/1e9;
        std::printf("%-14s %.3f ms/iter   %.1f GB/s   (%.1f%% of 536)\n",
                    name, sec/iters*1e3, gbps, 100.0*gbps/536.0);
    };
    // ---- (c) grouped + coalesced weight layout ----
    auto coal_once=[&]{
        int KS=ksplit(TR,N);
        q.submit([&](sycl::handler& h){
            sycl::local_accessor<float,1> slm((size_t)KS*8*16,h);
            h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)(TR/8)*KS,N),sycl::range<2>(KS,16)),
                [=](sycl::nd_item<2> it)[[sycl::reqd_sub_group_size(16)]]{
                    int m0=(int)it.get_group(0)*8, s=(int)it.get_local_id(0), lane=(int)it.get_local_id(1);
                    int n=(int)it.get_group(1)*16+lane; int e=m0/M;
                    const uint8_t* wce=wc+(size_t)e*N*(K/2); const uint8_t* wse=ws+(size_t)e*(K/16)*N;
                    v8f c={0,0,0,0,0,0,0,0};
                    for(int kt=s;kt<K/16;kt+=KS){ int k0=kt*16; v8i b=dqb_coal(wce,wse,n,lane,k0,K,N);
                        v8s a; for(int m=0;m<8;++m)a[m]=(short)A[(size_t)(m0+m)*K+k0+lane];
                        c=__spirv_SubgroupMatrixMultiplyAccumulateINTEL(16,a,b,c,kBF16Ops); }
                    for(int m=0;m<8;++m) slm[(size_t)(s*8+m)*16+lane]=c[m];
                    it.barrier(sycl::access::fence_space::local_space);
                    if(s==0)for(int m=0;m<8;++m){ float sum=0; for(int ss=0;ss<KS;++ss)sum+=slm[(size_t)(ss*8+m)*16+lane];
                        C[(size_t)(m0+m)*N+lane]=f2bf(sum); } });
        });
    };

    // ---- (d) coalesced but NO dequant math (same loads, raw bytes -> B) ----
    // Isolates: is the kernel memory-bound (this ~= coal) or dequant-ALU-bound
    // (this >> coal)?
    auto nodq_once=[&]{
        int KS=ksplit(TR,N);
        q.submit([&](sycl::handler& h){
            sycl::local_accessor<float,1> slm((size_t)KS*8*16,h);
            h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)(TR/8)*KS,N),sycl::range<2>(KS,16)),
                [=](sycl::nd_item<2> it)[[sycl::reqd_sub_group_size(16)]]{
                    int m0=(int)it.get_group(0)*8, s=(int)it.get_local_id(0), lane=(int)it.get_local_id(1);
                    int e=m0/M; const uint8_t* wce=wc+(size_t)e*N*(K/2);
                    v8f c={0,0,0,0,0,0,0,0};
                    for(int kt=s;kt<K/16;kt+=KS){ int k0=kt*16;
                        const uint8_t* row=wce+(size_t)(((int)it.get_group(1)*16+lane)/16)*(K/16)*128+(size_t)kt*128+lane*8;
                        v8i b; for(int j=0;j<8;++j) b[j]=row[j];   // raw, no dequant
                        v8s a; for(int m=0;m<8;++m)a[m]=(short)A[(size_t)(m0+m)*K+k0+lane];
                        c=__spirv_SubgroupMatrixMultiplyAccumulateINTEL(16,a,b,c,kBF16Ops); }
                    for(int m=0;m<8;++m) slm[(size_t)(s*8+m)*16+lane]=c[m];
                    it.barrier(sycl::access::fence_space::local_space);
                    if(s==0)for(int m=0;m<8;++m){ float sum=0; for(int ss=0;ss<KS;++ss)sum+=slm[(size_t)(ss*8+m)*16+lane];
                        C[(size_t)(m0+m)*N+lane]=f2bf(sum); } });
        });
    };

    // ---- (e) coalesced + LUT dequant: precomputed (scale_byte,nibble) -> bf16
    // table (256*16 = 8 KB, L1/constant resident). Replaces e2m1/e4m3/f2bf math
    // with one table load per nibble. ----
    uint16_t* lut=sycl::malloc_device<uint16_t>(256*16,q);
    { std::vector<uint16_t> h(256*16);
      for(int sb=0;sb<256;++sb) for(int nb=0;nb<16;++nb)
          h[sb*16+nb]=f2bf(e2m1_fast((uint8_t)nb)*e4m3_fast((uint8_t)sb));
      q.memcpy(lut,h.data(),256*16*sizeof(uint16_t)).wait(); }
    auto lut_once=[&]{
        int KS=ksplit(TR,N);
        q.submit([&](sycl::handler& h){
            sycl::local_accessor<float,1> slm((size_t)KS*8*16,h);
            h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)(TR/8)*KS,N),sycl::range<2>(KS,16)),
                [=](sycl::nd_item<2> it)[[sycl::reqd_sub_group_size(16)]]{
                    int m0=(int)it.get_group(0)*8, s=(int)it.get_local_id(0), lane=(int)it.get_local_id(1);
                    int n=(int)it.get_group(1)*16+lane; int e=m0/M;
                    const uint8_t* wce=wc+(size_t)e*N*(K/2); const uint8_t* wse=ws+(size_t)e*(K/16)*N;
                    v8f c={0,0,0,0,0,0,0,0};
                    for(int kt=s;kt<K/16;kt+=KS){ int k0=kt*16;
                        const uint16_t* lrow=lut+(size_t)wse[(size_t)kt*N+n]*16;
                        const uint8_t* row=wce+(size_t)(n/16)*(K/16)*128+(size_t)kt*128+lane*8;
                        v8i b; for(int j=0;j<8;++j){ uint8_t by=row[j];
                            b[j]=(int)((uint32_t)lrow[by&0xf]|((uint32_t)lrow[by>>4]<<16)); }
                        v8s a; for(int m=0;m<8;++m)a[m]=(short)A[(size_t)(m0+m)*K+k0+lane];
                        c=__spirv_SubgroupMatrixMultiplyAccumulateINTEL(16,a,b,c,kBF16Ops); }
                    for(int m=0;m<8;++m) slm[(size_t)(s*8+m)*16+lane]=c[m];
                    it.barrier(sycl::access::fence_space::local_space);
                    if(s==0)for(int m=0;m<8;++m){ float sum=0; for(int ss=0;ss<KS;++ss)sum+=slm[(size_t)(ss*8+m)*16+lane];
                        C[(size_t)(m0+m)*N+lane]=f2bf(sum); } });
        });
    };

    bench("per-expert", loop_once);
    bench("grouped", grouped_once);
    bench("grouped+coal", coal_once);
    bench("coal+nodequant", nodq_once);
    bench("coal+lut", lut_once);
    sycl::free(lut,q);
    std::printf("weight bytes/iter = %.1f MB\n", wbytes/1e6);
    sycl::free(wp,q);sycl::free(ws,q);sycl::free(A,q);sycl::free(C,q);sycl::free(wc,q);
    return 0;
}
