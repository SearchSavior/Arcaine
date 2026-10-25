#pragma once
#include "grouped_moe.hpp"

namespace flint::flash_decode_xmx {
// BF16 QK on XMX. P is split into BF16 high/residual components for P*V,
// retaining near-FP32 probability precision with FP32 output accumulation.
inline void partials(sycl::queue& q,const bf16* query,const bf16* kc,const bf16* vc,
                     const bf16* sink,float* partial,float* stats,int kv_heads,
                     bool global,int cap,int minimum_parts,int max_parts,const int* position){
    q.parallel_for(sycl::nd_range<1>(size_t(kv_heads)*max_parts*16,16),
        [=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(16)]]{
            const int work=int(it.get_group(0)),part=work%max_parts,kv=work/max_parts;
            const int lane=int(it.get_local_id(0)),pos=*position;
            const int count=global?pos+1:sycl::min(pos+1,cap);
            const int parts=sycl::max(minimum_parts,
                count<=1024?1:count<=4096?16:count<=16384?32:64);
            if(part>=parts)return;
            const int start=pos+1-count;
            const int nkeys=count>part?(count-1-part)/parts+1:0;
            const auto sg=it.get_sub_group();
            flint_moe_v8f accum[8];
            float hi[8],z[8];
#pragma unroll
            for(int n=0;n<8;++n)accum[n]=flint_moe_v8f{};
#pragma unroll
            for(int m=0;m<8;++m){hi[m]=part? -3.402823466e+38f:float(sink[kv*8+m]);z[m]=part?0.f:1.f;}
            for(int first=0;first<nkeys;first+=16){
                const int key=first+lane;
                const bool valid=key<nkeys;
                const int absolute=start+part+key*parts;
                const int slot=global?absolute:absolute%cap;
                flint_moe_v8f score{};
#pragma unroll
                for(int k0=0;k0<128;k0+=16){
                    flint_moe_v8s a;flint_moe_v8i b;
#pragma unroll
                    for(int m=0;m<8;++m)
                        a[m]=sycl::bit_cast<short>(query[size_t(kv*8+m)*128+k0+lane]);
#pragma unroll
                    for(int j=0;j<8;++j){
                        const size_t index=(size_t(kv)*cap+slot)*128+k0+2*j;
                        const uint16_t lo=valid?sycl::bit_cast<uint16_t>(kc[index]):0;
                        const uint16_t high=valid?sycl::bit_cast<uint16_t>(kc[index+1]):0;
                        b[j]=int(uint32_t(lo)|(uint32_t(high)<<16));
                    }
                    score=__spirv_SubgroupMatrixMultiplyAccumulateINTEL(16,a,b,score,0x3000);
                }
                flint_moe_v8s p_hi,p_lo;
                float old[8];
#pragma unroll
                for(int m=0;m<8;++m){
                    const float value=valid?score[m]*0.08838834764831845f:-3.402823466e+38f;
                    const float tile_max=sycl::reduce_over_group(sg,value,sycl::maximum<>());
                    const float next=sycl::fmax(hi[m],tile_max);
                    old[m]=sycl::exp(hi[m]-next);
                    const float probability=valid?sycl::exp(value-next):0.f;
                    z[m]=z[m]*old[m]+sycl::reduce_over_group(sg,probability,sycl::plus<>());
                    hi[m]=next;
                    const bf16 high=bf16(probability);
                    p_hi[m]=sycl::bit_cast<short>(high);
                    p_lo[m]=sycl::bit_cast<short>(bf16(probability-float(high)));
                }
#pragma unroll
                for(int n=0;n<8;++n){
#pragma unroll
                    for(int m=0;m<8;++m)accum[n][m]*=old[m];
                    flint_moe_v8i b;
#pragma unroll
                    for(int j=0;j<8;++j){
                        const int key0=first+2*j,key1=key0+1;
                        const int pos0=start+part+key0*parts,pos1=start+part+key1*parts;
                        const int slot0=global?pos0:pos0%cap,slot1=global?pos1:pos1%cap;
                        const size_t index0=(size_t(kv)*cap+slot0)*128+n*16+lane;
                        const size_t index1=(size_t(kv)*cap+slot1)*128+n*16+lane;
                        const uint16_t lo=key0<nkeys?sycl::bit_cast<uint16_t>(vc[index0]):0;
                        const uint16_t high=key1<nkeys?sycl::bit_cast<uint16_t>(vc[index1]):0;
                        b[j]=int(uint32_t(lo)|(uint32_t(high)<<16));
                    }
                    accum[n]=__spirv_SubgroupMatrixMultiplyAccumulateINTEL(16,p_hi,b,accum[n],0x3000);
                    accum[n]=__spirv_SubgroupMatrixMultiplyAccumulateINTEL(16,p_lo,b,accum[n],0x3000);
                }
            }
#pragma unroll
            for(int m=0;m<8;++m){
                const size_t row=size_t(kv*8+m)*max_parts+part;
#pragma unroll
                for(int n=0;n<8;++n)partial[row*128+n*16+lane]=accum[n][m];
                if(!lane){stats[row*2]=hi[m];stats[row*2+1]=z[m];}
            }
        });
}
}
