#pragma once
#include <cstdlib>
#include <cstring>
#include "grouped_moe.hpp"

namespace flint::flash_decode_twopass {
inline bool adaptive(){const char* p=std::getenv("ARCAINE_FLINT_FLASH_DECODE_TWO_PASS");return p&&!std::strcmp(p,"auto");}
inline bool enabled(){const char* p=std::getenv("ARCAINE_FLINT_FLASH_DECODE_TWO_PASS");return p&&(!std::strcmp(p,"1")||!std::strcmp(p,"auto"));}
// Scores are temporary FP32 GPU data. Separate QK and PV kernels keep each
// XMX accumulator set small; PV groups each own 64 output channels.
// Return the first event so profiling can span both kernels up to the merge.
template<bool PackedV=false>
inline sycl::event partials(sycl::queue& q,const bf16* query,const bf16* kc,const bf16* vc,
                     const bf16* sink,float* partial,float* stats,float* scores,const uint32_t* packed_v,int kv_heads,
                     int cap,int minimum_parts,int max_parts,const int* position){
    auto first_event=q.parallel_for(sycl::nd_range<1>(size_t(kv_heads)*max_parts*16,16),
        [=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(16)]]{
            const int work=int(it.get_group(0)),part=work%max_parts,kv=work/max_parts;
            const int lane=int(it.get_local_id(0)),count=*position+1;
            const int parts=sycl::max(minimum_parts,count<=1024?1:count<=4096?16:count<=16384?32:64);
            if(part>=parts)return;
            const int raw_partition=(count+parts-1)/parts;
            const int partition=PackedV?((raw_partition+15)/16)*16:raw_partition,begin=part*partition;
            const int nkeys=sycl::max(0,sycl::min(partition,count-begin));
            const auto sg=it.get_sub_group();
            flint_moe_v8s queries[8];float hi[8];
#pragma unroll
            for(int k=0;k<8;++k){
#pragma unroll
                for(int m=0;m<8;++m)queries[k][m]=sycl::bit_cast<short>(query[size_t(kv*8+m)*128+k*16+lane]);
            }
#pragma unroll
            for(int m=0;m<8;++m)hi[m]=part?-3.402823466e+38f:float(sink[kv*8+m]);
            for(int first=0;first<nkeys;first+=16){
                const int key=first+lane,absolute=begin+key;const bool valid=key<nkeys;
                flint_moe_v8f score{};
#pragma unroll
                for(int k=0;k<8;++k){
                    sycl::vec<uint32_t,8> packed{};
                    if(valid)packed.load(0,reinterpret_cast<const uint32_t*>(kc+(size_t(kv)*cap+absolute)*128+k*16));
                    flint_moe_v8i b;
#pragma unroll
                    for(int j=0;j<8;++j)b[j]=int(packed[j]);
                    score=__spirv_SubgroupMatrixMultiplyAccumulateINTEL(16,queries[k],b,score,0x3000);
                }
#pragma unroll
                for(int m=0;m<8;++m){
                    const float value=valid?score[m]*0.08838834764831845f:-3.402823466e+38f;
                    if(valid)scores[size_t(kv*8+m)*cap+absolute]=value;
                    hi[m]=sycl::fmax(hi[m],sycl::reduce_over_group(sg,value,sycl::maximum<>()));
                }
            }
            if(!lane){
#pragma unroll
                for(int m=0;m<8;++m)stats[(size_t(kv*8+m)*max_parts+part)*2]=hi[m];
            }
        });
    q.parallel_for(sycl::nd_range<1>(size_t(kv_heads)*max_parts*2*16,16),
        [=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(16)]]{
            const int work=int(it.get_group(0)),tile=work%2,part=(work/2)%max_parts,kv=work/(2*max_parts);
            const int lane=int(it.get_local_id(0)),count=*position+1;
            const int parts=sycl::max(minimum_parts,count<=1024?1:count<=4096?16:count<=16384?32:64);
            if(part>=parts)return;
            const int raw_partition=(count+parts-1)/parts;
            const int partition=PackedV?((raw_partition+15)/16)*16:raw_partition,begin=part*partition;
            const int nkeys=sycl::max(0,sycl::min(partition,count-begin));
            const auto sg=it.get_sub_group();
            flint_moe_v8f accum[4];float hi[8],z[8];
#pragma unroll
            for(int n=0;n<4;++n)accum[n]=flint_moe_v8f{};
#pragma unroll
            for(int m=0;m<8;++m){
                hi[m]=stats[(size_t(kv*8+m)*max_parts+part)*2];
                z[m]=part?0.f:sycl::native::exp2((float(sink[kv*8+m])-hi[m])*1.4426950408889634f);
            }
            for(int first=0;first<nkeys;first+=16){
                const int key=first+lane,absolute=begin+key;const bool valid=key<nkeys;
                flint_moe_v8s p_hi,p_lo;
#pragma unroll
                for(int m=0;m<8;++m){
                    const float value=valid?scores[size_t(kv*8+m)*cap+absolute]:0.f;
                    const float probability=valid?sycl::native::exp2((value-hi[m])*1.4426950408889634f):0.f;
                    z[m]+=sycl::reduce_over_group(sg,probability,sycl::plus<>());
                    const bf16 high=bf16(probability);
                    p_hi[m]=sycl::bit_cast<short>(high);
                    p_lo[m]=sycl::bit_cast<short>(bf16(probability-float(high)));
                }
#pragma unroll
                for(int n=0;n<4;++n){
                    flint_moe_v8i b;
#pragma unroll
                    for(int j=0;j<8;++j){
                        if constexpr(!PackedV){
                            const int key0=first+2*j,key1=key0+1,d=tile*64+n*16+lane;
                            const uint16_t lo=key0<nkeys?sycl::bit_cast<uint16_t>(vc[(size_t(kv)*cap+begin+key0)*128+d]):0;
                            const uint16_t high=key1<nkeys?sycl::bit_cast<uint16_t>(vc[(size_t(kv)*cap+begin+key1)*128+d]):0;
                            b[j]=int(uint32_t(lo)|(uint32_t(high)<<16));
                        }
                    }
                    if constexpr(PackedV){
                        const int blocks=(cap+15)/16;
                        const size_t index=((((size_t(kv)*blocks+(begin+first)/16)*8+tile*4+n)*16+lane)*8);
                        sycl::vec<uint32_t,8> packed;
                        packed.load(0,packed_v+index);
#pragma unroll
                        for(int j=0;j<8;++j){
                            const uint32_t mask=(first+2*j<nkeys?0xffffu:0u)|(first+2*j+1<nkeys?0xffff0000u:0u);
                            b[j]=int(packed[j]&mask);
                        }
                    }
                    accum[n]=__spirv_SubgroupMatrixMultiplyAccumulateINTEL(16,p_hi,b,accum[n],0x3000);
                    accum[n]=__spirv_SubgroupMatrixMultiplyAccumulateINTEL(16,p_lo,b,accum[n],0x3000);
                }
            }
#pragma unroll
            for(int m=0;m<8;++m){
                const size_t row=size_t(kv*8+m)*max_parts+part;
#pragma unroll
                for(int n=0;n<4;++n)partial[row*128+tile*64+n*16+lane]=accum[n][m];
                if(!lane&&!tile)stats[row*2+1]=z[m];
            }
        });
    return first_event;
}
}
