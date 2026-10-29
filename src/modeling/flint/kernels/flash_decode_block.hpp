#pragma once
#include "../weights.hpp"

namespace flint::flash_decode_block {
// Eight GQA heads share one contiguous KV tile. Each subgroup computes 32
// independent QK scores, then updates its FP32 softmax state once per tile.
template<int KeyStride=32,bool ShuffleQuery=false>
inline sycl::event partials(sycl::queue& q,const bf16* query,const bf16* kc,const bf16* vc,
                     const bf16* sink,float* partial,float* stats,int kv_heads,
                     bool global,int cap,int minimum_parts,int max_parts,const int* position){
    return q.submit([&](sycl::handler& h){
        sycl::local_accessor<bf16> keys(KeyStride*128,h),values(32*128,h);
        sycl::local_accessor<float> queries(ShuffleQuery?1:8*128,h);
        h.parallel_for(sycl::nd_range<1>(size_t(kv_heads)*max_parts*256,256),
            [=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(32)]]{
                const int work=int(it.get_group(0)),part=work%max_parts,kv=work/max_parts;
                const int tid=int(it.get_local_id(0)),lane=tid%32,head_group=tid/32;
                const int head=kv*8+head_group,pos=*position;
                const int count=global?pos+1:sycl::min(pos+1,cap);
                const int parts=sycl::max(minimum_parts,
                    count<=1024?1:count<=4096?16:count<=16384?32:64);
                if(part>=parts)return;
                const int partition=(count+parts-1)/parts;
                const int start=pos+1-count+part*partition;
                const int nkeys=sycl::max(0,sycl::min(partition,count-part*partition));
                float qv[4]={};
                if constexpr(ShuffleQuery){
#pragma unroll
                    for(int j=0;j<4;++j)qv[j]=float(query[size_t(head)*128+lane*4+j])*0.08838834764831845f;
                }else for(int i=tid;i<8*128;i+=256)
                    queries[i]=float(query[size_t(kv)*8*128+i])*0.08838834764831845f;
                it.barrier(sycl::access::fence_space::local_space);
                const auto sg=it.get_sub_group();
                float hi=part==0?float(sink[head]):-3.402823466e+38f;
                float z=part==0?1.f:0.f,acc[4]={};
                for(int first=0;first<nkeys;first+=32){
                    const int length=sycl::min(32,nkeys-first);
                    for(int i=tid;i<32*128;i+=256){
                        const int key=i/128,d=i%128;
                        if(key<length){
                            const int absolute=start+first+key;
                            const int slot=global?absolute:absolute%cap;
                            const size_t index=(size_t(kv)*cap+slot)*128+d;
                            keys[d*KeyStride+key]=kc[index];values[i]=vc[index];
                        }else{keys[d*KeyStride+key]=bf16(0.f);values[i]=bf16(0.f);}
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    float dot[4]={};
                    for(int d=0;d<128;d+=4){
#pragma unroll
                        for(int j=0;j<4;++j){
                            float qvalue;
                            if constexpr(ShuffleQuery)qvalue=sycl::group_broadcast(sg,qv[j],d/4);
                            else qvalue=queries[head_group*128+d+j];
                            dot[j]=sycl::fma(qvalue,float(keys[(d+j)*KeyStride+lane]),dot[j]);
                        }
                    }
                    const float score=lane<length?(dot[0]+dot[1])+(dot[2]+dot[3]):-3.402823466e+38f;
                    const float next=sycl::fmax(hi,sycl::reduce_over_group(sg,score,sycl::maximum<>()));
                    const float old_weight=sycl::exp(hi-next);
                    const float probability=lane<length?sycl::exp(score-next):0.f;
                    z=z*old_weight+sycl::reduce_over_group(sg,probability,sycl::plus<>());
#pragma unroll
                    for(int d=0;d<4;++d)acc[d]*=old_weight;
                    for(int key=0;key<length;++key){
                        const float weight=sycl::group_broadcast(sg,probability,key);
#pragma unroll
                        for(int d=0;d<4;++d)
                            acc[d]=sycl::fma(weight,float(values[key*128+lane*4+d]),acc[d]);
                    }
                    hi=next;
                    it.barrier(sycl::access::fence_space::local_space);
                }
                const size_t row=size_t(head)*max_parts+part;
#pragma unroll
                for(int d=0;d<4;++d)partial[row*128+lane*4+d]=acc[d];
                if(!lane){stats[row*2]=hi;stats[row*2+1]=z;}
            });
    });
}
}
