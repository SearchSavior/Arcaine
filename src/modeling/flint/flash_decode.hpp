#pragma once
#include <cstdlib>
#include <cstring>
#include "weights.hpp"
#include "flash_decode_xmx.hpp"

namespace flint::flash_decode {
inline int tile() {
    const char* p=std::getenv("ARCAINE_FLINT_FLASH_DECODE");
    if(!p)return 0;
    if(!std::strcmp(p,"1")||!std::strcmp(p,"16"))return 16;
    if(!std::strcmp(p,"32"))return 32;
    if(!std::strcmp(p,"64"))return 64;
    return 0;
}
// One work-group serves all eight Q heads sharing a KV head. Cooperatively
// stage a cache tile once, then keep independent FP32 online-softmax states.
// Strided partitions retain the existing split-attention reduction order.
template<bool Native>
inline float softmax_exp(float x){
    if constexpr(Native)return sycl::native::exp2(x*1.4426950408889634f);
    else return sycl::exp(x);
}
template<int Tile,bool Native=false,bool SingleExp=false>
inline void partials(sycl::queue& q,const bf16* query,const bf16* kc,const bf16* vc,
                     const bf16* sink,float* partial,float* stats,int kv_heads,
                     bool global,int cap,int minimum_parts,int max_parts,const int* position) {
    q.submit([&](sycl::handler& h){
        sycl::local_accessor<bf16> keys(size_t(Tile)*128,h),values(size_t(Tile)*128,h);
        h.parallel_for(sycl::nd_range<1>(size_t(kv_heads)*max_parts*256,256),
            [=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(32)]]{
                const int work=int(it.get_group(0)),part=work%max_parts,kv=work/max_parts;
                const int tid=int(it.get_local_id(0)),lane=tid%32,head=kv*8+tid/32;
                const int pos=*position;
                const int count=global?pos+1:sycl::min(pos+1,cap);
                const int parts=sycl::max(minimum_parts,
                    count<=1024?1:count<=4096?16:count<=16384?32:64);
                if(part>=parts)return;
                const int start=pos+1-count;
                const int nkeys=count>part?(count-1-part)/parts+1:0;
                const auto sg=it.get_sub_group();
                float qv[4],acc[4]={};
#pragma unroll
                for(int d=0;d<4;++d)
                    qv[d]=float(query[size_t(head)*128+lane*4+d])*0.08838834764831845f;
                float hi=part==0?float(sink[head]):-3.402823466e+38f;
                float z=part==0?1.f:0.f;
                // A short ring is cheaper to share through the Xe-core
                // cache. Keep the eight GQA heads in one work-group, avoiding
                // SLM copies/barriers until the context benefits from tiling.
                if(count<=1024){
                    for(int absolute=start+part;absolute<=pos;absolute+=parts){
                        const int slot=global?absolute:absolute%cap;
                        const size_t index=(size_t(kv)*cap+slot)*128+lane*4;
                        float score=0.f;
#pragma unroll
                        for(int d=0;d<4;++d)score+=qv[d]*float(kc[index+d]);
                        score=sycl::reduce_over_group(sg,score,sycl::plus<>());
                        const float next=sycl::fmax(hi,score);
                        float old_weight,weight;
                        if constexpr(SingleExp){
                            const float scale=softmax_exp<Native>(-sycl::fabs(score-hi));
                            old_weight=score>hi?scale:1.f;
                            weight=score>hi?1.f:scale;
                        }else{
                            old_weight=softmax_exp<Native>(hi-next);
                            weight=softmax_exp<Native>(score-next);
                        }
                        z=z*old_weight+weight;
#pragma unroll
                        for(int d=0;d<4;++d)acc[d]=acc[d]*old_weight+weight*float(vc[index+d]);
                        hi=next;
                    }
                }else for(int first=0;first<nkeys;first+=Tile){
                    for(int i=tid;i<Tile*128;i+=256){
                        const int key=first+i/128;
                        if(key<nkeys){
                            const int absolute=start+part+key*parts;
                            const int slot=global?absolute:absolute%cap;
                            const size_t index=(size_t(kv)*cap+slot)*128+i%128;
                            keys[i]=kc[index];values[i]=vc[index];
                        }else{keys[i]=bf16(0.f);values[i]=bf16(0.f);}
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    const int length=sycl::min(Tile,nkeys-first);
                    for(int j=0;j<length;++j){
                        float score=0.f;
#pragma unroll
                        for(int d=0;d<4;++d)score+=qv[d]*float(keys[j*128+lane*4+d]);
                        score=sycl::reduce_over_group(sg,score,sycl::plus<>());
                        const float next=sycl::fmax(hi,score);
                        float old_weight,weight;
                        if constexpr(SingleExp){
                            const float scale=softmax_exp<Native>(-sycl::fabs(score-hi));
                            old_weight=score>hi?scale:1.f;
                            weight=score>hi?1.f:scale;
                        }else{
                            old_weight=softmax_exp<Native>(hi-next);
                            weight=softmax_exp<Native>(score-next);
                        }
                        z=z*old_weight+weight;
#pragma unroll
                        for(int d=0;d<4;++d)
                            acc[d]=acc[d]*old_weight+weight*float(values[j*128+lane*4+d]);
                        hi=next;
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
                const size_t row=size_t(head)*max_parts+part;
#pragma unroll
                for(int d=0;d<4;++d)partial[row*128+lane*4+d]=acc[d];
                if(!lane){stats[row*2]=hi;stats[row*2+1]=z;}
            });
    });
}
// Spread partition accumulation over eight subgroups, then combine in SLM.
inline bool merge(sycl::queue& q,const float* partial,const float* stats,bf16* output,
                  int heads,bool global,int cap,int minimum_parts,int max_parts,const int* position){
    const char* flag=std::getenv("ARCAINE_FLINT_FLASH_DECODE_PARALLEL_MERGE");
    if(!flag||flag[0]!=49||flag[1]!=0)return false;
    q.submit([&](sycl::handler& h){
        sycl::local_accessor<float> sums(8*128,h),normalizers(8,h);
        h.parallel_for(sycl::nd_range<1>(size_t(heads)*256,256),
            [=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(32)]]{
                const int head=int(it.get_group(0)),tid=int(it.get_local_id(0));
                const int lane=tid%32,group=tid/32,pos=*position;
                const int count=global?pos+1:sycl::min(pos+1,cap);
                const int parts=sycl::max(minimum_parts,
                    count<=1024?1:count<=4096?16:count<=16384?32:64);
                float hi=-3.402823466e+38f;
                for(int p=tid;p<parts;p+=256)
                    hi=sycl::fmax(hi,stats[(size_t(head)*max_parts+p)*2]);
                hi=sycl::reduce_over_group(it.get_group(),hi,sycl::maximum<>());
                float z=0.f,acc[4]={};
                for(int p=group;p<parts;p+=8){
                    const size_t row=size_t(head)*max_parts+p;
                    const float weight=sycl::exp(stats[row*2]-hi);
                    z+=weight*stats[row*2+1];
#pragma unroll
                    for(int d=0;d<4;++d)acc[d]+=weight*partial[row*128+lane*4+d];
                }
#pragma unroll
                for(int d=0;d<4;++d)sums[group*128+lane*4+d]=acc[d];
                if(!lane)normalizers[group]=z;
                it.barrier(sycl::access::fence_space::local_space);
                if(group==0){
                    float total=0.f,result[4]={};
#pragma unroll
                    for(int g=0;g<8;++g){
                        total+=normalizers[g];
#pragma unroll
                        for(int d=0;d<4;++d)result[d]+=sums[g*128+lane*4+d];
                    }
                    const float inv=total==0.f?0.f:1.f/total;
#pragma unroll
                    for(int d=0;d<4;++d)output[size_t(head)*128+lane*4+d]=bf16(result[d]*inv);
                }
            });
    });
    return true;
}
inline void launch(sycl::queue& q,const bf16* query,const bf16* kc,const bf16* vc,
                   const bf16* sink,float* partial,float* stats,int kv_heads,
                   bool global,int cap,int minimum_parts,int max_parts,const int* position) {
    const char* matrix=std::getenv("ARCAINE_FLINT_FLASH_DECODE_XMX");
    if(matrix&&matrix[0]=='1'&&matrix[1]=='\0'){
        flash_decode_xmx::partials(q,query,kc,vc,sink,partial,stats,kv_heads,
                                  global,cap,minimum_parts,max_parts,position);
        return;
    }
    const int size=tile();
    const char* single=std::getenv("ARCAINE_FLINT_FLASH_DECODE_SINGLE_EXP");
    if(single&&single[0]==49&&single[1]==0){
        if(size==16)partials<16,false,true>(q,query,kc,vc,sink,partial,stats,kv_heads,global,cap,minimum_parts,max_parts,position);
        else if(size==32)partials<32,false,true>(q,query,kc,vc,sink,partial,stats,kv_heads,global,cap,minimum_parts,max_parts,position);
        else partials<64,false,true>(q,query,kc,vc,sink,partial,stats,kv_heads,global,cap,minimum_parts,max_parts,position);
        return;
    }
    const char* fast=std::getenv("ARCAINE_FLINT_FLASH_DECODE_NATIVE_EXP");
    if(fast&&fast[0]==49&&fast[1]==0){
        if(size==16)partials<16,true>(q,query,kc,vc,sink,partial,stats,kv_heads,global,cap,minimum_parts,max_parts,position);
        else if(size==32)partials<32,true>(q,query,kc,vc,sink,partial,stats,kv_heads,global,cap,minimum_parts,max_parts,position);
        else partials<64,true>(q,query,kc,vc,sink,partial,stats,kv_heads,global,cap,minimum_parts,max_parts,position);
        return;
    }
    if(size==16)partials<16>(q,query,kc,vc,sink,partial,stats,kv_heads,global,cap,minimum_parts,max_parts,position);
    else if(size==32)partials<32>(q,query,kc,vc,sink,partial,stats,kv_heads,global,cap,minimum_parts,max_parts,position);
    else partials<64>(q,query,kc,vc,sink,partial,stats,kv_heads,global,cap,minimum_parts,max_parts,position);
}
}
