#pragma once
#include <cstdlib>
#include "weights.hpp"

namespace flint::decode_router {
inline bool enabled(){
    const char* p=std::getenv("ARCAINE_FLINT_DECODE_ROUTER");
    return p&&p[0]=='1'&&p[1]=='\0';
}
inline bool subgroup_topk_enabled(){
    const char* p=std::getenv("ARCAINE_FLINT_DECODE_ROUTER_SG_TOPK");
    return p&&p[0]=='1'&&p[1]=='\0';
}
inline void launch(sycl::queue& q,const bf16* w,const bf16* x,int E,int H,int top,
                   int* ids,float* weights,float* scratch){
    q.parallel_for(sycl::nd_range<1>(size_t((E+7)/8)*256,256),
        [=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(32)]]{
            const int row=int(it.get_group(0))*8+int(it.get_local_id(0))/32;
            if(row>=E)return;
            const auto sg=it.get_sub_group();const int lane=int(it.get_local_id(0))%32;
            float a=0.f;
            for(int i=lane;i<H;i+=32)a+=float(w[size_t(row)*H+i])*float(x[i]);
            a=sycl::reduce_over_group(sg,a,sycl::plus<>());
            if(!lane){a=float(bf16(a));scratch[row]=1.f/(1.f+sycl::exp(-a));}
        });
    if(subgroup_topk_enabled()){
        q.parallel_for(sycl::nd_range<1>(32,32),
            [=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(32)]]{
                const int lane=int(it.get_local_id(0));const auto sg=it.get_sub_group();
                float values[8];
#pragma unroll
                for(int j=0;j<8;++j)values[j]=lane+32*j<E?scratch[lane+32*j]:-3.402823466e+38f;
                float sum=1e-20f;
                for(int rank=0;rank<top;++rank){
                    float best=-3.402823466e+38f;int id=-1;
#pragma unroll
                    for(int j=0;j<8;++j){
                        const int candidate=lane+32*j;
                        if(candidate<E&&(values[j]>best||(values[j]==best&&candidate>id))){
                            best=values[j];id=candidate;
                        }
                    }
                    const float global_best=sycl::reduce_over_group(sg,best,sycl::maximum<>());
                    const int winner=sycl::reduce_over_group(sg,best==global_best?id:-1,sycl::maximum<>());
                    if(!lane){ids[rank]=winner;weights[rank]=global_best;}
                    sum+=global_best;
                    if(lane==winner%32)values[winner/32]=-3.402823466e+38f;
                }
                if(!lane)for(int rank=0;rank<top;++rank)weights[rank]/=sum;
            });
        return;
    }
    q.submit([&](sycl::handler& h){
        sycl::local_accessor<float> values(256,h),chosen(16,h);
        sycl::local_accessor<int> indices(256,h);
        h.parallel_for(sycl::nd_range<1>(256,256),[=](sycl::nd_item<1> it){
            const int tid=int(it.get_local_id(0));
            float score=tid<E?scratch[tid]:-3.402823466e+38f;
            for(int rank=0;rank<top;++rank){
                values[tid]=score;indices[tid]=tid;
                it.barrier(sycl::access::fence_space::local_space);
                for(int stride=128;stride;stride>>=1){
                    if(tid<stride&&(values[tid+stride]>values[tid]||
                       (values[tid+stride]==values[tid]&&indices[tid+stride]>indices[tid]))){
                        values[tid]=values[tid+stride];indices[tid]=indices[tid+stride];
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
                const int winner=indices[0];
                if(!tid){chosen[rank]=values[0];ids[rank]=winner;}
                if(tid==winner)score=-3.402823466e+38f;
                it.barrier(sycl::access::fence_space::local_space);
            }
            if(!tid){
                float sum=1e-20f;for(int rank=0;rank<top;++rank)sum+=chosen[rank];
                for(int rank=0;rank<top;++rank)weights[rank]=chosen[rank]/sum;
            }
        });
    });
}
}
