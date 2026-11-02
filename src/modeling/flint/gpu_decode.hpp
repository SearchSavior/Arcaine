#pragma once
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>
#include "weights.hpp"

namespace flint::gpu_decode {
constexpr int kBlock = 8;
constexpr int kMaxTopK = 64;
inline bool enabled() {
    const char* p = std::getenv("ARCAINE_FLINT_GPU_DECODE");
    return p && p[0] == '1' && p[1] == '\0';
}
struct State {
    int position = 0, token = 0, active = 1, emitted = 0;
    uint64_t rng = 1;
    float temperature = 0.f, top_p = 1.f;
    int top_k = 1;
};
inline void emit(State* state, int* out, int token, const int* eos, int count) {
    *out = token; state->token = token; ++state->emitted;
    for (int i = 0; i < count; ++i) if (token == eos[i]) state->active = 0;
}
inline void greedy(sycl::queue& q, const float* logits, int vocab, State* state,
                   int* out, const int* eos, int count) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float> values(256, h);
        sycl::local_accessor<int> indices(256, h);
        h.parallel_for(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> it) {
            const int tid = int(it.get_local_id(0));
            if (!state->active) { if (tid == 0) *out = -1; return; }
            float best = -std::numeric_limits<float>::infinity();
            int index = vocab;
            for (int i = tid; i < vocab; i += 256)
                if (logits[i] > best || (logits[i] == best && i < index)) {
                    best = logits[i]; index = i;
                }
            values[tid] = best; indices[tid] = index;
            it.barrier(sycl::access::fence_space::local_space);
            for (int stride = 128; stride; stride >>= 1) {
                if (tid < stride && (values[tid + stride] > values[tid] ||
                    (values[tid + stride] == values[tid] && indices[tid + stride] < indices[tid]))) {
                    values[tid] = values[tid + stride]; indices[tid] = indices[tid + stride];
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            if (!tid) emit(state, out, indices[0], eos, count);
        });
    });
}
// Device top-k/nucleus selection with native Flint tie order and xorshift RNG.
// Correctness-first sparse insertion scan; K<=64, no host logit download.
inline void sample(sycl::queue& q, const float* logits, int vocab, State* state,
                   int* out, const int* eos, int count) {
    q.single_task([=] {
        if (!state->active) { *out = -1; return; }
        const int k = state->top_k;
        float values[kMaxTopK], p[kMaxTopK]; int indices[kMaxTopK];
        for (int i = 0; i < k; ++i) {
            values[i] = -std::numeric_limits<float>::infinity(); indices[i] = vocab;
        }
        for (int id = 0; id < vocab; ++id) {
            const float v = logits[id];
            if (v < values[k-1] || (v == values[k-1] && id >= indices[k-1])) continue;
            int at = k-1;
            while (at && (v > values[at-1] || (v == values[at-1] && id < indices[at-1]))) {
                values[at] = values[at-1]; indices[at] = indices[at-1]; --at;
            }
            values[at] = v; indices[at] = id;
        }
        float sum = 0.f;
        for (int i = 0; i < k; ++i) {
            p[i] = sycl::exp((values[i] - values[0]) / state->temperature); sum += p[i];
        }
        for (int i = 0; i < k; ++i) p[i] /= sum;
        int keep = k;
        if (state->top_p > 0.f && state->top_p < 1.f) {
            float c = 0.f;
            for (int i = 0; i < k; ++i) {
                c += p[i]; if (c >= state->top_p) { keep = i+1; break; }
            }
            float kept = 0.f;
            for (int i = 0; i < keep; ++i) kept += p[i];
            for (int i = 0; i < keep; ++i) p[i] /= kept;
        }
        uint64_t rng = state->rng;
        rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27; state->rng = rng;
        const float u = float((rng * 2685821657736338717ull) >> 11) *
                        (1.f / 9007199254740992.f);
        float c = 0.f; int selected = indices[keep-1];
        for (int i = 0; i < keep; ++i) { c += p[i]; if (u < c) { selected = indices[i]; break; } }
        emit(state, out, selected, eos, count);
    });
}
inline bool parallel_sample_enabled(){
    const char* p=std::getenv("ARCAINE_FLINT_DECODE_PARALLEL_SAMPLE");
    return p&&p[0]=='1'&&p[1]=='\0';
}
// Partition the vocabulary over 256 threads. Each thread keeps its local
// top-20; a stable WG merge extracts the exact global top-k before sampling.
template<bool SubgroupMerge, bool PartialInput=false, bool OutputPartial=false>
inline void parallel_sample20_impl(sycl::queue& q,const float* logits,int vocab,State* state,
                              int* out,const int* eos,int eos_count,
                              const int* input_ids=nullptr,float* partial_values=nullptr,
                              int* partial_ids=nullptr,int groups=1){
    q.submit([&](sycl::handler& h){
        sycl::local_accessor<float> candidates(SubgroupMerge?8:256,h),selected(20,h);
        sycl::local_accessor<int> candidate_ids(SubgroupMerge?8:256,h),selected_ids(20,h);
        h.parallel_for(sycl::nd_range<1>(size_t(OutputPartial?groups:1)*256,256),[=](sycl::nd_item<1> it)[[sycl::reqd_sub_group_size(32)]]{
            const int tid=int(it.get_local_id(0));
            if(!state->active){if constexpr(!OutputPartial){if(!tid)*out=-1;}return;}
            const int k=state->top_k;
            {
                float values[20];int indices[20];
                for(int i=0;i<k;++i){values[i]=-std::numeric_limits<float>::infinity();indices[i]=std::numeric_limits<int>::max();}
                const int group=OutputPartial?int(it.get_group(0)):0;
                for(int source=group*256+tid;source<vocab;source+=(OutputPartial?groups:1)*256){
                    const int id=PartialInput?input_ids[source]:source;
                    const float v=logits[source];
                    if(v<values[k-1]||(v==values[k-1]&&id>=indices[k-1]))continue;
                    int at=k-1;
                    while(at&&(v>values[at-1]||(v==values[at-1]&&id<indices[at-1]))){
                        values[at]=values[at-1];indices[at]=indices[at-1];--at;
                    }
                    values[at]=v;indices[at]=id;
                }
                int cursor=0;
                for(int rank=0;rank<k;++rank){
                    const float candidate=cursor<k?values[cursor]:-std::numeric_limits<float>::infinity();
                    const int candidate_id=cursor<k?indices[cursor]:std::numeric_limits<int>::max();
                    if constexpr(SubgroupMerge){
                        const auto sg=it.get_sub_group();
                        const int lane=tid%32,group=tid/32;
                        const float best=sycl::reduce_over_group(sg,candidate,sycl::maximum<>());
                        const int winner=sycl::reduce_over_group(sg,candidate==best?candidate_id:std::numeric_limits<int>::max(),sycl::minimum<>());
                        if(!lane){candidates[group]=best;candidate_ids[group]=winner;}
                        it.barrier(sycl::access::fence_space::local_space);
                        if(group==0){
                            const float value=lane<8?candidates[lane]:-std::numeric_limits<float>::infinity();
                            const int id=lane<8?candidate_ids[lane]:std::numeric_limits<int>::max();
                            const float global_best=sycl::reduce_over_group(sg,value,sycl::maximum<>());
                            const int global_id=sycl::reduce_over_group(sg,value==global_best?id:std::numeric_limits<int>::max(),sycl::minimum<>());
                            if(!lane){selected[rank]=global_best;selected_ids[rank]=global_id;}
                        }
                    }else{
                        candidates[tid]=candidate;candidate_ids[tid]=candidate_id;
                        it.barrier(sycl::access::fence_space::local_space);
                        for(int stride=128;stride;stride>>=1){
                            if(tid<stride&&(candidates[tid+stride]>candidates[tid]||
                               (candidates[tid+stride]==candidates[tid]&&candidate_ids[tid+stride]<candidate_ids[tid]))){
                                candidates[tid]=candidates[tid+stride];candidate_ids[tid]=candidate_ids[tid+stride];
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                        if(!tid){selected[rank]=candidates[0];selected_ids[rank]=candidate_ids[0];}
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if(candidate_id==selected_ids[rank]&&candidate==selected[rank])++cursor;
                }
            }
            if constexpr(OutputPartial){
                if(!tid)for(int i=0;i<20;++i){
                    const int offset=int(it.get_group(0))*20+i;
                    partial_values[offset]=i<k?selected[i]:-std::numeric_limits<float>::infinity();
                    partial_ids[offset]=i<k?selected_ids[i]:std::numeric_limits<int>::max();
                }
                return;
            }
            if(!tid){
                float p[20],sum=0.f;
                for(int i=0;i<k;++i){p[i]=sycl::exp((selected[i]-selected[0])/state->temperature);sum+=p[i];}
                for(int i=0;i<k;++i)p[i]/=sum;
                int keep=k;
                if(state->top_p>0.f&&state->top_p<1.f){
                    float c=0.f;
                    for(int i=0;i<k;++i){c+=p[i];if(c>=state->top_p){keep=i+1;break;}}
                    float kept=0.f;for(int i=0;i<keep;++i)kept+=p[i];
                    for(int i=0;i<keep;++i)p[i]/=kept;
                }
                uint64_t rng=state->rng;rng^=rng>>12;rng^=rng<<25;rng^=rng>>27;state->rng=rng;
                const float u=float((rng*2685821657736338717ull)>>11)*(1.f/9007199254740992.f);
                float c=0.f;int token=selected_ids[keep-1];
                for(int i=0;i<keep;++i){c+=p[i];if(u<c){token=selected_ids[i];break;}}
                emit(state,out,token,eos,eos_count);
            }
        });
    });
}

inline bool subgroup_sample_merge_enabled(){
    const char* p=std::getenv("ARCAINE_FLINT_DECODE_SAMPLE_SG_MERGE");
    return p&&p[0]=='1'&&p[1]=='\0';
}
inline void parallel_sample20(sycl::queue& q,const float* logits,int vocab,State* state,
                              int* out,const int* eos,int eos_count){
    if(subgroup_sample_merge_enabled())
        parallel_sample20_impl<true>(q,logits,vocab,state,out,eos,eos_count);
    else parallel_sample20_impl<false>(q,logits,vocab,state,out,eos,eos_count);
}
inline int sample_groups(){
    const char* p=std::getenv("ARCAINE_FLINT_DECODE_SAMPLE_GROUPS");
    const int n=p?std::atoi(p):1;
    return n==8||n==16||n==32||n==64?n:1;
}
// A vocabulary partition contributes only its top-k to the final top-k.
// Both passes remain on device; only the final pass mutates RNG/EOS state.
inline void distributed_sample20(sycl::queue& q,const float* logits,int vocab,State* state,
                                 int* out,const int* eos,int eos_count,
                                 float* partial_values,int* partial_ids,int groups){
    parallel_sample20_impl<false,false,true>(q,logits,vocab,state,out,eos,eos_count,
                                            nullptr,partial_values,partial_ids,groups);
    parallel_sample20_impl<false,true,false>(q,partial_values,groups*20,state,out,eos,eos_count,
                                            partial_ids);
}
inline void embed(sycl::queue& q, const QTensor& t, const int* token, bf16* out) {
    q.parallel_for(sycl::range<1>(size_t(t.k)), [=](sycl::id<1> id) {
        const int n = *token, i = int(id[0]);
        if (t.format == WeightFormat::BF16) out[i] = t.dense[size_t(n) * t.k + i];
        else if (t.format == WeightFormat::AWQ) {
            const size_t param = size_t(i / 64) * t.n + n;
            const uint8_t byte = t.w[size_t(n) * (t.k / 2) + i / 2];
            const int code = (byte >> ((i & 1) * 4)) & 15;
            out[i] = bf16(float(code - int(t.awq_z[param])) * float(t.awq_s[param]));
        } else {
            const size_t param = size_t(n) * (t.k / 64) + i / 64;
            out[i] = bf16(float(t.s[param]) * float(t.w[size_t(n) * t.k + i]) + float(t.b[param]));
        }
    });
}
}
