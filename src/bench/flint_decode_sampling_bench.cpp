#include "../common/bench/registry.hpp"
#include "../modeling/flint/gpu_decode.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace flint_sampling_bench {
using namespace flint;
struct Buffers {
    sycl::queue& q;std::vector<void*> all;
    explicit Buffers(sycl::queue& queue):q(queue){}
    ~Buffers(){try{q.wait();}catch(...){}for(auto* p:all)sycl::free(p,q);}
    template<class T>T* alloc(size_t n){auto* p=sycl::malloc_device<T>(n,q);if(!p)throw std::bad_alloc{};all.push_back(p);return p;}
};
uint32_t hash(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
int run(int argc,char** argv) {
    try {
        int vocab=150144,k=20,iters=100;bool check=false,greedy=false;
        for(int i=1;i<argc;++i){
            const std::string a=argv[i];
            if(a=="--check"){check=true;continue;}if(a=="--greedy"){greedy=true;continue;}
            if(a=="--help"){std::printf("flint-decode-sampling --vocab 150144 --top-k 20 --iters 100 [--greedy] [--check]\n"
                "Both ARCAINE_FLINT_DECODE_PARALLEL_SAMPLE=0/1 paths execute on GPU.\n");return 0;}
            if(++i>=argc)throw std::invalid_argument("missing value");
            size_t used=0;const int n=std::stoi(argv[i],&used);
            if(used!=std::strlen(argv[i])||n<=0)throw std::invalid_argument("invalid integer");
            if(a=="--vocab")vocab=n;else if(a=="--top-k")k=n;else if(a=="--iters")iters=n;
            else throw std::invalid_argument("unknown option");
        }
        if(vocab<64||vocab>200000||k<2||k>20)throw std::invalid_argument("require 64<=vocab<=200000,2<=top-k<=20");
        sycl::queue q(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
            [](sycl::exception_list errors){for(const auto& e:errors)std::rethrow_exception(e);},
            sycl::property::queue::in_order{});
        Buffers b(q);auto* logits=b.alloc<float>(vocab);auto* eos=b.alloc<int>(1);
        auto* state=b.alloc<gpu_decode::State>(1);auto* reference_state=b.alloc<gpu_decode::State>(1);
        auto* output=b.alloc<int>(16);auto* reference=b.alloc<int>(16);
        auto* partial_values=b.alloc<float>(64*20);auto* partial_ids=b.alloc<int>(64*20);
        const int groups=gpu_decode::sample_groups();
        auto parallel_sample=[&](int* out,const int* eos_ids,int count){
            if(groups>1)gpu_decode::distributed_sample20(q,logits,vocab,state,out,eos_ids,count,partial_values,partial_ids,groups);
            else gpu_decode::parallel_sample20(q,logits,vocab,state,out,eos_ids,count);
        };
        auto initialize=[&]{
            q.parallel_for(sycl::range<1>(size_t(vocab)),[=](sycl::id<1> i){
                logits[i]=float(bf16(float(int(hash(uint32_t(i[0])+17)%4097)-2048)/128.f));
            });
        };
        initialize();
        if(check){
            for(int top:{2,6,20})for(float p:{0.01f,0.95f,1.f}){
                gpu_decode::State a,e;
                a.top_k=e.top_k=top;a.temperature=e.temperature=1.f;a.top_p=e.top_p=p;
                a.rng=e.rng=0x9e3779b97f4a7c15ull;
                q.memcpy(state,&a,sizeof(a));q.memcpy(reference_state,&e,sizeof(e)).wait_and_throw();
                for(int i=0;i<16;++i){
                    gpu_decode::sample(q,logits,vocab,reference_state,reference+i,nullptr,0);
                    parallel_sample(output+i,nullptr,0);
                }
                int actual[16],expected[16];
                q.memcpy(actual,output,sizeof(actual));q.memcpy(expected,reference,sizeof(expected));
                q.memcpy(&a,state,sizeof(a));q.memcpy(&e,reference_state,sizeof(e)).wait_and_throw();
                bool ok=a.rng==e.rng&&a.emitted==e.emitted&&a.active==e.active;
                for(int i=0;i<16;++i)ok=ok&&actual[i]==expected[i];
                std::printf("sampling GPU-reference k=%d top_p=%g %s\n",top,p,ok?"PASS":"FAIL");
                if(!ok)return 1;
            }
            // Exact tie order and EOS: top-p keeps the lower-ID tied winner.
            q.fill(logits,-100.f,vocab);
            q.single_task([=]{logits[7]=5.f;logits[19]=5.f;*eos=7;});
            gpu_decode::State a;a.top_k=2;a.temperature=1.f;a.top_p=0.1f;a.rng=17;
            q.memcpy(state,&a,sizeof(a)).wait_and_throw();
            for(int i=0;i<4;++i)parallel_sample(output+i,eos,1);
            int ids[4];q.memcpy(ids,output,sizeof(ids));q.memcpy(&a,state,sizeof(a)).wait_and_throw();
            const bool eos_ok=ids[0]==7&&ids[1]==-1&&ids[2]==-1&&ids[3]==-1&&a.active==0&&a.emitted==1;
            std::printf("sampling stable ties/EOS %s\n",eos_ok?"PASS":"FAIL");if(!eos_ok)return 1;
            initialize();
        }
        gpu_decode::State a;a.top_k=k;a.temperature=1.f;a.top_p=0.95f;a.rng=0x9e3779b97f4a7c15ull;
        q.memcpy(state,&a,sizeof(a)).wait_and_throw();
        const bool parallel=gpu_decode::parallel_sample_enabled();
        auto once=[&]{
            if(greedy)gpu_decode::greedy(q,logits,vocab,state,output,nullptr,0);
            else if(parallel)parallel_sample(output,nullptr,0);
            else gpu_decode::sample(q,logits,vocab,state,output,nullptr,0);
        };
        once();q.wait_and_throw();
        const auto start=std::chrono::steady_clock::now();
        for(int i=0;i<iters;++i)once();
        q.wait_and_throw();
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/iters;
        std::printf("flint-decode-sampling mode=%s parallel=%d groups=%d vocab=%d k=%d iters=%d ms=%.4f\n",
                    greedy?"greedy":"topk",parallel,groups,vocab,k,iters,ms);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"flint-decode-sampling: %s\n",e.what());return 1;}
}
}
REGISTER_BENCH("flint-decode-sampling","GPU greedy/top-k/nucleus sampler with stable merge and device RNG",flint_sampling_bench::run)
