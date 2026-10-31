#include "../common/bench/registry.hpp"
#include "../modeling/flint/kernels/kernels.hpp"
#include "../modeling/flint/kernels/bf16_decode.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace flint_head_bench {
using namespace flint;
struct Buffers {
    sycl::queue& q; std::vector<void*> pointers;
    explicit Buffers(sycl::queue& queue) : q(queue) {}
    ~Buffers() { try { q.wait(); } catch (...) {} for (auto* p : pointers) sycl::free(p,q); }
    template<class T> T* alloc(size_t n) {
        T* p=sycl::malloc_device<T>(n,q); if(!p) throw std::bad_alloc{};
        pointers.push_back(p); return p;
    }
};
uint32_t hash(uint32_t x) { x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16); }
int run(int argc,char** argv) {
    try {
        int N=150144,K=2048,iters=20; bool check=false;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--check") {check=true;continue;}
            if(arg=="--help") {
                std::printf("flint-decode-head --n 150144 --k 2048 --iters 20 [--check]\n"
                            "ARCAINE_FLINT_DECODE_VEC=0|4|8|16 selects native or vector GPU loads.\n");
                return 0;
            }
            if(++i>=argc) throw std::invalid_argument("missing option value");
            size_t used=0; const int n=std::stoi(argv[i],&used);
            if(used!=std::string(argv[i]).size() || n<=0) throw std::invalid_argument("invalid integer");
            if(arg=="--n")N=n; else if(arg=="--k")K=n; else if(arg=="--iters")iters=n;
            else throw std::invalid_argument("unknown option");
        }
        if(K%512 || N>200000 || K>16384) throw std::invalid_argument("require K%512=0, K<=16384,N<=200000");
        sycl::queue q(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
            [](sycl::exception_list errors){for(const auto& e:errors)std::rethrow_exception(e);},
            {sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}});
        Buffers b(q);
        auto* w=b.alloc<bf16>(size_t(N)*K);
        auto* x=b.alloc<bf16>(K);
        auto* y=b.alloc<float>(N);
        auto* ref=b.alloc<float>(N);
        q.parallel_for(sycl::range<1>(size_t(N)*K),[=](sycl::id<1> i){
            w[i]=bf16(float(int(hash(uint32_t(i[0])+17)%2049)-1024)/32768.f);
        });
        q.parallel_for(sycl::range<1>(size_t(K)),[=](sycl::id<1> i){
            x[i]=bf16(float(int(hash(uint32_t(i[0])+91)%2049)-1024)/1024.f);
        });
        QTensor t; t.format=WeightFormat::BF16;t.n=N;t.k=K;t.dense=w;
        auto once=[&]{kernels::lm_head(q,t,x,y);};
        once();q.wait_and_throw();
        if(check) {
            kernels::decode_rows(q,N,[=](int row,int lane,sycl::sub_group sg){
                const float v=kernels::dot_tensor(t,row,0,x,lane,sg);
                if(!lane)ref[row]=float(bf16(v));
            });
            std::vector<float>a(N),e(N);
            q.memcpy(a.data(),y,size_t(N)*sizeof(float));
            q.memcpy(e.data(),ref,size_t(N)*sizeof(float)).wait_and_throw();
            int bad=0;float max_abs=0;
            for(int i=0;i<N;++i){const float error=std::fabs(a[i]-e[i]);max_abs=std::max(max_abs,error);
                if(!std::isfinite(a[i]) || error>0.01f+0.01f*std::fabs(e[i]))++bad;}
            std::printf("head GPU-reference checked=%d failed=%d max_abs=%g %s\n",
                N,bad,max_abs,bad?"FAIL":"PASS");
            if(bad)return 1;
        }
        const auto begin_event=q.ext_oneapi_submit_barrier();
        const auto start=std::chrono::steady_clock::now();
        for(int i=0;i<iters;++i)once();
        auto end_event=q.ext_oneapi_submit_barrier();
        end_event.wait_and_throw();
        const double wall=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-start).count()/iters;
        const double gpu=double(end_event.get_profiling_info<sycl::info::event_profiling::command_start>()-
            begin_event.get_profiling_info<sycl::info::event_profiling::command_end>())/1e6/iters;
        std::printf("flint-decode-head vector=%d N=%d K=%d iters=%d wall_ms=%.4f gpu_ms=%.4f GB/s=%.2f\n",
            decode_vec::width(),N,K,iters,wall,gpu,double(N)*K*2/(gpu*1e6));
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"flint-decode-head: %s\n",e.what());return 1;}
}
}
REGISTER_BENCH("flint-decode-head","Flint BF16 vocabulary projection: native and vectorized GPU GEMV", flint_head_bench::run)
