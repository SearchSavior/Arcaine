#include "../common/bench/registry.hpp"
#include "../modeling/flint/kernels.hpp"
#include "../modeling/flint/decode_router.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
namespace flint_router_bench {
using namespace flint;
struct Buffers{
    sycl::queue& q;std::vector<void*> all;
    explicit Buffers(sycl::queue& queue):q(queue){}
    ~Buffers(){try{q.wait();}catch(...){}for(auto* p:all)sycl::free(p,q);}
    template<class T>T* alloc(size_t n){auto* p=sycl::malloc_device<T>(n,q);if(!p)throw std::bad_alloc{};all.push_back(p);return p;}
};
uint32_t hash(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
int run(int argc,char** argv){
    try{
        int E=256,H=2048,top=6,iters=100;bool check=false,zero=false;
        for(int i=1;i<argc;++i){
            const std::string a=argv[i];
            if(a=="--check"){check=true;continue;}if(a=="--zero"){zero=true;continue;}
            if(a=="--help"){std::printf("flint-decode-router --experts 256 --hidden 2048 --top-k 6 --iters 100 [--zero] [--check]\n");return 0;}
            if(++i>=argc)throw std::invalid_argument("missing value");
            size_t used=0;const int n=std::stoi(argv[i],&used);
            if(used!=std::string(argv[i]).size()||n<=0)throw std::invalid_argument("invalid integer");
            if(a=="--experts")E=n;else if(a=="--hidden")H=n;else if(a=="--top-k")top=n;
            else if(a=="--iters")iters=n;else throw std::invalid_argument("unknown option");
        }
        if(E>256||top>E||top>16||H%32||H>16384)throw std::invalid_argument("invalid shape");
        sycl::queue q(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
            [](sycl::exception_list errors){for(const auto& e:errors)std::rethrow_exception(e);},
            sycl::property::queue::in_order{});
        Buffers b(q);auto* w=b.alloc<bf16>(size_t(E)*H);auto* x=b.alloc<bf16>(H);
        auto* ids=b.alloc<int>(top);auto* ref_ids=b.alloc<int>(top);
        auto* weights=b.alloc<float>(top);auto* ref_weights=b.alloc<float>(top);auto* scratch=b.alloc<float>(E);
        q.parallel_for(sycl::range<1>(size_t(E)*H),[=](sycl::id<1> i){
            w[i]=zero?bf16(0.f):bf16(float(int(hash(uint32_t(i[0])+17)%2049)-1024)/32768.f);
        });
        q.parallel_for(sycl::range<1>(size_t(H)),[=](sycl::id<1> i){
            x[i]=bf16(float(int(hash(uint32_t(i[0])+91)%2049)-1024)/1024.f);
        });
        auto once=[&]{kernels::router_topk(q,w,x,E,H,top,ids,weights,scratch);};
        once();q.wait_and_throw();
        if(check){
            kernels::router_topk(q,w,x,E,H,top,ref_ids,ref_weights);
            std::vector<int>a(top),e(top);std::vector<float>aw(top),ew(top);
            q.memcpy(a.data(),ids,size_t(top)*sizeof(int));q.memcpy(e.data(),ref_ids,size_t(top)*sizeof(int));
            q.memcpy(aw.data(),weights,size_t(top)*sizeof(float));q.memcpy(ew.data(),ref_weights,size_t(top)*sizeof(float)).wait_and_throw();
            int bad=0;for(int i=0;i<top;++i)if(a[i]!=e[i]||!std::isfinite(aw[i])||std::fabs(aw[i]-ew[i])>1e-7f)++bad;
            std::printf("router GPU-reference failed=%d %s\n",bad,bad?"FAIL":"PASS");
            if(bad){for(int i=0;i<top;++i)std::printf("rank=%d actual=%d expected=%d weight=%g reference=%g\n",i,a[i],e[i],aw[i],ew[i]);return 1;}
        }
        const auto start=std::chrono::steady_clock::now();for(int i=0;i<iters;++i)once();q.wait_and_throw();
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/iters;
        std::printf("flint-decode-router parallel=%d subgroup=%d E=%d H=%d top=%d zero=%d iters=%d ms=%.4f\n",
                    decode_router::enabled(),decode_router::subgroup_topk_enabled(),E,H,top,zero,iters,ms);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"flint-decode-router: %s\n",e.what());return 1;}
}
}
REGISTER_BENCH("flint-decode-router","Flint BF16 sigmoid/top-k router with parallel GPU projection and selection", flint_router_bench::run)
