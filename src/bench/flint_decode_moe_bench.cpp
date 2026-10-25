#include "../common/bench/registry.hpp"
#include "../modeling/flint/kernels.hpp"
#include "../modeling/flint/awq_decode.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace flint_moe_bench {
using namespace flint;
struct Buffers {
    sycl::queue& q;std::vector<void*> pointers;
    explicit Buffers(sycl::queue& queue):q(queue){}
    ~Buffers(){try{q.wait();}catch(...){}for(auto* p:pointers)sycl::free(p,q);}
    template<class T>T* alloc(size_t n){
        auto* p=sycl::malloc_device<T>(n,q);if(!p)throw std::bad_alloc{};
        pointers.push_back(p);return p;
    }
};
uint32_t hash(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
int run(int argc,char** argv){
    try{
        int H=2048,I=512,S=1024,E=256,top=6,iters=50;bool check=false;
        std::string which="layer";
        for(int i=1;i<argc;++i){
            const std::string arg=argv[i];
            if(arg=="--check"){check=true;continue;}
            if(arg=="--help"){
                std::printf("flint-decode-moe --which layer|gateup|down --hidden 2048 --inter 512 "
                            "--shared 1024 --experts 256 --top-k 6 --iters 50 [--check]\n"
                            "ARCAINE_FLINT_DECODE_AWQ_VEC=1 enables packed AWQ loads; "
                            "ARCAINE_FLINT_DECODE_PARALLEL_DOWN=1 enables parallel expert down.\n");
                return 0;
            }
            if(++i>=argc)throw std::invalid_argument("missing value");
            if(arg=="--which"){which=argv[i];continue;}
            size_t used=0;const int n=std::stoi(argv[i],&used);
            if(used!=std::strlen(argv[i])||n<=0)throw std::invalid_argument("invalid integer");
            if(arg=="--hidden")H=n;else if(arg=="--inter")I=n;else if(arg=="--shared")S=n;
            else if(arg=="--experts")E=n;else if(arg=="--top-k")top=n;else if(arg=="--iters")iters=n;
            else throw std::invalid_argument("unknown argument");
        }
        if(H%128||I%128||S%128||S<I||top>E||E>256||H>8192||S>8192||
           (which!="layer"&&which!="gateup"&&which!="down"))throw std::invalid_argument("invalid shape/case");
        sycl::queue q(sycl::ext::oneapi::filter_selector("level_zero:gpu"),
            [](sycl::exception_list errors){for(const auto& e:errors)std::rethrow_exception(e);},
            sycl::property::queue::in_order{});
        Buffers b(q);
        auto tensor=[&](int N,int K,int experts,uint32_t seed){
            QTensor t;t.format=WeightFormat::AWQ;t.decode_vector=awq_decode::enabled();
            t.n=N;t.k=K;t.experts=experts;t.stride_w=size_t(N)*(K/2);
            t.stride_s=t.stride_b=size_t(N)*(K/64);
            const size_t count=experts?experts:1;
            auto* w=b.alloc<uint8_t>(count*t.stride_w);
            auto* s=b.alloc<sycl::half>(count*t.stride_s);
            auto* z=b.alloc<uint8_t>(count*t.stride_b);
            t.w=w;t.awq_s=s;t.awq_z=z;
            q.parallel_for(sycl::range<1>(count*t.stride_w),[=](sycl::id<1> id){
                w[id]=uint8_t(hash(uint32_t(id[0])+seed));
            });
            q.parallel_for(sycl::range<1>(count*t.stride_s),[=](sycl::id<1> id){
                const auto r=hash(uint32_t(id[0])+seed+23);
                s[id]=sycl::half(0.001f+float(r%100)*0.0001f);z[id]=uint8_t((r>>16)&15);
            });
            return t;
        };
        FlintLayer L;L.moe=true;
        L.egate=tensor(I,H,E,11);L.eup=tensor(I,H,E,31);L.edown=tensor(H,I,E,57);
        L.sgate=tensor(S,H,0,71);L.sup=tensor(S,H,0,97);L.sdown=tensor(H,S,0,113);
        auto reference=L;
        for(auto* t:{&reference.egate,&reference.eup,&reference.edown,&reference.sgate,&reference.sup,&reference.sdown})
            t->decode_vector=false;
        auto* x=b.alloc<bf16>(H);auto* act=b.alloc<bf16>(size_t(top+1)*S);
        auto* expected_act=b.alloc<bf16>(size_t(top+1)*S);
        auto* output=b.alloc<bf16>(H);auto* expected=b.alloc<bf16>(H);
        auto* down=b.alloc<bf16>(size_t(top+1)*H);
        auto* ids=b.alloc<int>(top);auto* weights=b.alloc<float>(top);
        q.parallel_for(sycl::range<1>(size_t(H)),[=](sycl::id<1> id){
            x[id]=bf16(float(int(hash(uint32_t(id[0])+137)%2049)-1024)/1024.f);
        });
        std::vector<int> hid(top);for(int i=0;i<top;++i)hid[i]=(i*37)%E;
        // Ensure unique ranks even when E shares a factor with 37.
        if(E%37==0)for(int i=0;i<top;++i)hid[i]=i;
        q.memcpy(ids,hid.data(),size_t(top)*sizeof(int));
        q.fill(weights,float(bf16(1.f/top)),top);
        q.memset(output,0,size_t(H)*sizeof(bf16));q.memset(expected,0,size_t(H)*sizeof(bf16));
        kernels::expert_gateup(q,reference,H,I,S,top,x,ids,expected_act);
        q.memcpy(act,expected_act,size_t(top+1)*S*sizeof(bf16)).wait_and_throw();
        const char* p=std::getenv("ARCAINE_FLINT_DECODE_PARALLEL_DOWN");
        const bool parallel=p&&std::strcmp(p,"1")==0;
        auto once=[&]{
            if(which!="down")kernels::expert_gateup(q,L,H,I,S,top,x,ids,act);
            if(which!="gateup"){
                if(parallel){
                    kernels::expert_down(q,L,H,I,S,top,act,ids,down);
                    kernels::moe_combine(q,output,down,weights,H,top);
                }else kernels::moe_down_combine(q,L,H,I,S,top,act,ids,weights,output);
            }
        };
        if(check){
            kernels::moe_down_combine(q,reference,H,I,S,top,expected_act,ids,weights,expected);
            once();q.wait_and_throw();
            std::vector<bf16>a(which=="gateup"?size_t(top+1)*S:size_t(H)),e(a.size());
            q.memcpy(a.data(),which=="gateup"?act:output,a.size()*sizeof(bf16));
            q.memcpy(e.data(),which=="gateup"?expected_act:expected,e.size()*sizeof(bf16)).wait_and_throw();
            size_t bad=0;float worst=0;
            for(size_t i=0;i<a.size();++i){
                if(which=="gateup"&&i/size_t(S)<size_t(top)&&i%S>=size_t(I))continue;
                const float error=std::fabs(float(a[i])-float(e[i]));worst=std::max(worst,error);
                if(!std::isfinite(float(a[i]))||error>0.01f+0.01f*std::fabs(float(e[i])))++bad;
            }
            std::printf("moe GPU-reference failed=%zu max_abs=%g %s\n",bad,worst,bad?"FAIL":"PASS");
            if(bad)return 1;
        }
        once();q.wait_and_throw();
        const auto start=std::chrono::steady_clock::now();
        for(int i=0;i<iters;++i)once();
        q.wait_and_throw();
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/iters;
        std::printf("flint-decode-moe case=%s vector=%d parallel_down=%d H=%d I=%d S=%d E=%d top=%d iters=%d ms=%.4f\n",
                    which.c_str(),awq_decode::enabled(),parallel,H,I,S,E,top,iters,ms);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"flint-decode-moe: %s\n",e.what());return 1;}
}
}
REGISTER_BENCH("flint-decode-moe","Flint routed/shared AWQ decode gate/up, down and residual", flint_moe_bench::run)
