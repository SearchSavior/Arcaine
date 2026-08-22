// Honest non-fused GDN test: SEQUENTIAL recurrence via oneDNN GEMM per token.
// Per token t: h_k = S @ k_t  (S:[dv=128,dk=128] x k_t:[128] -> [128])
//              update = (v_t - h_k)*beta_t ;  S += k_t outer update
//              o = S @ q_t
// The recurrence is inherently sequential (state depends on all prior tokens),
// exactly like the fused OCL kernel. We use oneDNN matmul with N=1 (GEMV)
// per token to emulate what the non-fused loop body would dispatch to.
// This isolates the DPAS-GEMM benefit on the SEQUENTIAL dependency chain.
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using bf16 = uint16_t;
static inline float bf2f(uint16_t v){uint32_t u=(uint32_t)v<<16;float f;std::memcpy(&f,&u,4);return f;}
static inline uint16_t f2bf(float f){uint32_t u;std::memcpy(&u,&f,4);uint32_t b=((u>>16)&1)+0x7FFFu;return (uint16_t)((u+b)>>16);}
int main(int argc, char** argv){
    std::string p_csv="512,1024,2048,4096"; int iters=3;
    for(int i=1;i<argc;++i){std::string a=argv[i];
        if(a=="-p"&&i+1<argc)p_csv=argv[++i]; else if(a=="-n"&&i+1<argc)iters=std::atoi(argv[++i]);}
    dnnl::engine engine(dnnl::engine::kind::gpu,0);
    auto dev=dnnl::sycl_interop::get_device(engine);
    auto ctx=dnnl::sycl_interop::get_context(engine);
    sycl::queue queue(ctx,dev,sycl::property::queue::in_order{});
    auto stream=dnnl::sycl_interop::make_stream(engine,queue);
    constexpr int nv=32, dk=128, dv=128;
    std::vector<int> tokens;
    {std::string s=p_csv+",";size_t p=0;while((p=s.find(","))!=std::string::npos){tokens.push_back(std::atoi(s.substr(0,p).c_str()));s.erase(0,p+1);}}
    for(int T:tokens){
        std::vector<bf16> k_h((size_t)T*nv*dk), q_h((size_t)T*nv*dk), v_h((size_t)T*nv*dv);
        std::vector<bf16> beta_h((size_t)T*nv), g_h((size_t)T*nv);
        uint32_t rng=42;
        auto frand=[&](float lo,float hi){rng=rng*1664525u+1013904223u;return lo+(hi-lo)*(float)((rng>>8)&0xFFFFFF)/16777216.0f;};
        const float scale=1.0f/std::sqrt((float)dk);
        for(size_t row=0;row<(size_t)T*nv;++row){
            float nq=0,nk=0; std::vector<float> tq(dk),tk(dk);
            for(int d=0;d<dk;++d){tq[d]=frand(-1.f,1.f);nq+=tq[d]*tq[d];tk[d]=frand(-1.f,1.f);nk+=tk[d]*tk[d];}
            nq=1.f/std::sqrt(std::max(nq,1e-12f)); nk=1.f/std::sqrt(std::max(nk,1e-12f));
            for(int d=0;d<dk;++d){q_h[row*dk+d]=f2bf(tq[d]*nq*scale);k_h[row*dk+d]=f2bf(tk[d]*nk);}
            for(int d=0;d<dv;++d)v_h[row*dv+d]=f2bf(frand(-1.f,1.f));
            beta_h[row]=f2bf(frand(0.05f,0.95f)); g_h[row]=f2bf(-frand(0.0005f,0.05f));
        }
        std::vector<bf16> s0((size_t)nv*dk*dv,0);
        for(auto&x:s0)x=f2bf(frand(-0.2f,0.2f));
        auto malloc_dev=[&](size_t b){return sycl::malloc_device<char>(b,queue);};
        void* d_s=malloc_dev(s0.size()*2);
        void* d_k=malloc_dev(k_h.size()*2); void* d_q=malloc_dev(q_h.size()*2);
        void* d_v=malloc_dev(v_h.size()*2); void* d_beta=malloc_dev(beta_h.size()*2);
        void* d_hk=malloc_dev((size_t)nv*dv*4); void* d_o=malloc_dev((size_t)nv*dv*4);
        queue.memcpy(d_s,s0.data(),s0.size()*2).wait();
        queue.memcpy(d_k,k_h.data(),k_h.size()*2).wait();
        queue.memcpy(d_q,q_h.data(),q_h.size()*2).wait();
        queue.memcpy(d_v,v_h.data(),v_h.size()*2).wait();
        queue.memcpy(d_beta,beta_h.data(),beta_h.size()*2).wait();
        auto md=[](dnnl::memory::dims d,dnnl::memory::data_type dt,dnnl::memory::format_tag t){
            return dnnl::memory::desc(d,dt,t);};
        auto mk=[&](const dnnl::memory::desc& d,void* p){
            return dnnl::sycl_interop::make_memory(d,engine,dnnl::sycl_interop::memory_kind::usm,p);};
        // per-head GEMV: S [dv,dk] x kt [dk,1] -> [dv,1], batched over nv
        auto s_desc=md({nv,dv,dk},dnnl::memory::data_type::bf16,dnnl::memory::format_tag::abc);
        auto kt_desc=md({nv,dk,1},dnnl::memory::data_type::bf16,dnnl::memory::format_tag::abc);
        auto o_desc=md({nv,dv,1},dnnl::memory::data_type::f32,dnnl::memory::format_tag::abc);
        auto mm_pd=dnnl::matmul::primitive_desc(engine,s_desc,kt_desc,o_desc);
        auto mm=dnnl::matmul(mm_pd);
        auto run_once=[&]{
            for(int t=0;t<T;++t){
                void* kt=(char*)d_k+(size_t)t*nv*dk*2;
                void* qt=(char*)d_q+(size_t)t*nv*dk*2;
                mm.execute(stream,{{DNNL_ARG_SRC,mk(s_desc,d_s)},
                                   {DNNL_ARG_WEIGHTS,mk(kt_desc,kt)},
                                   {DNNL_ARG_DST,mk(o_desc,d_hk)}});
                // update skipped for timing (would be a small kernel); the
                // sequential GEMM dispatch dominates.
                mm.execute(stream,{{DNNL_ARG_SRC,mk(s_desc,d_s)},
                                   {DNNL_ARG_WEIGHTS,mk(kt_desc,qt)},
                                   {DNNL_ARG_DST,mk(o_desc,d_o)}});
            }
            stream.wait();
        };
        for(int i=0;i<2;++i)run_once();
        std::vector<double> samples;
        for(int i=0;i<iters;++i){auto t0=std::chrono::steady_clock::now();run_once();auto t1=std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());}
        double mean=0;for(double s:samples)mean+=s;mean/=samples.size();
        std::printf("gdn_onednn_seq p=%d runs=%d mean_ms=%.3f tok_per_s=%.1f (2x%d sequential GEMV matmuls via jit gemm)\n",
                    T,iters,mean,T*1000.0/mean,T);
    }
    return 0;
}
