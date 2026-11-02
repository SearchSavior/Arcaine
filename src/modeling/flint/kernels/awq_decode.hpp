#pragma once
#include <cstdlib>
#include "../weights.hpp"
namespace flint::awq_decode {
inline bool enabled() {
    const char* p=std::getenv("ARCAINE_FLINT_DECODE_AWQ_VEC");
    return p && p[0]=='1' && p[1]=='\0';
}
template<bool Pair>
inline void dot4(const QTensor& t, const QTensor& up, int row, int expert,
                 const bf16* x, int lane, sycl::sub_group sg, float& out0, float& out1) {
    const uint8_t* w=t.w+size_t(expert)*t.stride_w+size_t(row)*(t.k/2);
    const uint8_t* u=nullptr;
    if constexpr(Pair) u=up.w+size_t(expert)*up.stride_w+size_t(row)*(up.k/2);
    float a[4]={},b[4]={};
    for(int i=lane*4;i<t.k;i+=128) {
        sycl::vec<uint32_t,2> input;
        input.load(0,reinterpret_cast<const uint32_t*>(x+i));
        const uint32_t lo=input[0],hi=input[1];
        const float xv[4]={sycl::bit_cast<float>(lo<<16),sycl::bit_cast<float>(lo&0xffff0000u),
                          sycl::bit_cast<float>(hi<<16),sycl::bit_cast<float>(hi&0xffff0000u)};
        const size_t p=size_t(expert)*t.stride_s+size_t(i/64)*t.n+row;
        const float scale=float(t.awq_s[p]);
        const int zero=int(t.awq_z[size_t(expert)*t.stride_b+size_t(i/64)*t.n+row]);
        const uint16_t codes=*reinterpret_cast<const uint16_t*>(w+i/2);
        uint16_t codes1=0; float scale1=0; int zero1=0;
        if constexpr(Pair) {
            const size_t p1=size_t(expert)*up.stride_s+size_t(i/64)*up.n+row;
            scale1=float(up.awq_s[p1]);
            zero1=int(up.awq_z[size_t(expert)*up.stride_b+size_t(i/64)*up.n+row]);
            codes1=*reinterpret_cast<const uint16_t*>(u+i/2);
        }
#pragma unroll
        for(int j=0;j<4;++j) {
            const float weight=float(bf16(float(int((codes>>(4*j))&15)-zero)*scale));
            a[j]=sycl::fma(weight,xv[j],a[j]);
            if constexpr(Pair) {
                const float weight1=float(bf16(float(int((codes1>>(4*j))&15)-zero1)*scale1));
                b[j]=sycl::fma(weight1,xv[j],b[j]);
            }
        }
    }
    float sum=a[0]+a[1]+a[2]+a[3];
    out0=sycl::reduce_over_group(sg,sum,sycl::plus<>());
    if constexpr(Pair) {
        sum=b[0]+b[1]+b[2]+b[3];
        out1=sycl::reduce_over_group(sg,sum,sycl::plus<>());
    }
}
}
