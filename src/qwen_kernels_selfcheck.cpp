// Build-verification + GPU smoke for src/modeling/qwen3_5_moe/kernels.hpp.
//
// An unused inline SYCL lambda is NOT device-codegen'd, so merely compiling
// model.cpp (whose forward() is a stub) does not validate the kernel bodies.
// This TU calls every kernel once with tiny buffers, which forces full SPIR-V
// codegen at compile time; running it on a GPU container additionally
// exercises enqueue + a light numerical sanity (finite + a few known values).
//
//   compile : arcaine-dev-1 (SPIR-V codegen needs no physical GPU)
//   run     : arcaine-dev-run-* (has the BMG G31s)
//
// Full numerical validation against the HF reference is Phase 6; this is the
// milestone-2 build/run smoke.
#include "modeling/qwen3_5_moe/kernels.hpp"
#include "common/gpu/engine.hpp"
#include "common/gpu/buffer.hpp"

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

using bf16 = uint16_t;

static std::vector<bf16> to_bf16(const std::vector<float>& vs) {
    std::vector<bf16> out(vs.size());
    for (size_t i = 0; i < vs.size(); ++i) out[i] = float_to_bf16(vs[i]);
    return out;
}
static std::vector<bf16> const_bf16(int n, float v) {
    return to_bf16(std::vector<float>(static_cast<size_t>(n), v));
}
static std::vector<float> to_float(const std::vector<bf16>& v) {
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = bf16_to_float(v[i]);
    return out;
}
static bool all_finite(const std::vector<float>& v) {
    for (float f : v) if (!std::isfinite(f)) return false;
    return true;
}

int main() {
    int fails = 0;
    auto check = [&](const char* name, bool ok) {
        std::printf("[selfcheck] %-22s %s\n", name, ok ? "OK" : "FAIL");
        if (!ok) ++fails;
    };

    try {
        auto& q = GpuEngine::get(0).queue;
        std::printf("[selfcheck] device: %s\n",
            q.get_device().get_info<sycl::info::device::name>().c_str());

        // silu_inplace: silu(0.5)=0.359, silu(0)=0.
        {
            const int n = 8;
            auto h = to_bf16({0.5f, 1.0f, -0.5f, 2.0f, -2.0f, 0.0f, 0.25f, -0.25f});
            GpuBuffer<bf16> d(n, q); d.upload(h.data(), n);
            silu_inplace(q, d.data(), n);
            auto f = to_float({}); { std::vector<bf16> o(n); d.download(o.data(), n); f = to_float(o); }
            // silu(0.5)=0.5/(1+e^-0.5)=0.3112; silu(0)=0.
            check("silu_inplace",
                  all_finite(f) && std::fabs(f[0] - 0.3112f) < 0.02f && f[5] == 0.0f);
        }
        // sigmoid_inplace: sigmoid(0)=0.5, sigmoid(10)~1.
        {
            const int n = 6;
            auto h = to_bf16({0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 10.0f});
            GpuBuffer<bf16> d(n, q); d.upload(h.data(), n);
            sigmoid_inplace(q, d.data(), n);
            std::vector<bf16> o(n); d.download(o.data(), n); auto f = to_float(o);
            check("sigmoid_inplace",
                  all_finite(f) && std::fabs(f[0] - 0.5f) < 0.02f && f[5] > 0.99f);
        }
        // swiglu_inplace: silu(1)*2 = 0.731*2 = 1.462.
        {
            const int n = 4;
            GpuBuffer<bf16> dg(n, q), du(n, q);
            dg.upload(const_bf16(n, 1.0f).data(), n);
            du.upload(const_bf16(n, 2.0f).data(), n);
            swiglu_inplace(q, dg.data(), du.data(), n);
            std::vector<bf16> o(n); dg.download(o.data(), n); auto f = to_float(o);
            check("swiglu_inplace", all_finite(f) && std::fabs(f[0] - 1.462f) < 0.05f);
        }
        // swiglu_strided: same math, stacked (seq,2*inter) -> (seq,inter).
        {
            const int seq = 2, inter = 4;
            std::vector<bf16> h(seq * 2 * inter);
            for (int s = 0; s < seq; ++s) {
                for (int i = 0; i < inter; ++i) h[s * 2 * inter + i] = float_to_bf16(1.0f);
                for (int i = 0; i < inter; ++i) h[s * 2 * inter + inter + i] = float_to_bf16(2.0f);
            }
            GpuBuffer<bf16> dh(seq * 2 * inter, q), dout(seq * inter, q);
            dh.upload(h.data(), seq * 2 * inter);
            swiglu_strided(q, dh.data(), dout.data(), seq, inter);
            std::vector<bf16> o(seq * inter); dout.download(o.data(), seq * inter); auto f = to_float(o);
            check("swiglu_strided", all_finite(f) && std::fabs(f[0] - 1.462f) < 0.05f);
        }
        // mul_sigmoid_inplace: a*=sigmoid(gate); sigmoid(0)=0.5, sigmoid(10)~1.
        {
            const int n = 4;
            GpuBuffer<bf16> da(n, q), dg(n, q);
            da.upload(const_bf16(n, 1.0f).data(), n);
            dg.upload(to_bf16({0.0f, 2.0f, -2.0f, 10.0f}).data(), n);
            mul_sigmoid_inplace(q, da.data(), dg.data(), n);
            std::vector<bf16> o(n); da.download(o.data(), n); auto f = to_float(o);
            check("mul_sigmoid_inplace",
                  all_finite(f) && std::fabs(f[0] - 0.5f) < 0.02f && f[3] > 0.99f);
        }
        // scale_rows_by_sigmoid: per-row scalar broadcast; row0 gate=0 ->*0.5, row1 gate=10 ->*~1.
        {
            const int n_rows = 2, row_dim = 4;
            GpuBuffer<bf16> dout(n_rows * row_dim, q), dg(n_rows, q);
            dout.upload(const_bf16(n_rows * row_dim, 1.0f).data(), n_rows * row_dim);
            dg.upload(to_bf16({0.0f, 10.0f}).data(), n_rows);
            scale_rows_by_sigmoid(q, dout.data(), dg.data(), n_rows, row_dim);
            std::vector<bf16> o(n_rows * row_dim); dout.download(o.data(), n_rows * row_dim);
            auto f = to_float(o);
            check("scale_rows_by_sigmoid",
                  all_finite(f) && std::fabs(f[0] - 0.5f) < 0.02f && f[row_dim] > 0.99f);
        }
        // apply_qwen_rope: head_dim=256, rotary_dim=64, half=32; offset=1 -> real rotation.
        //   pair_i=0: inv_freq=1, angle=1 rad, cos=0.5403, sin=0.8415
        //   row[0]=c-s=-0.3012, row[32]=s+c=1.3818; channel 64 (pass-through) stays 1.0.
        {
            const int seq = 2, nq = 4, nkv = 2, head_dim = 256, rotary_dim = 64;
            const int offset = 1;
            const float theta = 1.0e6f;
            const int qn = seq * nq * head_dim, kn = seq * nkv * head_dim;
            GpuBuffer<bf16> dq(qn, q), dk(kn, q);
            dq.upload(const_bf16(qn, 1.0f).data(), qn);
            dk.upload(const_bf16(kn, 1.0f).data(), kn);
            apply_qwen_rope(q, dq.data(), dk.data(), seq, offset, nq, nkv,
                            head_dim, rotary_dim, theta);
            std::vector<bf16> oq(qn); dq.download(oq.data(), qn); auto fq = to_float(oq);
            std::vector<bf16> ok_raw(kn); dk.download(ok_raw.data(), kn); auto fk = to_float(ok_raw);
            bool ok = all_finite(fq) && all_finite(fk);
            ok = ok && std::fabs(fq[0] - (-0.3012f)) < 0.05f;          // rotated pair
            ok = ok && std::fabs(fq[32] - 1.3818f) < 0.05f;            // rotated pair
            ok = ok && std::fabs(fq[rotary_dim] - 1.0f) < 0.02f;       // pass-through channel
            ok = ok && std::fabs(fk[0] - (-0.3012f)) < 0.05f;          // k rotated too
            check("apply_qwen_rope", ok);
        }
        // gated_rmsnorm: all-ones x, ones weight, z=10 (silu(10)~9.9995).
        //   rmsnorm_D(all-ones, D=64) = 1; out = 1*1*silu(10) ~ 9.9995.
        {
            const int N = 2, D = 64; const float eps = 1e-6f;
            GpuBuffer<bf16> dx(N * D, q), dz(N * D, q), dw(D, q), dout(N * D, q);
            dx.upload(const_bf16(N * D, 1.0f).data(), N * D);
            dz.upload(const_bf16(N * D, 10.0f).data(), N * D);
            dw.upload(const_bf16(D, 1.0f).data(), D);
            gated_rmsnorm(q, dx.data(), dz.data(), dw.data(), dout.data(), N, D, eps);
            std::vector<bf16> o(N * D); dout.download(o.data(), N * D); auto f = to_float(o);
            check("gated_rmsnorm", all_finite(f) && std::fabs(f[0] - 10.0f) < 0.5f);
        }
        // l2norm: all-3 over D=64 -> norm=sqrt(576)=24, out=3/24=0.125.
        {
            const int N = 2, D = 64; const float eps = 1e-6f;
            GpuBuffer<bf16> dx(N * D, q), dout(N * D, q);
            dx.upload(const_bf16(N * D, 3.0f).data(), N * D);
            l2norm(q, dx.data(), dout.data(), N, D, eps);
            std::vector<bf16> o(N * D); dout.download(o.data(), N * D); auto f = to_float(o);
            check("l2norm", all_finite(f) && std::fabs(f[0] - 0.125f) < 0.01f);
        }

        std::printf("[selfcheck] %s (%d failures)\n",
                    fails ? "FAILED" : "ALL PASSED", fails);
        return fails ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("[selfcheck] EXCEPTION: %s\n", e.what());
        return 2;
    }
}
