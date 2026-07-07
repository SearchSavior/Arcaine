// Numerical self-check for the DiffusionGemma attention leaf kernels in
// attention_kernels.hpp: fused RMSNorm+RoPE (pow / table / fused modes) and
// fused masked softmax (span / branchy).  Calls each kernel directly (not via
// the attention.cpp dispatch, whose DIFF_FUSED_NORM_ROPE / DIFF_SOFTMAX_SPAN
// knobs cache on first read) so all variants exercise in one binary, and
// compares each to a host float reference + to each other.
//
//   build : arcaine-dev-1 (SPIR-V codegen needs no physical GPU)
//   run   : arcaine-dev-run-* (BMG G31)
//
// Tolerance is bf16-rounded composed-ops: ~3 decimal digits, so abs/rel ~2e-2.
#include "modeling/diffusion_gemma/attention_kernels.hpp"
#include "common/gpu/buffer.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using bf16 = uint16_t;

static std::vector<bf16> to_bf16(const std::vector<float>& vs) {
    std::vector<bf16> out(vs.size());
    for (size_t i = 0; i < vs.size(); ++i) out[i] = float_to_bf16(vs[i]);
    return out;
}
static std::vector<float> to_float(const std::vector<bf16>& v) {
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = bf16_to_float(v[i]);
    return out;
}

// Host reference for fused RMSNorm + RoPE (rotate_half, partial rotary).
static std::vector<float> rms_norm_rope_ref(
    const std::vector<float>& x, const std::vector<float>& weight,
    int seq, int nheads, int hd, int offset,
    float theta, float partial, float eps)
{
    int pair_offset = hd / 2;
    int n_active_pairs = (int)(partial * hd / 2.0f);
    float freq_denom = (float)hd;
    std::vector<float> out = x;
    for (int tok = 0; tok < seq; ++tok)
    for (int head = 0; head < nheads; ++head) {
        float* row = &out[((size_t)tok * nheads + head) * hd];
        float ss = 0.0f;
        for (int d = 0; d < hd; ++d) ss += row[d] * row[d];
        float rms_inv = 1.0f / std::sqrt(ss / float(hd) + eps);
        for (int d = 0; d < hd; ++d) row[d] = row[d] * rms_inv * weight[d];
        for (int p = 0; p < n_active_pairs; ++p) {
            float inv_freq = 1.0f / std::pow(theta, 2.0f * p / freq_denom);
            float angle = (float)(offset + tok) * inv_freq;
            float c = std::cos(angle), s = std::sin(angle);
            float x0 = row[p], x1 = row[p + pair_offset];
            row[p] = x0 * c - x1 * s;
            row[p + pair_offset] = x0 * s + x1 * c;
        }
    }
    return out;
}

// Host reference for fused masked softmax (contiguous-span semantics).
static std::vector<float> softmax_ref(
    const std::vector<float>& scores,
    int nq, int seq, int kv_len,
    int q_pos0, int kv_pos0, int sliding_window, bool causal)
{
    std::vector<float> out = scores;
    for (int q = 0; q < nq; ++q)
    for (int sq = 0; sq < seq; ++sq) {
        size_t row = (size_t)(q * seq + sq) * kv_len;
        int q_global = q_pos0 + sq;
        int lo = 0, hi = kv_len;
        if (causal) {
            int h = q_global - kv_pos0 + 1;
            hi = h < kv_len ? h : kv_len;
            if (sliding_window != INT_MAX) {
                int l = q_global - kv_pos0 - sliding_window + 1;
                lo = l > 0 ? l : 0;
            }
        }
        float m = -3.4028235e38f;
        for (int c = lo; c < hi; ++c) m = std::max(m, out[row + c]);
        float z = 0.0f;
        for (int c = lo; c < hi; ++c) z += std::exp(out[row + c] - m);
        float inv_z = 1.0f / z;
        for (int c = 0; c < kv_len; ++c)
            out[row + c] = (c >= lo && c < hi) ? std::exp(out[row + c] - m) * inv_z : 0.0f;
    }
    return out;
}

static bool approx(const std::vector<float>& a, const std::vector<float>& b,
                   float abs_tol, float rel_tol, const char*& why) {
    if (a.size() != b.size()) { why = "size mismatch"; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > abs_tol && d > rel_tol * (std::fabs(a[i]) + std::fabs(b[i]) + 1e-6f)) {
            why = "value"; return false;
        }
    }
    return true;
}

int main() {
    int fails = 0;
    auto check = [&](const char* name, bool ok) {
        std::printf("[selfcheck] %-34s %s\n", name, ok ? "OK" : "FAIL");
        if (!ok) ++fails;
    };

    try {
        auto& q = GpuEngine::get(0).queue;
        std::printf("[selfcheck] device: %s\n",
            q.get_device().get_info<sycl::info::device::name>().c_str());

        // ----- RMSNorm + RoPE: tiny dims, full rotary -----
        const int seq = 4, nheads = 8, hd = 16, offset = 3;
        const float theta = 10000.0f, partial = 1.0f, eps = 1e-6f;
        const size_t N = (size_t)seq * nheads * hd;

        std::vector<float> x_host(N), w_host(hd);
        // deterministic non-trivial input
        for (size_t i = 0; i < N; ++i) x_host[i] = -1.0f + 0.03f * (float)((i * 7) % 64);
        for (int d = 0; d < hd; ++d) w_host[d] = 0.5f + 0.1f * (float)d;

        auto ref = rms_norm_rope_ref(x_host, w_host, seq, nheads, hd, offset, theta, partial, eps);
        auto x_bf = to_bf16(x_host);
        auto w_bf = to_bf16(w_host);

        // pow mode (null table)
        {
            GpuBuffer<bf16> d(N, q); d.upload(x_bf.data(), N);
            GpuBuffer<bf16> w(hd, q); w.upload(w_bf.data(), hd);
            fused_norm_rope_inplace(q, d.data(), w.data(), seq, nheads, hd, offset,
                                    theta, partial, eps, /*rope_cos=*/nullptr, /*rope_sin=*/nullptr);
            q.wait();
            std::vector<bf16> o(N); d.download(o.data(), N);
            const char* why = ""; bool ok = approx(to_float(o), ref, 0.02f, 1e-2f, why);
            check("norm+rope pow (null table)", ok);
            if (!ok) std::printf("    why=%s idx0 ref=%f got=%f\n", why, ref[0], to_float(o)[0]);
        }
        // table mode
        {
            GpuBuffer<bf16> d(N, q); d.upload(x_bf.data(), N);
            GpuBuffer<bf16> w(hd, q); w.upload(w_bf.data(), hd);
            auto& ctx = GpuEngine::get(0);
            RopeTable rope = make_rope_table(ctx, seq, offset, hd, theta, partial);
            q.wait();
            fused_norm_rope_inplace(q, d.data(), w.data(), seq, nheads, hd, offset,
                                    theta, partial, eps, rope.cos_data(), rope.sin_data());
            q.wait();
            std::vector<bf16> o(N); d.download(o.data(), N);
            const char* why = ""; bool ok = approx(to_float(o), ref, 0.02f, 1e-2f, why);
            check("norm+rope table", ok);
            if (!ok) std::printf("    why=%s idx0 ref=%f got=%f\n", why, ref[0], to_float(o)[0]);
        }
        // fused mode (per-token local cos/sin)
        {
            GpuBuffer<bf16> d(N, q); d.upload(x_bf.data(), N);
            GpuBuffer<bf16> w(hd, q); w.upload(w_bf.data(), hd);
            fused_norm_rope_inplace_fused(q, d.data(), w.data(), seq, nheads, hd, offset,
                                          theta, partial, eps);
            q.wait();
            std::vector<bf16> o(N); d.download(o.data(), N);
            const char* why = ""; bool ok = approx(to_float(o), ref, 0.02f, 1e-2f, why);
            check("norm+rope fused (per-token)", ok);
            if (!ok) std::printf("    why=%s idx0 ref=%f got=%f\n", why, ref[0], to_float(o)[0]);
        }

        // ----- masked softmax: causal + sliding -----
        const int nq = 4, sseq = 4, kv_len = 8, sw = 3;
        const int q_pos0 = 3, kv_pos0 = 1;
        const size_t S = (size_t)nq * sseq * kv_len;
        std::vector<float> sc_host(S);
        for (size_t i = 0; i < S; ++i) sc_host[i] = -2.0f + 0.1f * (float)((i * 5) % 40);
        auto sc_bf = to_bf16(sc_host);
        // Device writes bf16-rounded outputs; round the ref through bf16 too so
        // the only remaining diff is the reduction's float accumulation.
        auto sc_ref = softmax_ref(to_float(sc_bf), nq, sseq, kv_len, q_pos0, kv_pos0, sw, /*causal=*/true);
        auto sc_ref_bf = to_float(to_bf16(sc_ref));

        // span
        {
            GpuBuffer<bf16> d(S, q); d.upload(sc_bf.data(), S);
            fused_masked_softmax(q, d.data(), nq, sseq, kv_len, q_pos0, kv_pos0, sw, true);
            q.wait();
            std::vector<bf16> o(S); d.download(o.data(), S);
            const char* why = ""; bool ok = approx(to_float(o), sc_ref_bf, 1e-4f, 1e-4f, why);
            check("softmax causal+sliding (span)", ok);
            if (!ok) std::printf("    why=%s idx0 ref=%f got=%f\n", why, sc_ref[0], to_float(o)[0]);
        }
        // branchy
        {
            GpuBuffer<bf16> d(S, q); d.upload(sc_bf.data(), S);
            fused_masked_softmax_branchy(q, d.data(), nq, sseq, kv_len, q_pos0, kv_pos0, sw, true);
            q.wait();
            std::vector<bf16> o(S); d.download(o.data(), S);
            const char* why = ""; bool ok = approx(to_float(o), sc_ref_bf, 1e-4f, 1e-4f, why);
            check("softmax causal+sliding (branchy)", ok);
            if (!ok) std::printf("    why=%s idx0 ref=%f got=%f\n", why, sc_ref[0], to_float(o)[0]);
        }
        // non-causal (bidirectional) — span == branchy == ref
        {
            auto ref_nc = to_float(to_bf16(softmax_ref(to_float(sc_bf), nq, sseq, kv_len, q_pos0, kv_pos0, INT_MAX, false)));
            GpuBuffer<bf16> a(S, q); a.upload(sc_bf.data(), S);
            GpuBuffer<bf16> b(S, q); b.upload(sc_bf.data(), S);
            fused_masked_softmax(q, a.data(), nq, sseq, kv_len, q_pos0, kv_pos0, INT_MAX, false);
            fused_masked_softmax_branchy(q, b.data(), nq, sseq, kv_len, q_pos0, kv_pos0, INT_MAX, false);
            q.wait();
            std::vector<bf16> oa(S), ob(S); a.download(oa.data(), S); b.download(ob.data(), S);
            const char* why = "";
            bool ok1 = approx(to_float(oa), ref_nc, 1e-3f, 1e-3f, why);
            bool ok2 = approx(to_float(ob), ref_nc, 1e-3f, 1e-3f, why);
            bool ok3 = approx(to_float(oa), to_float(ob), 0.0f, 0.0f, why);  // bit-identical
            check("softmax non-causal (span)", ok1);
            check("softmax non-causal (branchy)", ok2);
            check("softmax span == branchy", ok3);
            if (!ok3) std::printf("    why=%s\n", why);
        }
    } catch (const std::exception& e) {
        std::printf("[selfcheck] EXCEPTION: %s\n", e.what());
        ++fails;
    }
    std::printf("[selfcheck] %s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
