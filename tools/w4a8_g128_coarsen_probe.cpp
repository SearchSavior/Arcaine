// Host-side probe: how much weight-dequantization error does coarsening the
// AWQ per-32-group weight scales to per-128-group introduce? This is the
// accuracy side of the "group 32 -> 128 gives +30% TF" finding — and it is
// doable WITHOUT the private calibration dataset, because we only merge the
// existing g32 scales (the s4 weight nibbles stay fixed).
//
// Dequant (g32, reference):  w[n,k] = scale[n, k/32] * (q_u[n,k] - zp_u[n,k/32])
// Dequant (g128, coarsened): w[n,k] = S[n, k/128] * (q_u[n,k] - ZP[n, k/128])
//
// For each (channel, 128-block) we fit the OPTIMAL affine map y ~ a*x + b over
// the 128 fixed s4 weights (x = q_u, y = g32 target). a*x+b = S*(x - ZP) with
// S=a, ZP=-b/a. This is the best a g128 coarsening can do given fixed nibbles,
// i.e. a PESSIMISTIC lower bound (a true g128 requant could re-pick nibbles).
//
// Build (host, no SYCL/oneDNN):
//   g++ -O2 -std=c++17 -I src -I third_party \
//       tools/w4a8_g128_coarsen_probe.cpp -o /tmp/g128_probe
//
// Run (inside container):
//   /tmp/g128_probe /workspace/models/cyankiwi_Qwen3.6-27B-AWQ-INT4 \
//       [tensor-name ...]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "runtime/quantization/safetensors.hpp"
#include "nlohmann/json.hpp"

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) bits = sign | 0x7F800000 | (mant << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float out; std::memcpy(&out, &bits, 4); return out;
}

static float bf16_to_f32(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float out; std::memcpy(&out, &bits, 4); return out;
}

struct Reader {
    std::vector<std::unique_ptr<SafetensorsFile>> shards;
    std::map<std::string, std::pair<int, TensorView>> idx;
    int num_shards = 0;

    // index.json: {"weight_map": {name -> "model-XXXX-of-YYYY.safetensors"}}
    void load(const std::string& dir) {
        // discover shard files from index.json weight_map
        nlohmann::json j;
        {
            FILE* f = std::fopen((dir + "/model.safetensors.index.json").c_str(), "rb");
            if (!f) { std::fprintf(stderr, "no index.json in %s\n", dir.c_str()); std::exit(1); }
            std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::rewind(f);
            std::string buf(sz, '\0');
            std::fread(&buf[0], 1, sz, f); std::fclose(f);
            j = nlohmann::json::parse(buf);
        }
        const auto& wm = j["weight_map"];
        // collect unique filenames in order.
        std::vector<std::string> files;
        std::map<std::string, int> fid;
        for (auto it = wm.begin(); it != wm.end(); ++it) {
            std::string file = it.value().get<std::string>();
            if (!fid.count(file)) { fid[file] = (int)files.size(); files.push_back(file); }
        }
        num_shards = (int)files.size();
        for (const auto& f : files) {
            shards.push_back(std::make_unique<SafetensorsFile>(dir + "/" + f));
        }
        // map every tensor name to (shard, view)
        for (int s = 0; s < num_shards; ++s) {
            for (const auto& kv : shards[s]->all())
                idx[kv.first] = {s, kv.second};
        }
    }

    const TensorView& get(const std::string& name) const {
        auto it = idx.find(name);
        if (it == idx.end()) { std::fprintf(stderr, "missing tensor %s\n", name.c_str()); std::exit(1); }
        return it->second.second;
    }
};

// Read a per-channel F32/F16/BF16 scale tensor [N,G] into a flat float vector
// indexed scale[n*G+g].
static std::vector<float> read_scale(const TensorView& tv, int N, int G) {
    std::vector<float> out((size_t)N * G);
    for (size_t i = 0; i < out.size(); ++i) {
        if (tv.dtype == "F32") out[i] = ((const float*)tv.data)[i];
        else if (tv.dtype == "F16") out[i] = f16_to_f32(((const uint16_t*)tv.data)[i]);
        else if (tv.dtype == "BF16") out[i] = bf16_to_f32(((const uint16_t*)tv.data)[i]);
        else { std::fprintf(stderr, "bad scale dtype %s\n", tv.dtype.c_str()); std::exit(1); }
    }
    return out;
}

struct Stats {
    double sum_y2 = 0.0, sum_r2 = 0.0, max_abs = 0.0, max_rel = 0.0;
    double sum_cos = 0.0, min_cos = 2.0;
    long n = 0, nchan = 0;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model_dir> [tensor-name ...]\n", argv[0]);
        return 2;
    }
    std::string dir = argv[1];
    std::vector<std::string> tensors;
    for (int i = 2; i < argc; ++i) tensors.push_back(argv[i]);
    if (tensors.empty()) {
        const char* defaults[] = {
            "model.language_model.layers.0.mlp.down_proj",
            "model.language_model.layers.0.mlp.gate_proj",
            "model.language_model.layers.0.mlp.up_proj",
            "model.language_model.layers.11.self_attn.q_proj",
            "model.language_model.layers.11.self_attn.o_proj",
            "model.language_model.layers.32.mlp.down_proj",
        };
        for (auto d : defaults) tensors.push_back(d);
    }

    Reader r;
    r.load(dir);

    for (const auto& prefix : tensors) {
        const TensorView& packed = r.get(prefix + ".weight_packed");
        const TensorView& scale  = r.get(prefix + ".weight_scale");
        const TensorView& zp     = r.get(prefix + ".weight_zero_point");

        int N = (int)packed.shape[0];
        int Kp = (int)packed.shape[1];   // K/8
        int K = Kp * 8;
        int G = (int)scale.shape[1];
        int gsz = K / G;                  // 32

        const int32_t* pk = (const int32_t*)packed.data;
        const int32_t* zpp = (const int32_t*)zp.data;
        std::vector<float> sc = read_scale(scale, N, G);

        std::printf("=== %s  N=%d K=%d G=%d group_size=%d\n",
                    prefix.c_str(), N, K, G, gsz);

        // --- basic scale / zp stats + intra-128 flatness --------------------
        double s_min = 1e30, s_max = -1e30, s_sum = 0.0;
        for (double v : sc) { s_min = std::min(s_min, v); s_max = std::max(s_max, v); s_sum += v; }
        std::printf("  scale: min=%.4g max=%.4g mean=%.4g\n",
                    s_min, s_max, s_sum / (double)sc.size());

        long zp_is8 = 0, zp_total = 0;
        double zp_min = 16, zp_max = -1;
        for (int n = 0; n < N; ++n) {
            int zrow = n >> 3, zsh = (n & 7) * 4;
            for (int g = 0; g < G; ++g) {
                int z = (zpp[(size_t)zrow * G + g] >> zsh) & 0xF;
                if (z == 8) zp_is8++;
                zp_total++;
                zp_min = std::min(zp_min, (double)z); zp_max = std::max(zp_max, (double)z);
            }
        }
        std::printf("  zero_point: %.1f%% ==8  min=%g max=%g\n",
                    100.0 * zp_is8 / zp_total, zp_min, zp_max);

        // intra-128 scale flatness: ratio max/min of the 4 g32 scales in each block
        if (gsz <= 128 && 128 % gsz == 0) {
            int per = 128 / gsz;  // groups per 128-block
            std::vector<double> ratios;
            for (int n = 0; n < N; ++n)
                for (int b = 0; b < G; b += per) {
                    double mn = 1e30, mx = -1e30;
                    for (int j = 0; j < per; ++j) {
                        double v = sc[(size_t)n * G + b + j];
                        mn = std::min(mn, v); mx = std::max(mx, v);
                    }
                    if (mn > 0) ratios.push_back(mx / mn);
                }
            std::sort(ratios.begin(), ratios.end());
            std::printf("  intra-128 scale max/min ratio: mean=%.4g p50=%.4g p99=%.4g max=%.4g\n",
                        std::accumulate(ratios.begin(), ratios.end(), 0.0) / ratios.size(),
                        ratios[ratios.size()/2],
                        ratios[(size_t)(ratios.size()*0.99)],
                        ratios.back());
        }

        // --- optimal g128 coarsening error --------------------------------
        Stats st;
        int per = gsz;   // group size 32; a 128-block = 4 groups
        int g128 = 128 / gsz;
        if (gsz == 32) {
            for (int n = 0; n < N; ++n) {
                // per-channel accumulators
                double ch_y2 = 0.0, ch_r2 = 0.0, ch_yy = 0.0, ch_y2sum = 0.0, ch_r2sum = 0.0;
                double ch_max = 0.0;
                for (int b = 0; b < G; b += g128) {
                    // gather 128 points
                    double sx = 0, sy = 0, sxx = 0, sxy = 0;
                    double y[128]; int x[128];
                    int cnt = 0;
                    for (int j = 0; j < g128; ++j) {
                        int g = b + j;
                        double s = sc[(size_t)n * G + g];
                        int zrow = n >> 3, zsh = (n & 7) * 4;
                        int z = (zpp[(size_t)zrow * G + g] >> zsh) & 0xF;
                        for (int kk = 0; kk < gsz; ++kk) {
                            int k = g * gsz + kk;
                            int q = (pk[(size_t)n * Kp + (k >> 3)] >> ((k & 7) * 4)) & 0xF;
                            double v = s * (q - z);
                            x[cnt] = q; y[cnt] = v;
                            sx += q; sy += v; sxx += (double)q * q; sxy += q * v;
                            cnt++;
                        }
                    }
                    double denom = cnt * sxx - sx * sx;
                    double a = 0, b0 = 0;
                    if (denom > 1e-9) {
                        a = (cnt * sxy - sx * sy) / denom;
                        b0 = (sy - a * sx) / cnt;
                    } else {
                        // degenerate (constant q); just use mean
                        b0 = sy / cnt; a = 0;
                    }
                    for (int j = 0; j < cnt; ++j) {
                        double yhat = a * x[j] + b0;
                        double e = y[j] - yhat;
                        double yy = y[j] * y[j];
                        st.sum_y2 += yy; st.sum_r2 += e * e;
                        st.max_abs = std::max(st.max_abs, std::fabs(e));
                        ch_y2 += yy; ch_r2 += e * e; ch_yy += y[j] * y[j];
                        ch_y2sum += yy; ch_r2sum += e * e;
                        ch_max = std::max(ch_max, std::fabs(y[j]));
                    }
                }
                st.nchan++;
                st.n += 128 * (G / g128);
                if (ch_max > 0)
                    st.max_rel = std::max(st.max_rel, std::sqrt(ch_r2) / (std::sqrt(ch_y2) + 1e-30));
            }
        }

        double rel_rms = std::sqrt(st.sum_r2 / (st.sum_y2 + 1e-30));
        std::printf("  g128 coarsen (optimal affine fit): rel_rms=%.4g  "
                    "max_abs=%.4g  max_channel_rel=%.4g\n",
                    rel_rms, st.max_abs, st.max_rel);
        std::printf("\n");
    }
    return 0;
}
