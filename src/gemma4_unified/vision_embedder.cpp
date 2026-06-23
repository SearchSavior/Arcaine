#include "vision_embedder.hpp"
#include "../common/gpu/engine.hpp"
#include "../common/gpu/ops.hpp"
#include "../common/kernels/layer_norm.hpp"
#include "../common/kernels/rms_norm.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>

// SYCL kernel: gather factorized 2D positional embeddings.
// table: (posemb_size, 2, H)  pos_ids: (N, 2)  out: (N, H)
// Patches with pos_ids == -1 get zero output.
static void gather_pos_embedding(
    sycl::queue& q,
    const bf16* table,
    const int32_t* pos_ids,
    bf16* out,
    int N, int posemb_size, int H
) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(N, H), [=](sycl::id<2> id) {
            int i = id[0], d = id[1];
            int px = pos_ids[i * 2 + 0];
            int py = pos_ids[i * 2 + 1];
            if (px < 0 || py < 0) {
                out[i * H + d] = float_to_bf16(0.0f);
                return;
            }
            // table layout: (posemb_size, 2, H)
            // axis 0: px   axis 1: py
            float vx = bf16_to_float(table[(px * 2 + 0) * H + d]);
            float vy = bf16_to_float(table[(py * 2 + 1) * H + d]);
            out[i * H + d] = float_to_bf16(vx + vy);
        });
    }).wait();
}

GpuBuffer<bf16> vision_embedder_forward(
    const VisionWeights& w,
    const ImageInput& img,
    const VisionConfig& cfg,
    int hidden_size
) {
    auto& gpu = GpuEngine::get();
    auto& q   = gpu.queue;

    int N          = img.max_patches;
    int patch_dim  = cfg.model_patch_size * cfg.model_patch_size * 3;
    int H          = hidden_size;
    int posemb_sz  = cfg.mm_posemb_size;
    if ((int)img.pixel_values.size() != N * patch_dim)
        throw std::runtime_error("Image patch buffer does not match vision_config model_patch_size");

    // Upload pixel_values (N, 6912) from float32 to BF16
    GpuBuffer<bf16> x(N * patch_dim);
    {
        std::vector<bf16> px_bf16(N * patch_dim);
        for (int i = 0; i < N * patch_dim; ++i)
            px_bf16[i] = float_to_bf16(img.pixel_values[i]);
        x.upload(px_bf16.data(), N * patch_dim);
    }

    // Upload position_ids (N, 2) as int32
    GpuBuffer<int32_t> pos_ids_dev(N * 2);
    {
        std::vector<int32_t> pid(N * 2);
        for (int i = 0; i < N; ++i) {
            pid[i * 2 + 0] = img.position_ids[i][0];
            pid[i * 2 + 1] = img.position_ids[i][1];
        }
        pos_ids_dev.upload(pid.data(), N * 2);
    }

    // patch_ln1: LayerNorm (N, 6912)
    layer_norm(q, x.data(), w.patch_ln1_w.data(), w.patch_ln1_b.data(),
               x.data(), N, patch_dim, cfg.rms_norm_eps);

    // patch_dense: (N, patch_dim) @ patch_dense_w.T + bias -> (N, H)
    GpuBuffer<bf16> y(N * H);
    matmul_bf16(x.data(), N, patch_dim, w.patch_dense_w.data(), H, y.data());

    // Add bias
    q.submit([&](sycl::handler& h) {
        const bf16* bias = w.patch_dense_b.data();
        bf16* yp = y.data();
        h.parallel_for(sycl::range<2>(N, H), [=](sycl::id<2> id) {
            int i = id[0], d = id[1];
            yp[i * H + d] = float_to_bf16(bf16_to_float(yp[i * H + d]) +
                                           bf16_to_float(bias[d]));
        });
    }).wait();

    // patch_ln2: LayerNorm (N, H)
    layer_norm(q, y.data(), w.patch_ln2_w.data(), w.patch_ln2_b.data(),
               y.data(), N, H, cfg.rms_norm_eps);

    // Factorized 2D positional embedding
    GpuBuffer<bf16> pos_emb(N * H);
    gather_pos_embedding(q, w.pos_embedding.data(), pos_ids_dev.data(),
                         pos_emb.data(), N, posemb_sz, H);

    // y = y + pos_emb
    q.submit([&](sycl::handler& h) {
        bf16* yp  = y.data();
        const bf16* pe = pos_emb.data();
        int total = N * H;
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
            yp[id[0]] = float_to_bf16(bf16_to_float(yp[id[0]]) + bf16_to_float(pe[id[0]]));
        });
    }).wait();

    // pos_norm: LayerNorm (N, H)
    layer_norm(q, y.data(), w.pos_norm_w.data(), w.pos_norm_b.data(),
               y.data(), N, H, cfg.rms_norm_eps);

    // Pre-projection RMSNorm — identity scale (weights not in safetensors)
    rms_norm(q, y.data(), nullptr, y.data(), N, H, cfg.rms_norm_eps);

    // Final linear projection: (N, H) @ proj_w.T → (N, H)
    GpuBuffer<bf16> out_all(N * H);
    matmul_bf16(y.data(), N, H, w.proj_w.data(), H, out_all.data());

    // Strip padding: extract first num_valid_patches rows
    int nv = img.num_valid_patches;
    GpuBuffer<bf16> out_valid(nv * H);
    q.memcpy(out_valid.data(), out_all.data(), nv * H * sizeof(bf16)).wait();

    return out_valid;
}
