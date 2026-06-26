#include "audio_embedder.hpp"
#include "../../common/gpu/engine.hpp"
#include "../../common/gpu/ops.hpp"
#include "../../common/kernels/rms_norm.hpp"
#include <stdexcept>

GpuBuffer<bf16> audio_embedder_forward(
    const AudioWeights& w,
    const AudioInput& audio,
    const AudioConfig& cfg,
    int hidden_size
) {
    auto& q  = GpuEngine::get().queue;
    int N    = audio.num_frames;
    int dim  = cfg.audio_samples_per_token;
    int H    = hidden_size;
    if ((int)audio.frames.size() != N * dim)
        throw std::runtime_error("Audio frame buffer does not match audio_samples_per_token");

    // Upload frames (N, audio_samples_per_token) as BF16
    GpuBuffer<bf16> x(N * dim);
    {
        std::vector<bf16> buf(N * dim);
        for (int i = 0; i < N * dim; ++i)
            buf[i] = float_to_bf16(audio.frames[i]);
        x.upload(buf.data(), N * dim);
    }

    // Pre-projection RMSNorm — identity scale
    rms_norm(q, x.data(), nullptr, x.data(), N, dim, cfg.rms_norm_eps);

    // Linear projection: (N, audio_samples_per_token) @ proj_w.T -> (N, H)
    GpuBuffer<bf16> out(N * H);
    matmul_bf16(x.data(), N, dim, w.proj_w.data(), H, out.data());

    return out;
}
