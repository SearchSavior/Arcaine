#pragma once
// Silero VAD — matches OnnxWrapper + VADIterator in silero-vad/utils_vad.py
// Removes non-speech audio from 16 kHz mono float32 input.
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <cstdio>

// apply_silero_vad — mirrors the Python pipeline:
//   OnnxWrapper (stateful LSTM forward) + VADIterator (threshold + hysteresis)
//   + collect_chunks (concatenate kept segments)
//
// Parameters match VADIterator defaults:
//   threshold        = 0.5
//   min_silence_ms   = 100   (don't split on silences shorter than this)
//   speech_pad_ms    = 30    (pad each segment on both sides)
inline std::vector<float> apply_silero_vad(
    const std::string& model_path,
    const std::vector<float>& audio,
    float threshold      = 0.5f,
    int   min_silence_ms = 100,
    int   speech_pad_ms  = 30
) {
    constexpr int SR      = 16000;
    constexpr int CHUNK   = 512;      // num_samples for 16 kHz (utils_vad.py: 512 if sr == 16000)
    constexpr int CTX     = 64;       // context_size for 16 kHz
    constexpr int INP_LEN = CTX + CHUNK;  // 576: cat([context, x], dim=1)

    const int min_silence_samp = SR * min_silence_ms / 1000;  // 1600
    const int speech_pad_samp  = SR * speech_pad_ms  / 1000;  // 480

    // --- Session (inter/intra = 1 matches OnnxWrapper) ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "silero_vad");
    Ort::SessionOptions sopts;
    sopts.SetInterOpNumThreads(1);
    sopts.SetIntraOpNumThreads(1);
    Ort::Session session(env, model_path.c_str(), sopts);

    // Query I/O names from the model (avoids hardcoding "stateN" etc.)
    Ort::AllocatorWithDefaultOptions alloc;
    auto in0  = session.GetInputNameAllocated(0, alloc);   // "input"
    auto in1  = session.GetInputNameAllocated(1, alloc);   // "state"
    auto in2  = session.GetInputNameAllocated(2, alloc);   // "sr"
    auto out0 = session.GetOutputNameAllocated(0, alloc);  // "output"
    auto out1 = session.GetOutputNameAllocated(1, alloc);  // "stateN"

    const char* inames[] = { in0.get(), in1.get(), in2.get() };
    const char* onames[] = { out0.get(), out1.get() };

    Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Persistent state: zeros((2, batch=1, 128)) — reset_states in OnnxWrapper
    std::vector<float> state(2 * 1 * 128, 0.0f);
    int64_t state_shape[] = {2, 1, 128};

    // Context: zeros(batch=1, 64) — self._context = torch.zeros(batch, context_size)
    std::vector<float> context(CTX, 0.0f);

    // SR scalar (0-d tensor, np.array(sr, dtype='int64'))
    int64_t sr_val = SR;

    // Input buffer: (1, 576)
    std::vector<float> inp_buf(INP_LEN);
    int64_t inp_shape[]   = {1, INP_LEN};

    // VADIterator state
    bool triggered      = false;
    int  current_sample = 0;
    int  temp_end       = 0;
    int  speech_start   = 0;

    struct Seg { int start, end; };
    std::vector<Seg> segs;

    // Pad to multiple of CHUNK (audio_forward: pad to multiple of num_samples)
    std::vector<float> padded = audio;
    {
        int rem = (int)padded.size() % CHUNK;
        if (rem) padded.insert(padded.end(), CHUNK - rem, 0.0f);
    }

    for (int off = 0; off < (int)padded.size(); off += CHUNK) {
        current_sample += CHUNK;

        // x = cat([context, chunk], dim=1)  shape (1, 576)
        std::memcpy(inp_buf.data(),       context.data(),          CTX   * sizeof(float));
        std::memcpy(inp_buf.data() + CTX, padded.data() + off,     CHUNK * sizeof(float));

        // context = x[..., -context_size:]  → last CTX samples of inp_buf = chunk tail
        std::memcpy(context.data(), inp_buf.data() + CHUNK, CTX * sizeof(float));

        auto t_inp = Ort::Value::CreateTensor<float>(cpu_mem,
            inp_buf.data(), INP_LEN, inp_shape, 2);
        auto t_state = Ort::Value::CreateTensor<float>(cpu_mem,
            state.data(), 2*128, state_shape, 3);
        // sr: 0-d int64 tensor (shape_len=0)
        auto t_sr = Ort::Value::CreateTensor<int64_t>(cpu_mem,
            &sr_val, 1, nullptr, 0);

        std::vector<Ort::Value> iv;
        iv.push_back(std::move(t_inp));
        iv.push_back(std::move(t_state));
        iv.push_back(std::move(t_sr));

        auto ov = session.Run(Ort::RunOptions{nullptr},
            inames, iv.data(), 3,
            onames, 2);

        float prob = *ov[0].GetTensorData<float>();
        std::memcpy(state.data(), ov[1].GetTensorData<float>(), 2*128 * sizeof(float));

        // --- VADIterator.__call__ logic (exact match to Python) ---

        if (prob >= threshold && temp_end)
            temp_end = 0;

        if (prob >= threshold && !triggered) {
            triggered    = true;
            speech_start = std::max(0, current_sample - speech_pad_samp - CHUNK);
        }

        if (prob < (threshold - 0.15f) && triggered) {
            if (!temp_end)
                temp_end = current_sample;
            if (current_sample - temp_end >= min_silence_samp) {
                int speech_end = temp_end + speech_pad_samp - CHUNK;
                speech_end = std::min(speech_end, (int)audio.size());
                if (speech_end > speech_start)
                    segs.push_back({speech_start, speech_end});
                temp_end  = 0;
                triggered = false;
            }
        }
    }

    // Close any open segment at end of stream
    if (triggered) {
        int speech_end = std::min(current_sample + speech_pad_samp, (int)audio.size());
        if (speech_end > speech_start)
            segs.push_back({speech_start, speech_end});
    }

    if (segs.empty()) {
        std::fprintf(stderr, "[vad] no speech detected — returning full audio\n");
        return audio;
    }

    // collect_chunks: concatenate kept segments
    std::vector<float> out;
    int kept = 0;
    for (auto& s : segs) {
        int a = std::max(0, s.start);
        int b = std::min(s.end, (int)audio.size());
        if (b > a) {
            out.insert(out.end(), audio.begin() + a, audio.begin() + b);
            kept += b - a;
        }
    }

    std::fprintf(stderr, "[vad] %d segment(s), kept %.2f s / %.2f s (%.0f%%)\n",
                 (int)segs.size(),
                 kept          / (float)SR,
                 (int)audio.size() / (float)SR,
                 100.0f * kept / (float)audio.size());

    return out;
}
