#include "audio_proc.hpp"
#include "vad.hpp"
#include <soxr.h>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cstdint>

AudioInput preprocess_audio(
    const float* waveform, int num_samples,
    int samples_per_frame
) {
    // Mirror of Gemma4UnifiedAudioFeatureExtractor._extract_waveform_features:
    //   pad_len = (-len(waveform)) % audio_samples_per_token
    //   features = waveform.reshape(num_tokens, audio_samples_per_token).astype(np.float32)
    int num_frames = (num_samples + samples_per_frame - 1) / samples_per_frame;

    AudioInput out;
    out.num_frames = num_frames;
    out.frames.resize(num_frames * samples_per_frame, 0.0f);
    std::memcpy(out.frames.data(), waveform, num_samples * sizeof(float));
    return out;
}

// Resample mono float32 using soxr HQ.
// Matches transformers audio_utils.load_audio_as:
//   soxr.resample(audio_array, original_sr, sampling_rate, quality="HQ")
static std::vector<float> resample_to_rate(
    const std::vector<float>& in, int in_rate, int out_rate)
{
    if (in_rate == out_rate) return in;

    size_t out_len = (size_t)((double)in.size() * out_rate / in_rate + 0.5);
    std::vector<float> out(out_len);

    soxr_error_t err;
    size_t done_in = 0, done_out = 0;

    // SOXR_HQ matches Python soxr quality="HQ"
    soxr_quality_spec_t q = soxr_quality_spec(SOXR_HQ, 0);
    soxr_t resampler = soxr_create(
        in_rate, out_rate, 1,
        &err,
        nullptr,  // default I/O format (float32)
        &q,       // HQ quality
        nullptr   // default runtime
    );
    if (err) throw std::runtime_error(std::string("soxr_create: ") + err);

    err = soxr_process(
        resampler,
        in.data(),   in.size(),   &done_in,
        out.data(), out_len,      &done_out
    );
    if (err) { soxr_delete(resampler); throw std::runtime_error(std::string("soxr_process: ") + err); }

    // Flush remaining output (end-of-stream)
    size_t extra_done = 0;
    err = soxr_process(
        resampler,
        nullptr, 0, nullptr,
        out.data() + done_out, out_len - done_out, &extra_done
    );
    soxr_delete(resampler);
    if (err) throw std::runtime_error(std::string("soxr_process flush: ") + err);

    out.resize(done_out + extra_done);
    return out;
}

// ---------------------------------------------------------------------------
// Minimal RIFF/WAV parser.
// Supports PCM int16 (format 1) and IEEE float32 (format 3), any channel count.
// Resamples to 16000 Hz with soxr HQ (matches transformers load_audio_as).
// ---------------------------------------------------------------------------
// Returns raw mono float32 PCM at target_sample_rate.
static std::vector<float> load_wav_mono(const std::string& path, int target_sample_rate) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open: " + path);

    auto rd16 = [&]() -> uint16_t { uint16_t v; std::fread(&v, 2, 1, f); return v; };
    auto rd32 = [&]() -> uint32_t { uint32_t v; std::fread(&v, 4, 1, f); return v; };

    char tag[4];
    std::fread(tag, 1, 4, f);
    if (std::memcmp(tag, "RIFF", 4)) throw std::runtime_error(path + ": not RIFF");
    rd32(); // file size
    std::fread(tag, 1, 4, f);
    if (std::memcmp(tag, "WAVE", 4)) throw std::runtime_error(path + ": not WAVE");

    uint16_t audio_format = 0, num_channels = 0, bits = 0;
    uint32_t sample_rate = 0, data_bytes = 0;
    long data_offset = 0;
    bool found_fmt = false, found_data = false;

    while (!found_data) {
        if (std::fread(tag, 1, 4, f) != 4) break;
        uint32_t chunk = rd32();
        long start = std::ftell(f);

        if (!std::memcmp(tag, "fmt ", 4)) {
            audio_format = rd16();
            num_channels = rd16();
            sample_rate  = rd32();
            rd32(); rd16(); // byte_rate, block_align
            bits         = rd16();
            found_fmt    = true;
        } else if (!std::memcmp(tag, "data", 4)) {
            data_offset  = start;
            data_bytes   = chunk;
            found_data   = true;
            break;
        }
        std::fseek(f, start + (long)((chunk + 1) & ~1u), SEEK_SET);
    }
    if (!found_fmt)  throw std::runtime_error(path + ": missing fmt  chunk");
    if (!found_data) throw std::runtime_error(path + ": missing data chunk");

    std::fprintf(stderr, "[audio] %s: fmt=%u ch=%u rate=%u bits=%u\n",
                 path.c_str(), audio_format, num_channels, sample_rate, bits);

    std::fseek(f, data_offset, SEEK_SET);

    // Decode to interleaved float32
    std::vector<float> interleaved;
    if (audio_format == 1 && bits == 16) {
        int n = (int)(data_bytes / 2);
        std::vector<int16_t> raw(n);
        std::fread(raw.data(), 2, n, f);
        interleaved.resize(n);
        for (int i = 0; i < n; ++i)
            interleaved[i] = raw[i] / 32768.0f;
    } else if (audio_format == 3 && bits == 32) {
        int n = (int)(data_bytes / 4);
        interleaved.resize(n);
        std::fread(interleaved.data(), 4, n, f);
    } else {
        std::fclose(f);
        throw std::runtime_error(path + ": unsupported fmt " +
            std::to_string(audio_format) + "/" + std::to_string(bits) + "bit");
    }
    std::fclose(f);

    // Downmix to mono (matches soundfile + mean-axis=1 in transformers)
    int step = (int)num_channels;
    int n_mono = (int)interleaved.size() / step;
    std::vector<float> mono(n_mono);
    for (int i = 0; i < n_mono; ++i) {
        float s = 0.0f;
        for (int c = 0; c < step; ++c)
            s += interleaved[i * step + c];
        mono[i] = s / (float)step;
    }

    // Resample to the processor-configured rate with soxr HQ.
    if ((int)sample_rate != target_sample_rate) {
        std::fprintf(stderr, "[audio] resampling %u -> %d Hz (%d -> ",
                     sample_rate, target_sample_rate, (int)mono.size());
        mono = resample_to_rate(mono, (int)sample_rate, target_sample_rate);
        std::fprintf(stderr, "%d samples)\n", (int)mono.size());
    }

    return mono;
}

AudioInput preprocess_audio_file(const std::string& path, int samples_per_frame,
                                   int target_sample_rate,
                                   const std::string& vad_model) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open: " + path);
    char magic[4] = {};
    std::fread(magic, 1, 4, f);
    std::fclose(f);

    std::vector<float> mono;

    if (!std::memcmp(magic, "RIFF", 4)) {
        mono = load_wav_mono(path, target_sample_rate);
    } else {
        // Legacy: raw float32 PCM at the processor-configured sample rate.
        f = std::fopen(path.c_str(), "rb");
        std::fseek(f, 0, SEEK_END);
        long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        mono.resize(sz / sizeof(float));
        std::fread(mono.data(), sizeof(float), mono.size(), f);
        std::fclose(f);
    }

    if (!vad_model.empty())
        mono = apply_silero_vad(vad_model, mono);

    return preprocess_audio(mono.data(), (int)mono.size(), samples_per_frame);
}
