#pragma once
#include <vector>
#include <string>

struct AudioInput {
    std::vector<float> frames;  // (num_frames, 640) — raw waveform chunks
    int                num_frames;
};

// Chunk raw waveform samples into 640-sample frames.
// waveform: PCM float32 at the processor-configured sample rate.
AudioInput preprocess_audio(
    const float* waveform, int num_samples,
    int samples_per_frame
);

// Load an audio file (WAV or raw float32 PCM), resample to target_sample_rate,
// optionally apply Silero VAD to strip silence, then chunk into frames.
// vad_model: path to silero_vad.onnx; pass "" to skip VAD.
AudioInput preprocess_audio_file(const std::string& path,
                                  int samples_per_frame,
                                  int target_sample_rate,
                                  const std::string& vad_model = "");
