#pragma once
#include <string>
#include <vector>
#include "../modeling/diffusion_gemma/model.hpp"

class TokenizerBridge;

// VRAM snapshot for one GPU (device-reported, taken after generation).
struct DiffGpuMem {
    double used_gb  = 0.0;
    double total_gb = 0.0;
};

// Records every denoising step during generate() and writes a self-contained
// HTML replay: a step slider/autoplay where each canvas token is a span whose
// background encodes entropy (red = noise, clear = converged), with accepted
// tokens underlined and committed blocks accumulating as plain text.
class DiffVizRecorder {
public:
    // Called from the generation stream callback.
    void record(const DiffStepEvent& ev);

    // Writes the replay. Uses `tok` to fetch per-token surface strings.
    void write_html(const std::string& path, const std::string& prompt,
                    TokenizerBridge& tok, const DiffPerfStats& perf,
                    const std::vector<DiffGpuMem>& gpu_mem) const;

private:
    struct Frame {
        int block, cur_step;
        float temperature, mean_entropy;
        bool committed;
        std::vector<int>   ids;
        std::vector<float> entropy;   // empty on committed frames
        std::vector<char>  accepted;  // empty on committed frames
    };
    std::vector<Frame> frames_;
};
