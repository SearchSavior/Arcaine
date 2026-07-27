#pragma once

namespace arcaine::inference {

struct StreamOptions {
    bool enabled        = false;
    bool stream_drafts  = false;  // diffusion draft-step events
    bool include_usage  = false;  // emit a final usage chunk (OpenAI)
};

}  // namespace arcaine::inference
