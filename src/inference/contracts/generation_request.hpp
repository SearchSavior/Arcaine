#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "inference/contracts/chat_template_options.hpp"
#include "inference/contracts/sampling_options.hpp"
#include "inference/contracts/stream_options.hpp"

namespace arcaine::inference {

class CancellationToken;

// Current serving inputs. The transport layer constructs this; a model
// session resolves it into a private, model-specific invocation. Optional
// fields are used where the request surface admits multiple input forms
// (messages vs. prompt text vs. raw token ids).
//
// `messages` and `tools` are carried as the normalized OpenAI JSON the model
// chat template consumes, so the template renders byte-identically to the
// prior transport path. The transport does not interpret them.
struct GenerationRequest {
    std::string  request_id;
    std::string  served_model_name;

    // One or more input forms. The model picks the form it supports based on
    // its declared input capabilities.
    nlohmann::ordered_json messages = nlohmann::ordered_json::array();  // normalized OpenAI messages
    std::string           prompt_text;      // raw prompt (non-chat)
    std::vector<int>      input_token_ids;  // raw token ids

    // OpenAI tools + tool_choice, serialized as JSON the model chat template
    // consumes. The transport does not interpret these.
    nlohmann::ordered_json tools      = nlohmann::ordered_json::array();
    nlohmann::ordered_json tool_choice;

    // Media paths; the model's request_mapper loads + preprocesses them with
    // model-specific parameters. (The transport does no model-specific
    // media transforms.)
    std::vector<std::string> image_paths;
    std::vector<std::string> audio_paths;
    std::string              vad_model_path;

    StreamOptions       stream;
    SamplingOptions     sampling;
    ChatTemplateOptions chat_template;

    int           max_output_tokens = 0;
    std::uint64_t  seed             = 0;

    // Diffusion-specific request options. Ignored by causal models.
    int  denoising_steps  = -1;   // -1 = use model default
    // (Draft streaming is governed by StreamOptions::stream_drafts — the single
    // authoritative field, set by the transport from `arcaine_stream_drafts`.)

    // Set by the transport when the client disconnects / SSE sink fails /
    // request is cancelled. The model session checks it at model-owned
    // interruption points.
    CancellationToken* cancellation = nullptr;
};

}  // namespace arcaine::inference
