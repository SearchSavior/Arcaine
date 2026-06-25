// OpenAI-compatible HTTP server for DiffusionGemma.
//   ./build/diffusion_server --model <dir> [--host 127.0.0.1] [--port 8000]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "common/gpu/placement.hpp"
#include "diffusion_gemma/model.hpp"
#include "utils/chat.hpp"
#include "utils/chat_template_kwargs.hpp"
#include "utils/tool_call_parser.hpp"

using json = nlohmann::ordered_json;

namespace {

struct ServerOptions {
    std::string model_dir;
    std::string served_model_name;
    std::string host = "127.0.0.1";
    int port = 8000;
    int max_seq = 2048;
    int default_max_tokens = 256;
    int steps = -1;
    unsigned seed = 42;
    bool print_placement = true;
    DiffPlacementOptions placement;
    json chat_template_kwargs = json::object();
};

struct ChatRequest {
    std::string model;
    json messages;
    json tools = json::array();
    bool stream = false;
    bool stream_drafts = false;
    bool include_usage = false;
    bool has_tools = false;
    int max_tokens = 0;
    int denoising_steps = -1;
    unsigned seed = 0;
};

struct ResponseMetrics {
    int input_token = 0;
    int new_token = 0;
    double ttft = 0.0;                // seconds
    double tpot = 0.0;                // milliseconds/token after first token
    double prefill_throughput = 0.0;  // tokens/second
    double decode_throuput = 0.0;     // tokens/second; key spelling matches API contract
    double duration = 0.0;            // seconds
};

struct OpenAiError : std::runtime_error {
    int status;
    std::string type;
    std::string code;

    OpenAiError(int s, std::string t, std::string c, const std::string& msg)
        : std::runtime_error(msg), status(s), type(std::move(t)), code(std::move(c)) {}
};

struct AppState {
    ServerOptions opts;
    std::time_t created = std::time(nullptr);
    std::string api_key;
    TokenizerBridge tokenizer;
    DiffusionGemmaModel model;
    std::mutex generate_mu;

    AppState(ServerOptions o)
        : opts(std::move(o)),
          api_key(std::getenv("ARCAINE_API_KEY") ? std::getenv("ARCAINE_API_KEY") : ""),
          tokenizer(opts.model_dir),
          model(opts.model_dir, opts.max_seq, opts.placement, opts.print_placement) {}
};

std::string basename_of(std::string path) {
    while (path.size() > 1 && path.back() == '/') path.pop_back();
    size_t pos = path.find_last_of('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

json error_body(const std::string& message, const std::string& type,
                const std::string& code) {
    return {
        {"error", {
            {"message", message},
            {"type", type},
            {"param", nullptr},
            {"code", code},
        }}
    };
}

void set_json(httplib::Response& res, int status, const json& body) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

std::string now_string() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return buf;
}

void log_line(const char* level, const std::string& message) {
    std::fprintf(stderr, "[api][%s][%s] %s\n", now_string().c_str(), level, message.c_str());
    std::fflush(stderr);
}

std::string request_label(const httplib::Request& req) {
    return req.method + " " + req.path + " from " + req.remote_addr;
}

void send_openai_error(httplib::Response& res, const OpenAiError& e) {
    set_json(res, e.status, error_body(e.what(), e.type, e.code));
}

void send_auth_error(httplib::Response& res, const httplib::Request& req) {
    log_line("error", request_label(req) + " -> HTTP 401 invalid_api_key");
    set_json(res, 401, error_body("invalid or missing bearer token",
                                  "authentication_error", "invalid_api_key"));
}

[[noreturn]] void bad_request(const std::string& message,
                              const std::string& code = "invalid_request_error") {
    throw OpenAiError(400, "invalid_request_error", code, message);
}

[[noreturn]] void unsupported_request(const std::string& message) {
    throw OpenAiError(400, "unsupported_request", "unsupported_request", message);
}

void require_bool(const json& body, const char* key, bool& out) {
    if (!body.contains(key)) return;
    if (!body.at(key).is_boolean()) bad_request(std::string(key) + " must be a boolean");
    out = body.at(key).get<bool>();
}

int require_positive_int(const json& body, const char* key, int fallback) {
    if (!body.contains(key)) return fallback;
    if (!body.at(key).is_number_integer()) bad_request(std::string(key) + " must be an integer");
    int value = body.at(key).get<int>();
    if (value <= 0) bad_request(std::string(key) + " must be greater than 0");
    return value;
}

unsigned require_seed(const json& body, unsigned fallback) {
    if (!body.contains("seed")) return fallback;
    if (!body.at("seed").is_number_unsigned() && !body.at("seed").is_number_integer())
        bad_request("seed must be an integer");
    long long value = body.at("seed").get<long long>();
    if (value < 0) bad_request("seed must be non-negative");
    return (unsigned)value;
}

json validate_tools(const json& body) {
    if (!body.contains("tools")) return json::array();
    if (!body.at("tools").is_array()) bad_request("tools must be an array");

    json tools = json::array();
    for (const auto& tool : body.at("tools")) {
        if (!tool.is_object()) bad_request("each tool must be an object");
        if (tool.value("type", std::string()) != "function")
            unsupported_request("only function tools are supported");
        if (!tool.contains("function") || !tool.at("function").is_object())
            bad_request("function tools must include a function object");
        const auto& fn = tool.at("function");
        if (!fn.contains("name") || !fn.at("name").is_string())
            bad_request("function.name must be a string");
        if (fn.contains("description") && !fn.at("description").is_string())
            bad_request("function.description must be a string");
        if (fn.contains("parameters") && !fn.at("parameters").is_object())
            bad_request("function.parameters must be an object");

        json normalized = tool;
        if (!normalized["function"].contains("description"))
            normalized["function"]["description"] = "";
        if (!normalized["function"].contains("parameters"))
            normalized["function"]["parameters"] = {{"type", "object"}, {"properties", json::object()}};
        tools.push_back(std::move(normalized));
    }
    return tools;
}

json apply_tool_choice(const json& body, json tools) {
    if (!body.contains("tool_choice")) return tools;
    const json& choice = body.at("tool_choice");
    if (choice.is_string()) {
        const std::string v = choice.get<std::string>();
        if (v == "auto" || v == "required") return tools;
        if (v == "none") return json::array();
        unsupported_request("tool_choice string must be auto, none, or required");
    }

    if (!choice.is_object()) bad_request("tool_choice must be a string or object");
    if (choice.value("type", std::string()) != "function")
        unsupported_request("only function tool_choice is supported");
    if (!choice.contains("function") || !choice.at("function").is_object() ||
        !choice.at("function").contains("name") || !choice.at("function").at("name").is_string()) {
        bad_request("function tool_choice must include function.name");
    }

    std::string chosen = choice.at("function").at("name").get<std::string>();
    json filtered = json::array();
    for (const auto& tool : tools) {
        if (tool.at("function").at("name").get<std::string>() == chosen)
            filtered.push_back(tool);
    }
    if (filtered.empty())
        bad_request("tool_choice references a function that is not present in tools");
    return filtered;
}

void reject_unsupported_fields(const json& body) {
    auto reject_present = [&](const char* key) {
        if (body.contains(key)) unsupported_request(std::string(key) + " is not supported by this server");
    };

    reject_present("response_format");
    reject_present("top_logprobs");
    reject_present("stop");
    reject_present("functions");
    reject_present("function_call");

    if (body.contains("logprobs")) {
        if (!body.at("logprobs").is_boolean() || body.at("logprobs").get<bool>())
            unsupported_request("logprobs is not supported by this server");
    }
    if (body.contains("n")) {
        if (!body.at("n").is_number_integer()) bad_request("n must be an integer");
        if (body.at("n").get<int>() != 1)
            unsupported_request("n greater than 1 is not supported by this server");
    }
}

std::string parse_content(const json& content) {
    if (content.is_string()) return content.get<std::string>();
    if (!content.is_array())
        bad_request("message content must be a string or an array of text parts");

    std::string out;
    for (const auto& part : content) {
        if (!part.is_object() || !part.contains("type") || !part.at("type").is_string())
            bad_request("message content parts must be objects with a string type");
        const std::string type = part.at("type").get<std::string>();
        if (type != "text")
            unsupported_request("only text message content is supported");
        if (!part.contains("text") || !part.at("text").is_string())
            bad_request("text content parts must contain a string text field");
        out += part.at("text").get<std::string>();
    }
    return out;
}

json parse_tool_calls_field(const json& tool_calls) {
    if (!tool_calls.is_array()) bad_request("message.tool_calls must be an array");
    json out = json::array();
    for (const auto& call : tool_calls) {
        if (!call.is_object()) bad_request("each tool call must be an object");
        if (call.value("type", std::string()) != "function")
            unsupported_request("only function tool calls are supported in messages");
        if (!call.contains("function") || !call.at("function").is_object())
            bad_request("tool call must contain a function object");
        const auto& fn = call.at("function");
        if (!fn.contains("name") || !fn.at("name").is_string())
            bad_request("tool call function.name must be a string");
        if (!fn.contains("arguments"))
            bad_request("tool call function.arguments is required");
        std::string arguments = fn.at("arguments").is_string()
            ? fn.at("arguments").get<std::string>()
            : fn.at("arguments").dump();

        json normalized = {
            {"type", "function"},
            {"function", {
                {"name", fn.at("name").get<std::string>()},
                {"arguments", arguments},
            }},
        };
        if (call.contains("id") && call.at("id").is_string())
            normalized["id"] = call.at("id").get<std::string>();
        out.push_back(std::move(normalized));
    }
    return out;
}

json parse_messages(const json& body) {
    if (!body.contains("messages") || !body.at("messages").is_array())
        bad_request("messages must be an array");

    json messages = json::array();
    for (const auto& item : body.at("messages")) {
        if (!item.is_object()) bad_request("each message must be an object");
        if (!item.contains("role") || !item.at("role").is_string())
            bad_request("each message must contain a string role");
        std::string role = item.at("role").get<std::string>();

        json message = {{"role", role}};
        if (item.contains("content") && !item.at("content").is_null()) {
            message["content"] = json::array({{{"type", "text"},
                                               {"text", parse_content(item.at("content"))}}});
        } else {
            message["content"] = json::array({{{"type", "text"}, {"text", ""}}});
        }

        if (role == "assistant" && item.contains("tool_calls"))
            message["tool_calls"] = parse_tool_calls_field(item.at("tool_calls"));
        if (role == "tool") {
            if (!item.contains("tool_call_id") || !item.at("tool_call_id").is_string())
                bad_request("tool messages must include tool_call_id");
            message["tool_call_id"] = item.at("tool_call_id").get<std::string>();
        }
        messages.push_back(std::move(message));
    }
    if (messages.empty()) bad_request("messages must contain at least one message");
    return messages;
}

ChatRequest parse_chat_request(const httplib::Request& req, const AppState& app) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception& e) {
        bad_request(std::string("request body is not valid JSON: ") + e.what());
    }
    if (!body.is_object()) bad_request("request body must be a JSON object");

    reject_unsupported_fields(body);

    if (!body.contains("model") || !body.at("model").is_string())
        bad_request("model must be a string");
    ChatRequest parsed;
    parsed.model = body.at("model").get<std::string>();
    if (parsed.model != app.opts.served_model_name)
        throw OpenAiError(404, "invalid_request_error", "model_not_found",
                          "model '" + parsed.model + "' is not served by this server");

    parsed.messages = parse_messages(body);
    parsed.tools = apply_tool_choice(body, validate_tools(body));
    parsed.has_tools = !parsed.tools.empty();
    require_bool(body, "stream", parsed.stream);
    require_bool(body, "arcaine_stream_drafts", parsed.stream_drafts);

    parsed.max_tokens = require_positive_int(body, "max_tokens", app.opts.default_max_tokens);
    parsed.max_tokens = require_positive_int(body, "max_completion_tokens", parsed.max_tokens);
    parsed.denoising_steps = require_positive_int(body, "arcaine_denoising_steps",
                                                  app.opts.steps > 0 ? app.opts.steps
                                                                     : app.model.config().gen.max_denoising_steps);
    parsed.seed = require_seed(body, app.opts.seed);

    if (body.contains("stream_options")) {
        if (!body.at("stream_options").is_object())
            bad_request("stream_options must be an object");
        const auto& options = body.at("stream_options");
        if (options.contains("include_usage")) {
            if (!options.at("include_usage").is_boolean())
                bad_request("stream_options.include_usage must be a boolean");
            parsed.include_usage = options.at("include_usage").get<bool>();
        }
    }
    return parsed;
}

bool authorized(const httplib::Request& req, const AppState& app) {
    if (app.api_key.empty()) return true;
    std::string expected = "Bearer " + app.api_key;
    return req.get_header_value("Authorization") == expected;
}

bool write_sse(httplib::DataSink& sink, const json& data, const char* event = nullptr) {
    std::string payload;
    if (event) {
        payload += "event: ";
        payload += event;
        payload += "\n";
    }
    payload += "data: ";
    payload += data.dump();
    payload += "\n\n";
    return sink.write(payload.data(), payload.size());
}

bool write_sse_done(httplib::DataSink& sink) {
    static const char done[] = "data: [DONE]\n\n";
    return sink.write(done, sizeof(done) - 1);
}

json usage_json(int prompt_tokens, int completion_tokens) {
    return {
        {"prompt_tokens", prompt_tokens},
        {"completion_tokens", completion_tokens},
        {"total_tokens", prompt_tokens + completion_tokens},
    };
}

double round4(double value) {
    if (!std::isfinite(value)) return 0.0;
    return std::round(value * 10000.0) / 10000.0;
}

ResponseMetrics make_metrics(int input_tokens, int new_tokens, double ttft_s,
                             double duration_s, const DiffPerfStats& stats) {
    ResponseMetrics m;
    m.input_token = input_tokens;
    m.new_token = new_tokens;
    m.ttft = round4(new_tokens > 0 ? ttft_s : 0.0);
    m.tpot = round4(new_tokens > 1 ? ((duration_s - ttft_s) * 1000.0) / (new_tokens - 1) : 0.0);
    m.prefill_throughput = round4(stats.prefill_s > 0 ? input_tokens / stats.prefill_s : 0.0);
    m.decode_throuput = round4(stats.decode_s > 0 ? new_tokens / stats.decode_s : 0.0);
    m.duration = round4(duration_s);
    return m;
}

json metrics_json(const ResponseMetrics& m) {
    return {
        {"input_token", m.input_token},
        {"new_token", m.new_token},
        {"ttft", m.ttft},
        {"tpot", m.tpot},
        {"prefill_throughput", m.prefill_throughput},
        {"decode_throuput", m.decode_throuput},
        {"duration", m.duration},
    };
}

std::string metrics_log(const ResponseMetrics& m) {
    return "input_token=" + std::to_string(m.input_token) +
           " new_token=" + std::to_string(m.new_token) +
           " ttft=" + std::to_string(m.ttft) +
           " tpot_ms=" + std::to_string(m.tpot) +
           " prefill_tps=" + std::to_string(m.prefill_throughput) +
           " decode_tps=" + std::to_string(m.decode_throuput) +
           " duration=" + std::to_string(m.duration);
}

json chat_completion_chunk(const std::string& id, std::time_t created,
                           const std::string& model, json delta,
                           json finish_reason = nullptr, json usage = nullptr,
                           json metrics = nullptr) {
    return {
        {"id", id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model},
        {"choices", json::array({{
            {"index", 0},
            {"delta", std::move(delta)},
            {"finish_reason", std::move(finish_reason)},
        }})},
        {"usage", std::move(usage)},
        {"metrics", std::move(metrics)},
    };
}

std::string completion_id() {
    static std::atomic<unsigned long long> seq{0};
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return "chatcmpl-arcaine-" + std::to_string((unsigned long long)now) +
           "-" + std::to_string(seq.fetch_add(1));
}

std::string finish_reason_for(size_t generated_tokens, int max_tokens) {
    return generated_tokens >= (size_t)max_tokens ? "length" : "stop";
}

json models_response(const AppState& app) {
    return {
        {"object", "list"},
        {"data", json::array({{
            {"id", app.opts.served_model_name},
            {"object", "model"},
            {"created", app.created},
            {"owned_by", "arcaine"},
        }})},
    };
}

json tool_calls_json(const std::vector<ParsedToolCall>& calls) {
    json out = json::array();
    for (const auto& call : calls) {
        out.push_back({
            {"id", call.id},
            {"type", "function"},
            {"function", {
                {"name", call.name},
                {"arguments", call.arguments},
            }},
        });
    }
    return out;
}

json chat_response(const std::string& id, const ChatRequest& chat,
                   std::time_t created, const ParsedAssistantOutput& parsed,
                   int prompt_tokens, int completion_tokens,
                   const std::string& finish_reason,
                   const ResponseMetrics& metrics) {
    json message = {
        {"role", "assistant"},
        {"content", parsed.content.empty() && !parsed.tool_calls.empty()
            ? json(nullptr)
            : json(parsed.content)},
    };
    if (!parsed.tool_calls.empty())
        message["tool_calls"] = tool_calls_json(parsed.tool_calls);

    return {
        {"id", id},
        {"object", "chat.completion"},
        {"created", created},
        {"model", chat.model},
        {"choices", json::array({{
            {"index", 0},
            {"message", std::move(message)},
            {"logprobs", nullptr},
            {"finish_reason", finish_reason},
        }})},
        {"usage", usage_json(prompt_tokens, completion_tokens)},
        {"metrics", metrics_json(metrics)},
    };
}

void handle_non_streaming(const ChatRequest& chat, const std::vector<int>& prompt_ids,
                          AppState& app, httplib::Response& res) {
    std::lock_guard<std::mutex> lock(app.generate_mu);
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    bool saw_first_commit = false;
    double ttft_s = 0.0;
    DiffStreamCallback on_step = [&](const DiffStepEvent& ev) {
        if (ev.committed && !saw_first_commit) {
            ttft_s = std::chrono::duration<double>(Clock::now() - start).count();
            saw_first_commit = true;
        }
    };
    std::vector<int> generated = app.model.generate(prompt_ids, chat.max_tokens,
                                                    chat.denoising_steps, chat.seed,
                                                    false, on_step);
    double duration_s = std::chrono::duration<double>(Clock::now() - start).count();
    if (generated.size() > (size_t)chat.max_tokens)
        generated.resize((size_t)chat.max_tokens);
    if (!saw_first_commit && !generated.empty()) ttft_s = duration_s;
    ParsedAssistantOutput parsed = parse_assistant_output(app.tokenizer.decode_raw(generated));
    std::string finish_reason = parsed.tool_calls.empty()
        ? finish_reason_for(generated.size(), chat.max_tokens)
        : "tool_calls";
    ResponseMetrics metrics = make_metrics((int)prompt_ids.size(), (int)generated.size(),
                                           ttft_s, duration_s, app.model.stats());
    set_json(res, 200, chat_response(completion_id(), chat, std::time(nullptr), parsed,
                                     (int)prompt_ids.size(), (int)generated.size(),
                                     finish_reason, metrics));
    log_line("info", "completion generated model=" + chat.model +
                     " output_tokens=" + std::to_string(generated.size()) +
                     " finish_reason=" + finish_reason +
                     " tool_calls=" + std::to_string(parsed.tool_calls.size()) +
                     " " + metrics_log(metrics));
}

void handle_streaming(const ChatRequest& chat, std::vector<int> prompt_ids,
                      AppState& app, httplib::Response& res) {
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider(
        "text/event-stream",
        [&app, chat, prompt_ids = std::move(prompt_ids), id = completion_id()]
        (size_t, httplib::DataSink& sink) mutable -> bool {
            std::lock_guard<std::mutex> lock(app.generate_mu);
            using Clock = std::chrono::steady_clock;
            auto start = Clock::now();
            bool saw_first_commit = false;
            double ttft_s = 0.0;
            const std::time_t created = std::time(nullptr);
            bool ok = write_sse(sink, chat_completion_chunk(
                id, created, chat.model, {{"role", "assistant"}}));

            std::vector<int> emitted_ids;
            std::string emitted_content;

            // Prefix-diff freshly parsed content against what we already
            // streamed; on a non-prefix change re-emit the whole string,
            // matching the prior content behavior.
            auto channel_delta = [](const std::string& full, std::string& emitted) -> std::string {
                std::string delta;
                if (full.size() >= emitted.size() &&
                    full.compare(0, emitted.size(), emitted) == 0) {
                    delta = full.substr(emitted.size());
                } else {
                    delta = full;
                }
                emitted = full;
                return delta;
            };

            DiffStreamCallback on_step = [&](const DiffStepEvent& ev) {
                if (!ok || !ev.canvas) return;
                if (ev.committed && !saw_first_commit) {
                    ttft_s = std::chrono::duration<double>(Clock::now() - start).count();
                    saw_first_commit = true;
                }
                if (chat.stream_drafts && !ev.committed) {
                    json draft = {
                        {"block", ev.block},
                        {"step", ev.cur_step},
                        {"temperature", ev.temperature},
                        {"mean_entropy", ev.mean_entropy},
                        {"text", app.tokenizer.decode(*ev.canvas)},
                    };
                    ok = write_sse(sink, draft, "arcaine.diffusion_step");
                    if (!ok) return;
                }
                if (chat.has_tools) return;
                if (!ev.committed || emitted_ids.size() >= (size_t)chat.max_tokens)
                    return;

                size_t remaining = (size_t)chat.max_tokens - emitted_ids.size();
                size_t take = std::min(remaining, ev.canvas->size());
                emitted_ids.insert(emitted_ids.end(), ev.canvas->begin(),
                                   ev.canvas->begin() +
                                       static_cast<std::vector<int>::difference_type>(take));

                ParsedAssistantOutput parsed =
                    parse_assistant_output(app.tokenizer.decode_raw(emitted_ids));
                std::string content_delta =
                    channel_delta(parsed.content, emitted_content);
                if (!content_delta.empty()) {
                    ok = write_sse(sink, chat_completion_chunk(
                        id, created, chat.model, {{"content", content_delta}}));
                }
            };

            try {
                std::vector<int> generated = app.model.generate(prompt_ids, chat.max_tokens,
                                                                chat.denoising_steps,
                                                                chat.seed, false, on_step);
                double duration_s = std::chrono::duration<double>(Clock::now() - start).count();
                if (generated.size() > (size_t)chat.max_tokens)
                    generated.resize((size_t)chat.max_tokens);
                if (!saw_first_commit && !generated.empty()) ttft_s = duration_s;

                if (chat.has_tools && ok) {
                    ParsedAssistantOutput parsed =
                        parse_assistant_output(app.tokenizer.decode_raw(generated));
                    emitted_ids = generated;
                    json delta = json::object();
                    if (!parsed.tool_calls.empty()) {
                        delta["tool_calls"] = tool_calls_json(parsed.tool_calls);
                    } else if (!parsed.content.empty()) {
                        delta["content"] = parsed.content;
                    }
                    if (!delta.empty()) {
                        ok = write_sse(sink, chat_completion_chunk(
                            id, created, chat.model, std::move(delta)));
                    }
                } else if (emitted_ids.empty() && !generated.empty() && ok) {
                    if (generated.size() > (size_t)chat.max_tokens)
                        generated.resize((size_t)chat.max_tokens);
                    emitted_ids = generated;
                    ParsedAssistantOutput parsed =
                        parse_assistant_output(app.tokenizer.decode_raw(emitted_ids));
                    json delta = json::object();
                    if (!parsed.content.empty())
                        delta["content"] = parsed.content;
                    if (!delta.empty()) {
                        ok = write_sse(sink, chat_completion_chunk(
                            id, created, chat.model, std::move(delta)));
                    }
                }

                if (emitted_ids.size() > (size_t)chat.max_tokens)
                    emitted_ids.resize((size_t)chat.max_tokens);
                const bool final_has_tool_calls = chat.has_tools &&
                    !parse_assistant_output(app.tokenizer.decode_raw(generated)).tool_calls.empty();
                const std::string finish_reason = final_has_tool_calls
                    ? "tool_calls"
                    : finish_reason_for(emitted_ids.size(), chat.max_tokens);
                ResponseMetrics metrics = make_metrics((int)prompt_ids.size(),
                                                       (int)emitted_ids.size(),
                                                       ttft_s, duration_s,
                                                       app.model.stats());
                log_line("info", "stream completion generated model=" + chat.model +
                                 " output_tokens=" + std::to_string(emitted_ids.size()) +
                                 " finish_reason=" + finish_reason +
                                 " " + metrics_log(metrics));
                if (ok) {
                    ok = write_sse(sink, chat_completion_chunk(
                        id, created, chat.model, json::object(), finish_reason,
                        nullptr, metrics_json(metrics)));
                }
                if (ok && chat.include_usage) {
                    json usage_chunk = {
                        {"id", id},
                        {"object", "chat.completion.chunk"},
                        {"created", created},
                        {"model", chat.model},
                        {"choices", json::array()},
                        {"usage", usage_json((int)prompt_ids.size(), (int)emitted_ids.size())},
                        {"metrics", metrics_json(metrics)},
                    };
                    ok = write_sse(sink, usage_chunk);
                }
            } catch (const std::exception& e) {
                log_line("error", "stream generation failed model=" + chat.model + ": " + e.what());
                ok = write_sse(sink, error_body(e.what(), "server_error", "internal_error"),
                               "error");
            }
            if (ok) ok = write_sse_done(sink);
            if (ok) sink.done();
            return ok;
        });
}

void apply_layers_spec(const std::string& value, DiffPlacementOptions& placement) {
    if (value == "auto") {
        placement.layer_mode = DiffLayerPlacementMode::Auto;
        placement.layer_split = -1;
    } else if (value == "single") {
        placement.layer_mode = DiffLayerPlacementMode::Single;
        placement.layer_split = -1;
    } else if (value.rfind("split:", 0) == 0) {
        placement.layer_mode = DiffLayerPlacementMode::Split;
        placement.layer_split = std::stoi(value.substr(6));
    } else if (value.rfind("ranges:", 0) == 0 || value.rfind("gpus:", 0) == 0) {
        throw std::runtime_error("--layers value '" + value + "' is not implemented by this runtime yet");
    } else {
        throw std::runtime_error("--layers must be one of: auto, single, split:N, ranges:N,N,..., gpus:N");
    }
}

void apply_experts_spec(const std::string& value, DiffPlacementOptions& placement) {
    if (value == "auto") {
        placement.expert_mode = DiffExpertPlacementMode::Auto;
    } else if (value == "layer-owner") {
        placement.expert_mode = DiffExpertPlacementMode::LayerOwner;
    } else if (value == "shard") {
        placement.expert_mode = DiffExpertPlacementMode::Shard;
    } else if (value == "local") {
        throw std::runtime_error("--experts local was renamed; use --experts layer-owner");
    } else if (value == "replicate" || value.rfind("ranges:", 0) == 0 || value.rfind("gpus:", 0) == 0) {
        throw std::runtime_error("--experts value '" + value + "' is not implemented by this runtime yet");
    } else {
        throw std::runtime_error("--experts must be one of: auto, layer-owner, replicate, shard, ranges:N,N,..., gpus:N");
    }
}

void apply_gpus_spec(const std::string& value) {
    if (value != "all")
        throw std::runtime_error("--gpus value '" + value + "' is not implemented by this runtime yet");
}

void usage(const char* p) {
    std::fprintf(stderr,
        "Usage: %s --model <dir> [options]\n"
        "  --model <dir>              model directory containing config/tokenizer/weights\n"
        "  --served-model-name <id>   API model id (default: basename of --model)\n"
        "  --host <addr>              listen address (default: 127.0.0.1)\n"
        "  --port <N>                 listen port (default: 8000)\n"
        "  --max-seq <N>              KV cache capacity (default: 2048)\n"
        "  --max-tokens <N>           default max completion tokens (default: 256)\n"
        "  --steps <N>                default denoising steps (default: model config)\n"
        "  --seed <S>                 default RNG seed (default: 42)\n"
        "  --layers <spec>            layer placement: auto, single, split:N, ranges:N,N,..., gpus:N\n"
        "  --experts <spec>           expert placement: auto, layer-owner, replicate, shard, ranges:N,N,..., gpus:N\n"
        "  --gpus <spec>              GPU selection: all, or future comma-list (default: all)\n"
        "  --print-placement          print resolved placement during model load (default)\n"
        "  --no-print-placement       suppress placement report during model load\n"
        "  --chat-template-kwargs <json>  JSON object merged into chat template vars,\n"
        "                             e.g. '{\"enable_thinking\": true}' (default: {})\n",
        p);
}

ServerOptions parse_args(int argc, char** argv) {
    ServerOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if      (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
        else if (a == "--model")             opts.model_dir = next();
        else if (a == "--served-model-name") opts.served_model_name = next();
        else if (a == "--host")              opts.host = next();
        else if (a == "--port")              opts.port = std::stoi(next());
        else if (a == "--max-seq")           opts.max_seq = std::stoi(next());
        else if (a == "--max-tokens")        opts.default_max_tokens = std::stoi(next());
        else if (a == "--steps")             opts.steps = std::stoi(next());
        else if (a == "--seed")              opts.seed = (unsigned)std::stoul(next());
        else if (a == "--layers")            apply_layers_spec(next(), opts.placement);
        else if (a == "--experts")           apply_experts_spec(next(), opts.placement);
        else if (a == "--gpus")              apply_gpus_spec(next());
        else if (a == "--print-placement")   opts.print_placement = true;
        else if (a == "--no-print-placement") opts.print_placement = false;
        else if (a == "--chat-template-kwargs") opts.chat_template_kwargs = parse_chat_template_kwargs(next());
        else throw std::runtime_error("unknown arg: " + a);
    }
    if (opts.model_dir.empty()) throw std::runtime_error("--model is required");
    if (opts.served_model_name.empty()) opts.served_model_name = basename_of(opts.model_dir);
    if (opts.port <= 0 || opts.port > 65535) throw std::runtime_error("--port must be 1..65535");
    if (opts.max_seq <= 0) throw std::runtime_error("--max-seq must be greater than 0");
    if (opts.default_max_tokens <= 0) throw std::runtime_error("--max-tokens must be greater than 0");
    return opts;
}

} // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        ServerOptions opts = parse_args(argc, argv);
        std::printf("[api] loading model from %s ...\n", opts.model_dir.c_str());
        AppState app(std::move(opts));
        std::printf("[api] serving model id '%s'\n", app.opts.served_model_name.c_str());
        std::printf("[api] auth: %s\n", app.api_key.empty() ? "disabled" : "ARCAINE_API_KEY");

        httplib::Server server;

        server.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
            log_line("info", request_label(req) + " received");
            if (!authorized(req, app)) {
                send_auth_error(res, req);
                return;
            }
            set_json(res, 200, models_response(app));
            log_line("info", request_label(req) + " -> HTTP 200");
        });

        server.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
            log_line("info", request_label(req) + " received");
            try {
                if (!authorized(req, app)) {
                    send_auth_error(res, req);
                    return;
                }
                ChatRequest chat = parse_chat_request(req, app);
                std::vector<int> prompt_ids = app.tokenizer.build_prompt_json(
                    chat.messages, chat.tools, app.opts.chat_template_kwargs);
                if ((int)prompt_ids.size() + chat.max_tokens > app.model.kv_cache_max_seq()) {
                    bad_request("prompt_tokens + max_tokens exceeds server max_seq");
                }
                log_line("info", request_label(req) + " model=" + chat.model +
                                 " stream=" + (chat.stream ? "true" : "false") +
                                 " tools=" + (chat.has_tools ? "true" : "false") +
                                 " prompt_tokens=" + std::to_string(prompt_ids.size()) +
                                 " max_tokens=" + std::to_string(chat.max_tokens));
                if (chat.stream) {
                    handle_streaming(chat, std::move(prompt_ids), app, res);
                    log_line("info", request_label(req) + " -> HTTP 200 stream");
                } else {
                    handle_non_streaming(chat, prompt_ids, app, res);
                    log_line("info", request_label(req) + " -> HTTP 200");
                }
            } catch (const OpenAiError& e) {
                log_line("error", request_label(req) + " -> HTTP " +
                                  std::to_string(e.status) + " " + e.code + ": " + e.what());
                send_openai_error(res, e);
            } catch (const std::exception& e) {
                log_line("error", request_label(req) + " -> HTTP 500 internal_error: " + e.what());
                set_json(res, 500, error_body(e.what(), "server_error", "internal_error"));
            }
        });

        server.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
            log_line("error", request_label(req) + " -> HTTP " +
                              std::to_string(res.status) + " not_found");
            set_json(res, res.status, error_body("not found", "invalid_request_error", "not_found"));
        });

        server.set_exception_handler([](const httplib::Request& req, httplib::Response& res,
                                        std::exception_ptr ep) {
            std::string message = "unknown exception";
            if (ep) {
                try {
                    std::rethrow_exception(ep);
                } catch (const std::exception& e) {
                    message = e.what();
                } catch (...) {
                    message = "non-standard exception";
                }
            }
            log_line("error", request_label(req) + " -> HTTP 500 unhandled exception: " + message);
            set_json(res, 500, error_body(message, "server_error", "internal_error"));
        });

        log_line("info", "listening on http://" + app.opts.host + ":" + std::to_string(app.opts.port));
        if (!server.listen(app.opts.host, app.opts.port)) {
            log_line("error", "failed to listen on " + app.opts.host + ":" +
                              std::to_string(app.opts.port));
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        log_line("error", e.what());
        usage(argv[0]);
        return 1;
    }
}
