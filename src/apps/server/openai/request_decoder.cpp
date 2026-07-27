#include "apps/server/openai/request_decoder.hpp"

#include "apps/server/app_state.hpp"
#include "apps/server/openai/error_encoder.hpp"
#include "apps/server/openai/schemas.hpp"

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <string>

namespace arcaine::openai {
namespace {
using json = nlohmann::ordered_json;

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
            normalized["function"]["parameters"] = {{"type", "object"},
                                                    {"properties", json::object()}};
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
        !choice.at("function").contains("name") || !choice.at("function").at("name").is_string())
        bad_request("function tool_choice must include function.name");
    std::string chosen = choice.at("function").at("name").get<std::string>();
    json filtered = json::array();
    for (const auto& tool : tools)
        if (tool.at("function").at("name").get<std::string>() == chosen)
            filtered.push_back(tool);
    if (filtered.empty())
        bad_request("tool_choice references a function that is not present in tools");
    return filtered;
}

void reject_unsupported_fields(const json& body) {
    auto reject_present = [&](const char* key) {
        if (body.contains(key))
            unsupported_request(std::string(key) + " is not supported by this server");
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
        if (type != "text") unsupported_request("only text message content is supported");
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
        if (!fn.contains("arguments")) bad_request("tool call function.arguments is required");
        json arguments;
        if (fn.at("arguments").is_string()) {
            const std::string raw = fn.at("arguments").get<std::string>();
            try {
                json parsed = json::parse(raw);
                arguments = parsed.is_object() ? std::move(parsed) : json(raw);
            } catch (const std::exception&) {
                arguments = raw;
            }
        } else {
            arguments = fn.at("arguments").is_object()
                            ? fn.at("arguments")
                            : json(fn.at("arguments").dump());
        }
        json normalized = {{"type", "function"},
                           {"function", {{"name", fn.at("name").get<std::string>()},
                                         {"arguments", std::move(arguments)}}}};
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
}  // namespace
}  // namespace arcaine::openai

namespace arcaine::server {
namespace {
using json = nlohmann::ordered_json;
}  // namespace

arcaine::inference::GenerationRequest decode_chat_completion_request(
    const httplib::Request& req, const AppState& app) {
    using namespace arcaine::openai;
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
    const std::string model = body.at("model").get<std::string>();
    if (model != app.opts.served_model_name)
        throw OpenAiError(404, "invalid_request_error", "model_not_found",
                         "model '" + model + "' is not served by this server");

    arcaine::inference::GenerationRequest gen;
    gen.served_model_name = app.opts.served_model_name;
    gen.messages = parse_messages(body);
    gen.tools = apply_tool_choice(body, validate_tools(body));
    // tool_choice is folded into `tools` above (apply_tool_choice filters);
    // carry the raw choice too for models that consume it directly.
    if (body.contains("tool_choice")) gen.tool_choice = body.at("tool_choice");

    bool stream = false;
    require_bool(body, "stream", stream);
    gen.stream.enabled = stream;
    require_bool(body, "arcaine_stream_drafts", gen.stream.stream_drafts);
    if (body.contains("stream_options") && body.at("stream_options").is_object()) {
        const auto& options = body.at("stream_options");
        if (options.contains("include_usage")) {
            if (!options.at("include_usage").is_boolean())
                bad_request("stream_options.include_usage must be a boolean");
            gen.stream.include_usage = options.at("include_usage").get<bool>();
        }
    }

    gen.max_output_tokens =
        require_positive_int(body, "max_tokens", app.opts.default_max_tokens);
    gen.max_output_tokens =
        require_positive_int(body, "max_completion_tokens", gen.max_output_tokens);
    gen.denoising_steps =
        require_positive_int(body, "arcaine_denoising_steps",
                              app.opts.steps > 0 ? app.opts.steps : -1);
    gen.seed = require_seed(body, app.opts.seed);

    if (body.contains("temperature")) {
        if (!body.at("temperature").is_number()) bad_request("temperature must be a number");
        gen.sampling.temperature = body.at("temperature").get<float>();
    }
    if (body.contains("top_p")) {
        if (!body.at("top_p").is_number()) bad_request("top_p must be a number");
        gen.sampling.top_p = body.at("top_p").get<float>();
    }
    if (body.contains("top_k")) {
        if (!body.at("top_k").is_number_integer()) bad_request("top_k must be an integer");
        gen.sampling.top_k = body.at("top_k").get<int>();
    }

    gen.chat_template.kwargs = app.opts.chat_template_kwargs;
    return gen;
}

}  // namespace arcaine::server
