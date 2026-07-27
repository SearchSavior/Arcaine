#include "apps/server/openai/error_encoder.hpp"

namespace arcaine::openai {

[[noreturn]] void bad_request(const std::string& message, const std::string& code) {
    throw OpenAiError(400, "invalid_request_error", code, message);
}

[[noreturn]] void unsupported_request(const std::string& message) {
    throw OpenAiError(400, "unsupported_request", "unsupported_request", message);
}

}  // namespace arcaine::openai
