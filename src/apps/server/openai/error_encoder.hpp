#pragma once

#include <stdexcept>
#include <string>

namespace arcaine::openai {

struct OpenAiError : std::runtime_error {
    int status;
    std::string type;
    std::string code;

    OpenAiError(int s, std::string t, std::string c, const std::string& msg)
        : std::runtime_error(msg), status(s), type(std::move(t)), code(std::move(c)) {}
};

[[noreturn]] void bad_request(const std::string& message,
                               const std::string& code = "invalid_request_error");
[[noreturn]] void unsupported_request(const std::string& message);

}  // namespace arcaine::openai
