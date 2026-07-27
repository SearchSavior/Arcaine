#pragma once

#include <memory>
#include <string>

#include "inference/service/cancellation_token.hpp"

namespace arcaine::inference {

// Per-session creation parameters. A session owns mutable generation state;
// the request itself is passed to generate().
struct SessionCreateRequest {
    std::string       request_id;
    CancellationToken* cancellation = nullptr;
};

}  // namespace arcaine::inference
