#include "check.hpp"

#include "inference/contracts/model_descriptor.hpp"
#include "inference/contracts/model_load_request.hpp"
#include "inference/registry/model_registry.hpp"
#include "inference/service/inference_session.hpp"
#include "inference/service/model_service.hpp"
#include "inference/service/session_create.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace arcaine::inference;

namespace {

// Minimal ModelService stub for registry tests.
struct StubService : ModelService {
    ModelDescriptor desc;
    explicit StubService(ModelDescriptor d) : desc(std::move(d)) {}
    const ModelDescriptor& descriptor() const noexcept override { return desc; }
    std::unique_ptr<InferenceSession> create_session(const SessionCreateRequest&) override {
        return nullptr;
    }
};

ModelRegistration make_reg(std::vector<std::string> types, std::string id) {
    ModelRegistration r;
    r.descriptor.model_types      = std::move(types);
    r.descriptor.implementation_id = std::move(id);
    r.factory = [](const ModelLoadRequest&) -> std::unique_ptr<ModelService> {
        return std::make_unique<StubService>(ModelDescriptor{});
    };
    return r;
}

// Create a temp dir with a config.json whose model_type is `mt`.
std::string make_model_dir(const std::string& mt) {
    static int counter = 0;
    std::string dir = "/tmp/arcaine_registry_test_" + std::to_string(++counter);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::ofstream(dir + "/config.json")
        << "{\"model_type\":\"" << mt << "\"}";
    return dir;
}

}  // namespace

int main() {
    // ---- registration + listing ----
    ModelRegistry reg;
    reg.register_model(make_reg({"qwen3_5"}, "qwen3_5"));
    reg.register_model(make_reg({"qwen3_5_moe_text"}, "qwen3_5_moe"));
    auto types = reg.registered_model_types();
    CHECK_EQ(types.size(), 2u);
    reg.validate();  // no duplicates -> ok

    // ---- duplicate model_type rejection ----
    ModelRegistry dup;
    dup.register_model(make_reg({"gemma4_unified"}, "gemma4_unified"));
    dup.register_model(make_reg({"gemma4_unified"}, "gemma4_unified_other"));
    CHECK_THROWS(dup.validate());

    // A model with multiple types where one collides is also rejected.
    ModelRegistry dup2;
    dup2.register_model(make_reg({"a", "b"}, "m1"));
    dup2.register_model(make_reg({"b", "c"}, "m2"));
    CHECK_THROWS(dup2.validate());

    // ---- unknown model_type error ----
    ModelRegistry r2;
    r2.register_model(make_reg({"qwen3_5"}, "qwen3_5"));
    r2.validate();
    ModelLoadRequest req;
    req.model_dir = make_model_dir("diffusion_gemma");
    bool threw = false;
    try {
        (void)r2.create(req);
    } catch (const std::exception& e) {
        threw = true;
        const std::string msg = e.what();
        CHECK(msg.find("diffusion_gemma") != std::string::npos);
        CHECK(msg.find("qwen3_5") != std::string::npos);
    }
    CHECK(threw);

    // ---- config model_type resolution -> construct service ----
    ModelLoadRequest ok_req;
    ok_req.model_dir = make_model_dir("qwen3_5");
    LoadedModel loaded = r2.create(ok_req);
    CHECK(loaded.service != nullptr);
    CHECK_EQ(loaded.descriptor.implementation_id, "qwen3_5");
    CHECK_EQ(loaded.load_request.model_dir, ok_req.model_dir);

    // ---- missing config.json ----
    ModelLoadRequest bad_req;
    bad_req.model_dir = "/tmp/arcaine_registry_test_nonexistent_dir";
    CHECK_THROWS(r2.create(bad_req));

    RETURN_TESTS();
}
