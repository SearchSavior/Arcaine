#include "weights.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace flint {
namespace {

struct STEntry {
    int shard = -1;
    uint64_t off = 0;    // absolute byte offset in the shard file
    uint64_t bytes = 0;
    std::string dtype;
    std::vector<uint64_t> shape;
};

struct Shard {
    std::string path;
    int fd = -1;
};

class Checkpoint {
public:
    ~Checkpoint() {
        for (auto& s : shards_)
            if (s.fd >= 0) ::close(s.fd);
    }

    void scan(const std::string& dir) {
        namespace fs = std::filesystem;
        std::vector<fs::path> paths;
        for (auto& e : fs::directory_iterator(dir)) {
            if (!e.is_regular_file()) continue;
            auto name = e.path().filename().string();
            if (name.rfind("model-", 0) == 0 && name.size() > 12 &&
                name.substr(name.size() - 12) == ".safetensors")
                paths.push_back(e.path());
        }
        if (paths.empty())
            throw std::runtime_error("flint: no model-*.safetensors in " + dir);
        std::sort(paths.begin(), paths.end());

        for (auto& p : paths) {
            Shard sh;
            sh.path = p.string();
            sh.fd = ::open(sh.path.c_str(), O_RDONLY);
            if (sh.fd < 0) throw std::runtime_error("flint: cannot open " + sh.path);
            uint64_t hlen = 0;
            if (::pread(sh.fd, &hlen, sizeof(hlen), 0) != (ssize_t)sizeof(hlen))
                throw std::runtime_error("flint: short header in " + sh.path);
            if (hlen == 0 || hlen > (1u << 30))
                throw std::runtime_error("flint: bad header length in " + sh.path);
            std::string header(hlen, '\0');
            if (::pread(sh.fd, header.data(), hlen, 8) != (ssize_t)hlen)
                throw std::runtime_error("flint: short header body in " + sh.path);
            nlohmann::json j = nlohmann::json::parse(header.begin(), header.end());
            const int shard_id = (int)shards_.size();
            for (auto it = j.begin(); it != j.end(); ++it) {
                if (it.key() == "__metadata__") continue;
                const auto& v = it.value();
                STEntry e;
                e.shard = shard_id;
                e.dtype = v.at("dtype").get<std::string>();
                for (auto& d : v.at("shape")) e.shape.push_back(d.get<uint64_t>());
                auto offs = v.at("data_offsets");
                e.off = 8 + hlen + offs.at(0).get<uint64_t>();
                e.bytes = offs.at(1).get<uint64_t>() - offs.at(0).get<uint64_t>();
                if (entries_.count(it.key()))
                    throw std::runtime_error("flint: duplicate tensor " + it.key());
                entries_.emplace(it.key(), std::move(e));
            }
            shards_.push_back(sh);
        }
        std::printf("[flint-load] %zu shards, %zu tensors\n", shards_.size(),
                    entries_.size());
    }

    const STEntry& get(const std::string& name) const {
        auto it = entries_.find(name);
        if (it == entries_.end())
            throw std::runtime_error("flint: missing tensor " + name);
        return it->second;
    }

    bool has(const std::string& name) const { return entries_.count(name) != 0; }

    uint64_t total_bytes() const {
        uint64_t n = 0;
        for (auto& kv : entries_) n += kv.second.bytes;
        return n;
    }

    // Copy a whole tensor into a device pointer, in bounded host chunks.
    void copy_to(const std::string& name, uint8_t* dst, sycl::queue& q,
                 std::vector<uint8_t>& staging) const {
        const STEntry& e = get(name);
        const Shard& sh = shards_[e.shard];
        constexpr uint64_t kChunk = 64ull << 20;
        for (uint64_t done = 0; done < e.bytes; done += kChunk) {
            const uint64_t n = std::min(kChunk, e.bytes - done);
            ssize_t r = ::pread(sh.fd, staging.data(), n, e.off + done);
            if (r != (ssize_t)n)
                throw std::runtime_error("flint: short read for " + name);
            q.memcpy(dst + done, staging.data(), n).wait();
        }
    }

    // Compare the first `bytes` of a tensor against the device copy.
    bool verify(const std::string& name, const uint8_t* src, uint64_t bytes,
                sycl::queue& q) const {
        const STEntry& e = get(name);
        const Shard& sh = shards_[e.shard];
        std::vector<uint8_t> a(bytes), b(bytes);
        if (::pread(sh.fd, a.data(), bytes, e.off) != (ssize_t)bytes) return false;
        q.memcpy(b.data(), src, bytes).wait();
        return std::memcmp(a.data(), b.data(), bytes) == 0;
    }

private:
    std::vector<Shard> shards_;
    std::unordered_map<std::string, STEntry> entries_;
};

void expect_shape(const STEntry& e, const std::string& name, const char* dtype,
                  std::initializer_list<uint64_t> shape) {
    if (e.dtype != dtype)
        throw std::runtime_error("flint: " + name + " dtype " + e.dtype +
                                 " != " + dtype);
    if (e.shape.size() != shape.size())
        throw std::runtime_error("flint: " + name + " rank mismatch");
    size_t i = 0;
    for (uint64_t d : shape) {
        if (e.shape[i] != d)
            throw std::runtime_error("flint: " + name + " shape mismatch at dim " +
                                     std::to_string(i));
        ++i;
    }
}

}  // namespace

FlintWeights load_flint_weights(const FlintConfig& cfg, sycl::queue& q,
                                int max_layers) {
    Checkpoint ck;
    ck.scan(cfg.model_dir);

    const int nl = cfg.num_hidden_layers;
    if (max_layers <= 0 || max_layers > nl) max_layers = nl;

    // Generous single arena: the full checkpoint plus alignment slack. This
    // keeps every tensor in one USM allocation while max_layers only limits
    // which tensors are actually copied.
    const size_t arena_bytes = (size_t)ck.total_bytes() + 4096 * 256;
    uint8_t* arena = sycl::malloc_device<uint8_t>(arena_bytes, q);
    if (!arena)
        throw std::runtime_error("flint: failed to allocate weight arena");

    size_t cursor = 0;
    std::vector<std::pair<std::string, uint64_t>> copies;
    auto add = [&](const std::string& name) -> uint64_t {
        const STEntry& e = ck.get(name);
        cursor = (cursor + 255u) & ~size_t(255u);
        if (cursor + e.bytes > arena_bytes)
            throw std::runtime_error("flint: weight arena overflow");
        const uint64_t off = cursor;
        cursor += e.bytes;
        copies.emplace_back(name, off);
        return off;
    };

    auto q4 = [&](const std::string& base, int n, int k, int experts) {
        const STEntry& we = ck.get(base + ".weight");
        const STEntry& se = ck.get(base + ".scales");
        const STEntry& be = ck.get(base + ".biases");
        if (experts > 0) {
            expect_shape(we, base + ".weight", "U8",
                         {(uint64_t)experts, (uint64_t)n, (uint64_t)(k / 2)});
            expect_shape(se, base + ".scales", "BF16",
                         {(uint64_t)experts, (uint64_t)n, (uint64_t)(k / 64)});
            expect_shape(be, base + ".biases", "BF16",
                         {(uint64_t)experts, (uint64_t)n, (uint64_t)(k / 64)});
        } else {
            expect_shape(we, base + ".weight", "U8",
                         {(uint64_t)n, (uint64_t)(k / 2)});
            expect_shape(se, base + ".scales", "BF16",
                         {(uint64_t)n, (uint64_t)(k / 64)});
            expect_shape(be, base + ".biases", "BF16",
                         {(uint64_t)n, (uint64_t)(k / 64)});
        }
        QTensor t;
        t.w = arena + add(base + ".weight");
        t.s = reinterpret_cast<const bf16*>(arena + add(base + ".scales"));
        t.b = reinterpret_cast<const bf16*>(arena + add(base + ".biases"));
        t.n = n;
        t.k = k;
        t.experts = experts;
        if (experts > 0) {
            t.stride_w = (size_t)n * (k / 2);
            t.stride_s = (size_t)n * (k / 64);
            t.stride_b = t.stride_s;
        }
        return t;
    };

    auto norm = [&](const std::string& name, int n) -> const bf16* {
        const STEntry& e = ck.get(name);
        expect_shape(e, name, "BF16", {(uint64_t)n});
        return reinterpret_cast<const bf16*>(arena + add(name));
    };

    FlintWeights w;
    w.arena = arena;
    w.arena_bytes = arena_bytes;

    // Embedding is 8-bit affine; lm_head is 4-bit affine.
    {
        const STEntry& we = ck.get("model.embed_tokens.weight");
        const STEntry& se = ck.get("model.embed_tokens.scales");
        const STEntry& be = ck.get("model.embed_tokens.biases");
        expect_shape(we, "model.embed_tokens.weight", "U8",
                     {(uint64_t)cfg.vocab_size, (uint64_t)cfg.hidden_size});
        expect_shape(se, "model.embed_tokens.scales", "BF16",
                     {(uint64_t)cfg.vocab_size, (uint64_t)(cfg.hidden_size / 64)});
        expect_shape(be, "model.embed_tokens.biases", "BF16",
                     {(uint64_t)cfg.vocab_size, (uint64_t)(cfg.hidden_size / 64)});
        w.embed.w = arena + add("model.embed_tokens.weight");
        w.embed.s = reinterpret_cast<const bf16*>(arena + add("model.embed_tokens.scales"));
        w.embed.b = reinterpret_cast<const bf16*>(arena + add("model.embed_tokens.biases"));
        w.embed.n = cfg.vocab_size;
        w.embed.k = cfg.hidden_size;
    }
    w.head = q4("lm_head", cfg.vocab_size, cfg.hidden_size, 0);
    w.final_norm = norm("model.norm.weight", cfg.hidden_size);

    w.layers.reserve(max_layers);
    for (int i = 0; i < max_layers; ++i) {
        const std::string lp = "model.layers." + std::to_string(i) + ".";
        FlintLayer L;
        L.index = i;
        L.global = cfg.is_global(i);
        L.moe = !cfg.is_dense_mlp(i);
        L.in_norm = norm(lp + "input_layernorm.weight", cfg.hidden_size);
        L.post_norm = norm(lp + "post_attention_layernorm.weight", cfg.hidden_size);
        L.q_norm = norm(lp + "self_attn.q_norm.weight", cfg.head_dim);
        L.k_norm = norm(lp + "self_attn.k_norm.weight", cfg.head_dim);
        L.sink = norm(lp + "self_attn.softmax_offset", cfg.num_attention_heads);
        L.q = q4(lp + "self_attn.q_proj", cfg.q_dim(), cfg.hidden_size, 0);
        L.k = q4(lp + "self_attn.k_proj", cfg.kv_dim(), cfg.hidden_size, 0);
        L.v = q4(lp + "self_attn.v_proj", cfg.kv_dim(), cfg.hidden_size, 0);
        L.o = q4(lp + "self_attn.o_proj", cfg.hidden_size, cfg.q_dim(), 0);

        if (!L.moe) {
            L.gate = q4(lp + "mlp.gate_proj", cfg.intermediate_size, cfg.hidden_size, 0);
            L.up = q4(lp + "mlp.up_proj", cfg.intermediate_size, cfg.hidden_size, 0);
            L.down = q4(lp + "mlp.down_proj", cfg.hidden_size, cfg.intermediate_size, 0);
        } else {
            const std::string mp = lp + "mlp.";
            const STEntry& re = ck.get(mp + "gate.weight");
            expect_shape(re, mp + "gate.weight", "BF16",
                         {(uint64_t)cfg.num_experts, (uint64_t)cfg.hidden_size});
            L.router = reinterpret_cast<const bf16*>(
                arena + add(mp + "gate.weight"));

            L.egate = q4(mp + "switch_mlp.gate_proj", cfg.moe_intermediate_size,
                         cfg.hidden_size, cfg.num_experts);
            L.eup = q4(mp + "switch_mlp.up_proj", cfg.moe_intermediate_size,
                       cfg.hidden_size, cfg.num_experts);
            L.edown = q4(mp + "switch_mlp.down_proj", cfg.hidden_size,
                         cfg.moe_intermediate_size, cfg.num_experts);
            L.sgate = q4(mp + "shared_expert.gate_proj",
                         cfg.shared_expert_intermediate_size, cfg.hidden_size, 0);
            L.sup = q4(mp + "shared_expert.up_proj",
                       cfg.shared_expert_intermediate_size, cfg.hidden_size, 0);
            L.sdown = q4(mp + "shared_expert.down_proj", cfg.hidden_size,
                         cfg.shared_expert_intermediate_size, 0);
        }
        w.layers.push_back(L);
    }

    // Copy every requested tensor straight into its arena slot.
    std::vector<uint8_t> staging(64ull << 20);
    uint64_t copied = 0;
    uint64_t next_report = 4ull << 30;
    for (auto& [name, off] : copies) {
        ck.copy_to(name, arena + off, q, staging);
        copied += ck.get(name).bytes;
        if (copied >= next_report) {
            std::printf("[flint-load] %.1f / %.1f GB\n", copied / 1e9,
                        (double)ck.total_bytes() / 1e9);
            std::fflush(stdout);
            next_report += 4ull << 30;
        }
    }
    q.wait();

    if (!ck.verify("model.norm.weight", (const uint8_t*)w.final_norm, 4096, q))
        throw std::runtime_error("flint: device readback mismatch");
    if (!ck.verify("model.embed_tokens.scales", (const uint8_t*)w.embed.s, 4096, q))
        throw std::runtime_error("flint: embedding scale readback mismatch");

    std::printf("[flint-load] loaded %d/%d layers, %.3f GB in arena (%.3f GB used)\n",
                max_layers, nl, arena_bytes / 1e9, cursor / 1e9);
    return w;
}

}  // namespace flint
