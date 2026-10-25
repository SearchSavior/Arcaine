#include "weights.hpp"
#include "awq_decode.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "prefill_xmx.hpp"

namespace flint {
namespace {

struct STEntry {
    int shard = -1;
    uint64_t off = 0;    // absolute byte offset in the shard file
    uint64_t bytes = 0;
    std::string dtype;
    std::vector<uint64_t> shape;
    bool consumed = false;
};

struct Shard {
    std::string path;
    int fd = -1;
};

size_t checked_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::runtime_error("flint: weight size overflow");
    return a + b;
}

size_t checked_mul(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::runtime_error("flint: weight size overflow");
    return a * b;
}

size_t tensor_bytes(const STEntry& e, const std::string& name) {
    size_t bytes = 0;
    if (e.dtype == "U8") bytes = 1;
    else if (e.dtype == "BF16" || e.dtype == "F16") bytes = 2;
    else if (e.dtype == "I32") bytes = 4;
    else throw std::runtime_error("flint: unsupported dtype " + e.dtype + " for " + name);
    for (uint64_t d : e.shape) {
        if (d > std::numeric_limits<size_t>::max())
            throw std::runtime_error("flint: dimension overflow for " + name);
        bytes = checked_mul(bytes, (size_t)d);
    }
    return bytes;
}

void read_exact(int fd, void* dst, uint64_t bytes, uint64_t off,
                const std::string& name) {
    auto* p = static_cast<uint8_t*>(dst);
    while (bytes) {
        const size_t n = (size_t)std::min<uint64_t>(bytes, 64ull << 20);
        const ssize_t r = ::pread(fd, p, n, (off_t)off);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) throw std::runtime_error("flint: short read for " + name);
        p += r;
        off += (uint64_t)r;
        bytes -= (uint64_t)r;
    }
}

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
            const int shard_id = (int)shards_.size();
            shards_.push_back({p.string(), -1});
            Shard& sh = shards_.back();
            sh.fd = ::open(sh.path.c_str(), O_RDONLY);
            if (sh.fd < 0) throw std::runtime_error("flint: cannot open " + sh.path);
            struct stat st;
            if (::fstat(sh.fd, &st) != 0 || st.st_size < 8)
                throw std::runtime_error("flint: bad shard size in " + sh.path);
            const uint64_t file_bytes = (uint64_t)st.st_size;
            uint64_t hlen = 0;
            read_exact(sh.fd, &hlen, sizeof(hlen), 0, sh.path);
            if (hlen == 0 || hlen > (1u << 30) || hlen > file_bytes - 8)
                throw std::runtime_error("flint: bad header length in " + sh.path);
            std::string header(hlen, '\0');
            read_exact(sh.fd, header.data(), hlen, 8, sh.path);
            nlohmann::json j = nlohmann::json::parse(header.begin(), header.end());
            if (!j.is_object())
                throw std::runtime_error("flint: bad header in " + sh.path);
            std::vector<std::pair<uint64_t, uint64_t>> ranges;
            for (auto it = j.begin(); it != j.end(); ++it) {
                if (it.key() == "__metadata__") continue;
                const auto& v = it.value();
                STEntry e;
                e.shard = shard_id;
                e.dtype = v.at("dtype").get<std::string>();
                const auto& shape = v.at("shape");
                if (!shape.is_array())
                    throw std::runtime_error("flint: bad shape for " + it.key());
                for (auto& d : shape) {
                    if (!d.is_number_integer() || d < 0)
                        throw std::runtime_error("flint: bad dimension for " + it.key());
                    e.shape.push_back(d.get<uint64_t>());
                }
                const auto& offs = v.at("data_offsets");
                if (!offs.is_array() || offs.size() != 2 ||
                    !offs[0].is_number_integer() || !offs[1].is_number_integer() ||
                    offs[0] < 0 || offs[1] < 0)
                    throw std::runtime_error("flint: bad offsets for " + it.key());
                const uint64_t begin = offs[0].get<uint64_t>();
                const uint64_t end = offs[1].get<uint64_t>();
                if (begin > end || end > file_bytes - 8 - hlen)
                    throw std::runtime_error("flint: out-of-bounds tensor " + it.key());
                e.off = 8 + hlen + begin;
                e.bytes = end - begin;
                if (e.bytes != tensor_bytes(e, it.key()))
                    throw std::runtime_error("flint: byte count mismatch for " + it.key());
                if (entries_.count(it.key()))
                    throw std::runtime_error("flint: duplicate tensor " + it.key());
                ranges.emplace_back(begin, end);
                entries_.emplace(it.key(), std::move(e));
            }
            std::sort(ranges.begin(), ranges.end());
            uint64_t end = 0;
            for (const auto& r : ranges) {
                if (r.first < end)
                    throw std::runtime_error("flint: overlapping tensors in " + sh.path);
                end = r.second;
            }
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

    const STEntry& consume(const std::string& name) {
        auto it = entries_.find(name);
        if (it == entries_.end())
            throw std::runtime_error("flint: missing tensor " + name);
        if (it->second.consumed)
            throw std::runtime_error("flint: tensor registered twice: " + name);
        it->second.consumed = true;
        return it->second;
    }

    void require_consumed() const {
        size_t count = 0;
        std::string first;
        for (const auto& kv : entries_) {
            if (kv.second.consumed) continue;
            if (first.empty()) first = kv.first;
            ++count;
        }
        if (count)
            throw std::runtime_error("flint: " + std::to_string(count) +
                                     " unconsumed tensors, including " + first);
    }

    void read_to(const std::string& name, uint8_t* dst, uint64_t bytes,
                 uint64_t off = 0) const {
        const STEntry& e = get(name);
        if (off > e.bytes || bytes > e.bytes - off)
            throw std::runtime_error("flint: read exceeds tensor " + name);
        read_exact(shards_[e.shard].fd, dst, bytes, e.off + off, name);
    }

    // Copy a whole tensor into a device pointer, in bounded host chunks.
    void copy_to(const std::string& name, uint8_t* dst, sycl::queue& q,
                 std::vector<uint8_t>& staging) const {
        const STEntry& e = get(name);
        if (staging.empty()) throw std::runtime_error("flint: empty copy staging");
        for (uint64_t done = 0; done < e.bytes;) {
            const uint64_t n = std::min<uint64_t>(staging.size(), e.bytes - done);
            read_to(name, staging.data(), n, done);
            q.memcpy(dst + done, staging.data(), n).wait_and_throw();
            done += n;
        }
    }

    // Compare the first `bytes` of a tensor against the device copy.
    bool verify(const std::string& name, const uint8_t* src, uint64_t bytes,
                sycl::queue& q) const {
        const STEntry& e = get(name);
        if (!src || bytes > e.bytes) return false;
        std::vector<uint8_t> a(bytes), b(bytes);
        read_to(name, a.data(), bytes);
        q.memcpy(b.data(), src, bytes).wait_and_throw();
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

struct AWQBank {
    std::string base;
    int n = 0;
    int k = 0;
    int experts = 0;
    size_t w_off = 0;
    size_t s_off = 0;
    size_t z_off = 0;

    std::string source(int expert) const {
        if (!experts) return base;
        const size_t dot = base.rfind('.');
        if (dot == std::string::npos)
            throw std::runtime_error("flint: bad AWQ expert base " + base);
        return base.substr(0, dot) + "." + std::to_string(expert) + base.substr(dot);
    }
};

uint64_t copy_awq_bank(const Checkpoint& ck, const AWQBank& bank,
                       uint8_t* arena, sycl::queue& q) {
    constexpr size_t kHostChunk = 64ull << 20;
    constexpr unsigned slots[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    const size_t n = (size_t)bank.n;
    const size_t k = (size_t)bank.k;
    const size_t w_stride = checked_mul(n, k / 2);
    const size_t s_stride = checked_mul(n, k / 64);
    const size_t s_bytes = checked_mul(s_stride, sizeof(sycl::half));
    const size_t per_expert = checked_add(w_stride, checked_add(s_bytes, s_stride));
    if (!per_expert || per_expert > kHostChunk)
        throw std::runtime_error("flint: AWQ expert exceeds bounded staging: " + bank.base);
    const size_t count = bank.experts > 0 ? (size_t)bank.experts : 1;
    const size_t batch = std::min(count, kHostChunk / per_expert);
    std::vector<uint8_t> weights(checked_mul(batch, w_stride));
    std::vector<uint8_t> scales(checked_mul(batch, s_bytes));
    std::vector<uint8_t> zeros(checked_mul(batch, s_stride));
    std::vector<uint32_t> raw_weights(w_stride / sizeof(uint32_t));
    std::vector<uint32_t> raw_zeros(s_stride / 8);
    const size_t packed_n = n / 8;
    for (size_t first = 0; first < count; first += batch) {
        const size_t take = std::min(batch, count - first);
        for (size_t e = 0; e < take; ++e) {
            const std::string source = bank.source((int)(first + e));
            ck.read_to(source + ".qweight",
                       reinterpret_cast<uint8_t*>(raw_weights.data()), w_stride);
            ck.read_to(source + ".qzeros",
                       reinterpret_cast<uint8_t*>(raw_zeros.data()), s_stride / 2);
            ck.read_to(source + ".scales", scales.data() + e * s_bytes, s_bytes);
            uint8_t* dst_w = weights.data() + e * w_stride;
            uint8_t* dst_z = zeros.data() + e * s_stride;
            for (size_t block = 0; block < packed_n; block += 16) {
                const size_t end = std::min(block + 16, packed_n);
                for (size_t kk = 0; kk < k; kk += 2) {
                    const uint32_t* lo = raw_weights.data() + kk * packed_n;
                    const uint32_t* hi = lo + packed_n;
                    for (size_t nn = block; nn < end; ++nn) {
                        const uint32_t a = lo[nn], b = hi[nn];
                        for (size_t j = 0; j < 8; ++j) {
                            const unsigned shift = slots[j] * 4;
                            dst_w[(nn * 8 + j) * (k / 2) + kk / 2] =
                                (uint8_t)(((a >> shift) & 15u) |
                                          (((b >> shift) & 15u) << 4));
                        }
                    }
                }
            }
            for (size_t group = 0; group < k / 64; ++group) {
                for (size_t nn = 0; nn < packed_n; ++nn) {
                    const uint32_t word = raw_zeros[group * packed_n + nn];
                    for (size_t j = 0; j < 8; ++j)
                        dst_z[group * n + nn * 8 + j] =
                            (uint8_t)((word >> (slots[j] * 4)) & 15u);
                }
            }
        }
        q.memcpy(arena + bank.w_off + first * w_stride, weights.data(),
                 take * w_stride).wait_and_throw();
        q.memcpy(arena + bank.s_off + first * s_bytes, scales.data(),
                 take * s_bytes).wait_and_throw();
        q.memcpy(arena + bank.z_off + first * s_stride, zeros.data(),
                 take * s_stride).wait_and_throw();
    }
    return checked_mul(count, per_expert);
}

struct DeviceFree {
    sycl::queue* q;
    void operator()(uint8_t* ptr) const noexcept {
        if (!ptr) return;
        try { q->wait(); } catch (...) {}
        try { sycl::free(ptr, *q); } catch (...) {}
    }
};

}  // namespace

FlintWeights load_flint_weights(const FlintConfig& cfg, sycl::queue& q,
                                int max_layers) {
    if (cfg.awq && std::getenv("ARCAINE_FLINT_PREFILL_INT8"))
        throw std::runtime_error("flint: ARCAINE_FLINT_PREFILL_INT8 is incompatible with AWQ");
    Checkpoint ck;
    ck.scan(cfg.model_dir);

    const int nl = cfg.num_hidden_layers;
    if (nl <= 0 || cfg.hidden_size <= 0 || cfg.hidden_size % 64 != 0 ||
        cfg.vocab_size <= 0 || cfg.head_dim <= 0 || cfg.num_attention_heads <= 0 ||
        cfg.num_key_value_heads <= 0 || cfg.num_experts <= 0 ||
        cfg.num_attention_heads > std::numeric_limits<int>::max() / cfg.head_dim ||
        cfg.num_key_value_heads > std::numeric_limits<int>::max() / cfg.head_dim)
        throw std::runtime_error("flint: invalid model dimensions");
    if (max_layers <= 0 || max_layers > nl) max_layers = nl;

    size_t cursor = 0;
    size_t planned_bytes = 0;
    uint8_t* arena = nullptr;
    std::vector<std::pair<std::string, uint64_t>> copies;
    std::vector<AWQBank> banks;
    auto allocate = [&](size_t bytes) -> size_t {
        if (!bytes) throw std::runtime_error("flint: empty weight allocation");
        cursor = checked_add(cursor, 255u) & ~size_t(255u);
        const size_t off = cursor;
        cursor = checked_add(cursor, bytes);
        if (arena && cursor > planned_bytes)
            throw std::runtime_error("flint: weight arena overflow");
        return off;
    };
    auto add = [&](const std::string& name) -> size_t {
        const STEntry& e = arena ? ck.get(name) : ck.consume(name);
        const size_t off = allocate((size_t)e.bytes);
        if (!arena) copies.emplace_back(name, off);
        return off;
    };
    auto at = [&](size_t off) -> uint8_t* {
        return arena ? arena + off : nullptr;
    };

    auto mlx_q4 = [&](const std::string& base, int n, int k, int experts) {
        if (n <= 0 || k <= 0 || k % 64 != 0 || experts < 0)
            throw std::runtime_error("flint: invalid quantized dimensions for " + base);
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
        t.w = at(add(base + ".weight"));
        t.s = reinterpret_cast<const bf16*>(at(add(base + ".scales")));
        t.b = reinterpret_cast<const bf16*>(at(add(base + ".biases")));
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
    auto q4 = [&](const std::string& base, int n, int k, int experts) {
        if (!cfg.awq) return mlx_q4(base, n, k, experts);
        if (n <= 0 || k <= 0 || n % 8 != 0 || k % 64 != 0 || experts < 0)
            throw std::runtime_error("flint: invalid AWQ dimensions for " + base);
        AWQBank bank;
        bank.base = base;
        bank.n = n;
        bank.k = k;
        bank.experts = experts;
        const size_t count = experts > 0 ? (size_t)experts : 1;
        const size_t w_stride = checked_mul((size_t)n, (size_t)k / 2);
        const size_t s_stride = checked_mul((size_t)n, (size_t)k / 64);
        if (checked_add(w_stride, checked_mul(s_stride, 3)) > (64ull << 20))
            throw std::runtime_error("flint: AWQ expert exceeds bounded staging: " + base);
        if (!arena) {
            for (size_t e = 0; e < count; ++e) {
                const std::string source = bank.source((int)e);
                const std::string wn = source + ".qweight";
                const std::string zn = source + ".qzeros";
                const std::string sn = source + ".scales";
                expect_shape(ck.consume(wn), wn, "I32", {(uint64_t)k, (uint64_t)(n / 8)});
                expect_shape(ck.consume(zn), zn, "I32", {(uint64_t)(k / 64), (uint64_t)(n / 8)});
                expect_shape(ck.consume(sn), sn, "F16", {(uint64_t)(k / 64), (uint64_t)n});
            }
        }
        bank.w_off = allocate(checked_mul(count, w_stride));
        bank.s_off = allocate(checked_mul(checked_mul(count, s_stride), sizeof(sycl::half)));
        bank.z_off = allocate(checked_mul(count, s_stride));
        if (!arena) banks.push_back(bank);
        QTensor t;
        t.format = WeightFormat::AWQ;
        t.decode_vector = awq_decode::enabled();
        t.w = at(bank.w_off);
        t.awq_s = reinterpret_cast<const sycl::half*>(at(bank.s_off));
        t.awq_z = at(bank.z_off);
        t.n = n;
        t.k = k;
        t.experts = experts;
        t.stride_w = w_stride;
        t.stride_s = s_stride;
        t.stride_b = s_stride;
        return t;
    };
    auto dense = [&](const std::string& name, int n, int k) {
        expect_shape(ck.get(name), name, "BF16", {(uint64_t)n, (uint64_t)k});
        QTensor t;
        t.format = WeightFormat::BF16;
        t.dense = reinterpret_cast<const bf16*>(at(add(name)));
        t.n = n;
        t.k = k;
        return t;
    };
    auto norm = [&](const std::string& name, int n) -> const bf16* {
        const STEntry& e = ck.get(name);
        expect_shape(e, name, "BF16", {(uint64_t)n});
        return reinterpret_cast<const bf16*>(at(add(name)));
    };

    FlintWeights w;
    auto map_weights = [&]() {
        if (cfg.awq) {
            w.embed = dense("model.embed_tokens.weight", cfg.vocab_size, cfg.hidden_size);
            w.head = dense("lm_head.weight", cfg.vocab_size, cfg.hidden_size);
        } else {
            const STEntry& we = ck.get("model.embed_tokens.weight");
            const STEntry& se = ck.get("model.embed_tokens.scales");
            const STEntry& be = ck.get("model.embed_tokens.biases");
            expect_shape(we, "model.embed_tokens.weight", "U8",
                         {(uint64_t)cfg.vocab_size, (uint64_t)cfg.hidden_size});
            expect_shape(se, "model.embed_tokens.scales", "BF16",
                         {(uint64_t)cfg.vocab_size, (uint64_t)(cfg.hidden_size / 64)});
            expect_shape(be, "model.embed_tokens.biases", "BF16",
                         {(uint64_t)cfg.vocab_size, (uint64_t)(cfg.hidden_size / 64)});
            w.embed.w = at(add("model.embed_tokens.weight"));
            w.embed.s = reinterpret_cast<const bf16*>(at(add("model.embed_tokens.scales")));
            w.embed.b = reinterpret_cast<const bf16*>(at(add("model.embed_tokens.biases")));
            w.embed.n = cfg.vocab_size;
            w.embed.k = cfg.hidden_size;
            w.head = q4("lm_head", cfg.vocab_size, cfg.hidden_size, 0);
        }
        w.final_norm = norm("model.norm.weight", cfg.hidden_size);

        w.layers.clear();
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
                L.router = reinterpret_cast<const bf16*>(at(add(mp + "gate.weight")));

                const std::string expert = mp + (cfg.awq ? "experts." : "switch_mlp.");
                L.egate = q4(expert + "gate_proj", cfg.moe_intermediate_size,
                             cfg.hidden_size, cfg.num_experts);
                L.eup = q4(expert + "up_proj", cfg.moe_intermediate_size,
                           cfg.hidden_size, cfg.num_experts);
                L.edown = q4(expert + "down_proj", cfg.hidden_size,
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
    };
    map_weights();
    if (max_layers == nl) ck.require_consumed();
    planned_bytes = cursor;
    const size_t arena_bytes = planned_bytes;
    if (arena_bytes > q.get_device().get_info<sycl::info::device::max_mem_alloc_size>())
        throw std::runtime_error("flint: weight arena exceeds device maximum allocation");
    arena = sycl::aligned_alloc_device<uint8_t>(256, arena_bytes, q);
    if (!arena) throw std::runtime_error("flint: failed to allocate weight arena");
    std::unique_ptr<uint8_t, DeviceFree> arena_guard(arena, DeviceFree{&q});
    w.arena = arena;
    w.arena_bytes = arena_bytes;
    cursor = 0;
    map_weights();
    if (cursor != planned_bytes)
        throw std::runtime_error("flint: inconsistent weight arena plan");

    uint64_t copied = 0;
    uint64_t next_report = 4ull << 30;
    auto report = [&]() {
        if (copied >= next_report) {
            std::printf("[flint-load] %.1f / %.1f GB\n", copied / 1e9,
                       arena_bytes / 1e9);
            std::fflush(stdout);
            next_report += 4ull << 30;
        }
    };
    {
        std::vector<uint8_t> staging(64ull << 20);
        for (const auto& [name, off] : copies) {
            ck.copy_to(name, arena + off, q, staging);
            copied += ck.get(name).bytes;
            report();
        }
    }
    for (const auto& bank : banks) {
        copied += copy_awq_bank(ck, bank, arena, q);
        report();
    }
    q.wait_and_throw();

    // The XMX int8 prefill GEMM reads the weights in DPAS B-fragment order (see
    // prefill_xmx.hpp).  Permute them once here: a single pass over the device
    // arena, staged through one temporary buffer sized to the largest tensor.
    // Only the int8 prefill path is affected -- lm_head and embed keep the
    // original layout (the decode GEMV / 8-bit embedding read them directly).
    if (!cfg.awq && std::getenv("ARCAINE_FLINT_PREFILL_INT8")) {
        std::vector<std::pair<std::string, const QTensor*>> pfx;
        for (auto& L : w.layers) {
            const std::string lp = "model.layers." + std::to_string(L.index) + ".";
            auto reg = [&](const char* nm, const QTensor& t) {
                if (t.valid()) pfx.emplace_back(lp + nm, &t);
            };
            reg("self_attn.q_proj", L.q);
            reg("self_attn.k_proj", L.k);
            reg("self_attn.v_proj", L.v);
            reg("self_attn.o_proj", L.o);
            reg("mlp.gate_proj", L.gate);
            reg("mlp.up_proj", L.up);
            reg("mlp.down_proj", L.down);
            reg("mlp.switch_mlp.gate_proj", L.egate);
            reg("mlp.switch_mlp.up_proj", L.eup);
            reg("mlp.switch_mlp.down_proj", L.edown);
            reg("mlp.shared_expert.gate_proj", L.sgate);
            reg("mlp.shared_expert.up_proj", L.sup);
            reg("mlp.shared_expert.down_proj", L.sdown);
        }
        // Fail loudly rather than run a GEMM over a half-repacked arena.
        size_t tmp_bytes = 0, total_bytes = 0;
        for (auto& [nm, t] : pfx) {
            const size_t bytes =
                (size_t)t->n * (t->k / 2) * (size_t)(t->experts > 0 ? t->experts : 1);
            total_bytes += bytes;
            if (t->k % 64 != 0)
                throw std::runtime_error("flint: cannot repack " + nm +
                                         " for the DPAS layout: K=" +
                                         std::to_string(t->k) + " is not a "
                                         "multiple of 64");
            if (t->n % 16 != 0)
                throw std::runtime_error("flint: cannot repack " + nm +
                                         " for the DPAS layout: N=" +
                                         std::to_string(t->n) + " is not a "
                                         "multiple of 16");
            if ((reinterpret_cast<uintptr_t>(t->w) & 255) != 0)
                throw std::runtime_error("flint: cannot repack " + nm +
                                         " for the DPAS layout: weight base is "
                                         "not 256-byte aligned");
            tmp_bytes = std::max(tmp_bytes, bytes);
        }
        if (!pfx.empty()) {
            uint8_t* tmp = sycl::malloc_device<uint8_t>(tmp_bytes, q);
            if (!tmp) throw std::runtime_error("flint: repack staging alloc failed");
            const auto t0 = std::chrono::steady_clock::now();
            for (auto& [nm, t] : pfx) prefill_xmx::repack_q4_dpas(q, *t, tmp);
            q.wait();
            const double sec = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t0)
                                   .count();
            sycl::free(tmp, q);
            std::printf("[flint-load] repacked %zu q4 tensors to DPAS order "
                        "(%.3f GB) in %.3f s\n", pfx.size(), total_bytes / 1e9,
                        sec);
            std::fflush(stdout);
        }
    }

    const uint64_t norm_bytes = std::min<uint64_t>(
        checked_mul((size_t)cfg.hidden_size, sizeof(bf16)), ck.get("model.norm.weight").bytes);
    if (!ck.verify("model.norm.weight", (const uint8_t*)w.final_norm, norm_bytes, q))
        throw std::runtime_error("flint: device readback mismatch");
    const std::string embed_name = cfg.awq ? "model.embed_tokens.weight" :
                                              "model.embed_tokens.scales";
    const uint8_t* embed_data = cfg.awq ? (const uint8_t*)w.embed.dense :
                                           (const uint8_t*)w.embed.s;
    const uint64_t embed_bytes = std::min<uint64_t>(4096, ck.get(embed_name).bytes);
    if (!ck.verify(embed_name, embed_data, embed_bytes, q))
        throw std::runtime_error("flint: embedding readback mismatch");

    std::printf("[flint-load] loaded %d/%d layers, %.3f GB in arena (%.3f GB used)\n",
                max_layers, nl, arena_bytes / 1e9, cursor / 1e9);
    arena_guard.release();
    return w;
}

}  // namespace flint
