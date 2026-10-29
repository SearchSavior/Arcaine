#include "model.hpp"
#include "gpu_decode.hpp"
#include "gpu_prefill.hpp"
#include "kernels/kernels.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <sycl/ext/oneapi/experimental/graph/command_graph.hpp>

namespace {
namespace graph_api = sycl::ext::oneapi::experimental;
using Executable = graph_api::command_graph<graph_api::graph_state::executable>;
template<class T> T* allocate(sycl::queue& q, size_t n) {
    T* p = sycl::malloc_device<T>(n, q);
    if (!p) throw std::bad_alloc{};
    return p;
}
}
struct FlintModel::ResidentDecode {
    sycl::queue queue;
    flint::gpu_decode::State* state = nullptr;
    int* output = nullptr;
    int* eos = nullptr;
    float* sample_values = nullptr;
    int* sample_ids = nullptr;
    int sample_group_count = 1;
    int eos_count = 0;
    struct PrefixBank { flint::bf16* data; size_t offset, elements; };
    std::vector<PrefixBank> prefix_banks;
    flint::bf16* prefix_data = nullptr;
    float* prefix_logits = nullptr;
    bool prefix_initialized = false;
    int prefix_position = -1;
    std::array<std::array<std::unique_ptr<Executable>, 9>, 10> graphs;
    explicit ResidentDecode(sycl::queue q, const std::vector<int>& ids) : queue(q) {
        try {
            state = allocate<flint::gpu_decode::State>(q, 1);
            output = allocate<int>(q, flint::gpu_decode::kBlock);
            eos_count = int(ids.size());
            if (eos_count) {
                eos = allocate<int>(q, eos_count);
                q.memcpy(eos, ids.data(), eos_count * sizeof(int)).wait_and_throw();
            }
        } catch (...) {
            if (state) sycl::free(state, q);
            if (output) sycl::free(output, q);
            if (eos) sycl::free(eos, q);
            throw;
        }
    }
    ~ResidentDecode() {
        queue.wait();
        for (auto& mode : graphs) for (auto& g : mode) g.reset();
        sycl::free(state, queue); sycl::free(output, queue);
        if (eos) sycl::free(eos, queue);
        if (sample_values) sycl::free(sample_values, queue);
        if (sample_ids) sycl::free(sample_ids, queue);
        if (prefix_data) sycl::free(prefix_data, queue);
        if (prefix_logits) sycl::free(prefix_logits, queue);
    }
};
bool FlintModel::device_generation_enabled() const { return flint::gpu_decode::enabled(); }

void FlintModel::enqueue_device_step(const int* token, int* position, const int* active,bool use_score_scratch) {
    using namespace flint;
    using namespace flint::kernels;
    const int H = cfg_.hidden_size, heads = cfg_.num_attention_heads;
    const int kvh = cfg_.num_key_value_heads, top = cfg_.num_experts_per_tok;
    const char* parallel_env = std::getenv("ARCAINE_FLINT_DECODE_PARALLEL_DOWN");
    const bool parallel_down = parallel_env && parallel_env[0]=='1' && parallel_env[1]=='\0';
    auto trace = [&](const bf16* src, bf16* dst) {
        if (!trace_enabled_) return;
        q_.parallel_for(sycl::range<1>(size_t(H)), [=](sycl::id<1> i) {
            if (*active) dst[i] = src[i];
        });
    };
    gpu_decode::embed(q_, w_.embed, token, h_);
    trace(h_, trace_);
    for (const auto& L : w_.layers) {
        const auto& c = kv_[L.index];
        rmsnorm(q_, h_, L.in_norm, normed_, H, cfg_.rms_norm_eps);
        qkv_gemv(q_, L.q, L.k, L.v, normed_, qbuf_, kbuf_, vbuf_);
        qkv_post(q_, qbuf_, kbuf_, vbuf_, L.q_norm, L.k_norm, c.k, c.v,
                 heads, kvh, L.global, 0, c.cap, cfg_.use_rope(L.index),
                 log2base_, cfg_.rms_norm_eps, position);
        gpu_prefill::attention(q_, qbuf_, nullptr, nullptr, c.k, c.v, L.sink, attn_,
                              attn_partial_, attn_stats_, 1, heads, kvh,
                              L.global, 0, c.cap, position,use_score_scratch?attn_scores_:nullptr);
        q4_gemv(q_, L.o, attn_, h_, 0, h_);
        rmsnorm(q_, h_, L.post_norm, normed_, H, cfg_.rms_norm_eps);
        if (!L.moe) {
            dense_gateup_act(q_, L.gate, L.up, normed_, gate_);
            q4_gemv(q_, L.down, gate_, h_, 0, h_);
        } else {
            router_topk(q_, L.router, normed_, cfg_.num_experts, H, top, ids_dev_, weights_dev_, attn_stats_);
            expert_gateup(q_, L, H, cfg_.moe_intermediate_size,
                          cfg_.shared_expert_intermediate_size, top, normed_, ids_dev_, act_);
            if (parallel_down) {
                expert_down(q_, L, H, cfg_.moe_intermediate_size,
                            cfg_.shared_expert_intermediate_size, top, act_, ids_dev_, down_);
                moe_combine(q_, h_, down_, weights_dev_, H, top);
            } else {
                moe_down_combine(q_, L, H, cfg_.moe_intermediate_size,
                                 cfg_.shared_expert_intermediate_size, top, act_,
                                 ids_dev_, weights_dev_, h_);
            }
        }
        trace(h_, trace_ ? trace_ + size_t(L.index+1) * H : nullptr);
    }
    rmsnorm(q_, h_, w_.final_norm, normed_, H, cfg_.rms_norm_eps);
    lm_head(q_, w_.head, normed_, logits_dev_, active);
    // EOS tails do not advance position or replace logits. KV writes touch
    // only the unused next slot, overwritten before the next active query.
    q_.single_task([=] { if (*active) ++*position; });
}
void FlintModel::run_device_block(int steps, int mode,int past_position) {
    auto& r = *resident_decode_;
    const bool use_score_scratch=!flint::flash_decode_twopass::adaptive()||past_position>=16384;
    auto& executable = r.graphs[mode+(use_score_scratch?5:0)][steps];
    if (!executable) {
        graph_api::command_graph<> graph(q_.get_context(), q_.get_device());
        graph.begin_recording(q_);
        bool recording = true;
        try {
            for (int i = 0; i < steps; ++i) {
                auto* state = r.state; int* output = r.output+i;
                if (mode == 0)
                    flint::gpu_decode::greedy(q_, logits_dev_, cfg_.vocab_size, state,
                                             output, r.eos, r.eos_count);
                else if (mode == 1)
                    flint::gpu_decode::sample(q_, logits_dev_, cfg_.vocab_size, state,
                                             output, r.eos, r.eos_count);
                else if(mode == 3)
                    flint::gpu_decode::parallel_sample20(q_,logits_dev_,cfg_.vocab_size,state,
                                                       output,r.eos,r.eos_count);
                else if(mode == 4)
                    flint::gpu_decode::distributed_sample20(q_,logits_dev_,cfg_.vocab_size,state,
                        output,r.eos,r.eos_count,r.sample_values,r.sample_ids,r.sample_group_count);
                else
                    q_.single_task([=] { *output = state->token; ++state->emitted; });
                enqueue_device_step(&state->token, &state->position, &state->active,use_score_scratch);
            }
            graph.end_recording(q_); recording = false;
            executable = std::make_unique<Executable>(graph.finalize());
        } catch (...) {
            if (recording) graph.end_recording(q_);
            throw;
        }
        std::fprintf(stderr, "[flint-gpu-decode] captured mode=%d tokens=%d\n", mode, steps);
    }
    q_.ext_oneapi_graph(*executable);
}
std::vector<int> FlintModel::generate_device(int max_tokens, float temperature,
                                            int top_k, float top_p, uint64_t& rng) {
    if (!device_generation_enabled()) throw std::logic_error("Flint GPU decode is disabled");
    if (max_tokens < 0 || !std::isfinite(temperature) || !std::isfinite(top_p))
        throw std::invalid_argument("invalid GPU generation parameters");
    const bool greedy = temperature <= 0.f || top_k == 1;
    const int k = greedy ? 1 : (top_k > 0 ? std::min(top_k, cfg_.vocab_size) : cfg_.vocab_size);
    if (k > flint::gpu_decode::kMaxTopK)
        throw std::invalid_argument("Flint GPU sampling supports top_k=1..64 (or greedy)");
    std::vector<int> result;
    if (!max_tokens || pos_ >= max_seq_) return result;
    if (!resident_decode_) resident_decode_ = std::make_shared<ResidentDecode>(q_, cfg_.eos_token_ids);
    auto& r = *resident_decode_;
    const bool parallel = k<=20 && flint::gpu_decode::parallel_sample_enabled();
    const int groups = flint::gpu_decode::sample_groups();
    const int mode = greedy?0:(!parallel?1:(groups==1?3:4));
    if(mode==4){
        if(!r.sample_values)r.sample_values=allocate<float>(q_,64*20);
        if(!r.sample_ids)r.sample_ids=allocate<int>(q_,64*20);
        if(r.sample_group_count!=groups){
            q_.wait_and_throw();
            for(int m:{4,9})for(auto& g:r.graphs[m])g.reset();
            r.sample_group_count=groups;
        }
    }
    flint::gpu_decode::State state;
    state.position = pos_; state.rng = rng;
    state.temperature = temperature; state.top_k = k; state.top_p = top_p;
    q_.memcpy(r.state, &state, sizeof(state)).wait_and_throw();
    while (int(result.size()) < max_tokens && state.active && state.position < max_seq_) {
        const int n = std::min({flint::gpu_decode::kBlock, max_tokens-int(result.size()),
                                max_seq_-state.position});
        run_device_block(n, mode,state.position);
        std::array<int, flint::gpu_decode::kBlock> ids;
        q_.memcpy(ids.data(), r.output, n * sizeof(int));
        q_.memcpy(&state, r.state, sizeof(state)).wait_and_throw();
        for (int i = 0; i < n; ++i) if (ids[i] >= 0) result.push_back(ids[i]);
    }
    int advanced = 0;
    for (int id : result) if (!info_.is_eos(id)) ++advanced;
    if (state.position != pos_ + advanced)
        throw std::runtime_error("Flint GPU decode position mismatch");
    pos_ = state.position; rng = state.rng;
    return result;
}
bool FlintModel::decode_repeated_device(int token, int steps, int past_len) {
    if (!device_generation_enabled()) return false;
    if (token < 0 || token >= cfg_.vocab_size || steps < 0 || past_len != pos_ ||
        steps > max_seq_-pos_) throw std::invalid_argument("invalid GPU repeated decode extent");
    if (!steps) return true;
    if (!resident_decode_) resident_decode_ = std::make_shared<ResidentDecode>(q_, cfg_.eos_token_ids);
    auto& r = *resident_decode_;
    flint::gpu_decode::State state;
    state.position = pos_; state.token = token;
    q_.memcpy(r.state, &state, sizeof(state)).wait_and_throw();
    for (int left = steps; left; ) {
        const int n = std::min(left, flint::gpu_decode::kBlock);
        run_device_block(n, 2,pos_+steps-left); left -= n;
    }
    q_.memcpy(&state, r.state, sizeof(state)).wait_and_throw();
    if (state.position != pos_ + steps) throw std::runtime_error("Flint GPU decode position mismatch");
    pos_ = state.position;
    return true;
}

void FlintModel::save_decode_prefix() {
    if (!device_generation_enabled()) throw std::logic_error("GPU prefix snapshots require GPU decode");
    if (!resident_decode_) resident_decode_ = std::make_shared<ResidentDecode>(q_,cfg_.eos_token_ids);
    auto& r = *resident_decode_;
    if (!r.prefix_initialized) {
        r.prefix_banks.clear();
        size_t elements = 0;
        for (const auto& c : kv_) {
            if (c.global) continue;
            const size_t n = size_t(cfg_.num_key_value_heads)*c.cap*cfg_.head_dim;
            r.prefix_banks.push_back({c.k,elements,n}); elements += n;
            r.prefix_banks.push_back({c.v,elements,n}); elements += n;
        }
        if (elements && !r.prefix_data) r.prefix_data = allocate<flint::bf16>(q_,elements);
        if (!r.prefix_logits) r.prefix_logits = allocate<float>(q_,cfg_.vocab_size);
        r.prefix_initialized = true;
        std::fprintf(stderr,"[flint-gpu-decode] prefix snapshot banks=%zu bytes=%zu (device only)\n",
                     r.prefix_banks.size(),elements*sizeof(flint::bf16)+size_t(cfg_.vocab_size)*sizeof(float));
    }
    // Global caches only append after the prefix; future entries are masked
    // and overwritten before use. Sliding rings must be restored completely.
    for (const auto& bank : r.prefix_banks)
        q_.memcpy(r.prefix_data+bank.offset,bank.data,bank.elements*sizeof(flint::bf16));
    q_.memcpy(r.prefix_logits,logits_dev_,size_t(cfg_.vocab_size)*sizeof(float));
    q_.wait_and_throw();
    r.prefix_position = pos_;
}

void FlintModel::restore_decode_prefix() {
    if (!resident_decode_ || resident_decode_->prefix_position < 0)
        throw std::logic_error("no saved GPU decode prefix");
    auto& r = *resident_decode_;
    for (const auto& bank : r.prefix_banks)
        q_.memcpy(bank.data,r.prefix_data+bank.offset,bank.elements*sizeof(flint::bf16));
    q_.memcpy(logits_dev_,r.prefix_logits,size_t(cfg_.vocab_size)*sizeof(float));
    q_.wait_and_throw();
    pos_ = r.prefix_position;
}

void FlintModel::clear_decode_graphs(){
    if(!resident_decode_)return;
    q_.wait_and_throw();
    for(auto& mode:resident_decode_->graphs)for(auto& graph:mode)graph.reset();
}
