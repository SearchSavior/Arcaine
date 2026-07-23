#include "mem_planner.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace memplan {
namespace {

// ggml_dyn_tallocr-style free-list arena (analysis only — tracks offsets/peak,
// never allocates real memory).  Best-fit alloc, coalescing free; `peak` is the
// high-water mark of allocated address space.
struct DynArena {
    struct Block { size_t off, size; };
    std::vector<Block> free;
    size_t peak = 0;
    static constexpr size_t ALIGN = 256;

    DynArena() { free.push_back({0, (size_t)1 << 60}); }  // one huge trailing block

    static size_t aligned(size_t s) { return ((s ? s : 1) + ALIGN - 1) & ~(ALIGN - 1); }

    size_t alloc(size_t size) {
        size = aligned(size);
        int best = -1; size_t best_sz = SIZE_MAX;
        for (int i = 0; i < (int)free.size(); ++i)
            if (free[i].size >= size && free[i].size < best_sz) { best = i; best_sz = free[i].size; }
        size_t off = free[best].off;
        free[best].off += size;
        free[best].size -= size;
        if (free[best].size == 0) free.erase(free.begin() + best);
        peak = std::max(peak, off + size);
        return off;
    }

    void release(size_t off, size_t size) {
        size = aligned(size);
        int i = 0;
        while (i < (int)free.size() && free[i].off < off) ++i;
        free.insert(free.begin() + i, {off, size});
        if (i + 1 < (int)free.size() && free[i].off + free[i].size == free[i + 1].off) {
            free[i].size += free[i + 1].size; free.erase(free.begin() + i + 1);
        }
        if (i > 0 && free[i - 1].off + free[i - 1].size == free[i].off) {
            free[i - 1].size += free[i].size; free.erase(free.begin() + i);
        }
    }
};

int round_up(int v, int m) { return (v + m - 1) / m * m; }

constexpr size_t B = 2;  // bf16 bytes
constexpr size_t kScoresBudgetElems = (size_t)256 * 1024 * 1024;  // matches attention.cpp

// Emit one transformer block's tensors (shapes/lifetimes mirror diff_layer_forward
// + dual_ffn_forward + dense_mlp + the MoE expert workspace).  `hidden` is the
// persistent residual stream, referenced (kept live) by every block.
void add_block(Graph& g, const DiffConfig& cfg, int layer, int hidden,
               int seq, int kv_len, bool causal) {
    const auto& t = cfg.text;
    bool nvfp4 = cfg.is_nvfp4_quantized();
    bool direct_decode_kv = !causal;  // decoder (non-causal) always direct-caches now
    bool fused_int4_attn = cfg.is_int4_quantized() && !direct_decode_kv;
    int H = t.hidden_size, nq = t.num_attn_heads;
    bool full = t.is_full_attention[layer];
    int nkv = full ? t.num_global_kv_heads : t.num_kv_heads;
    int hd  = full ? t.global_head_dim     : t.head_dim;

    int tq = (int)std::max<size_t>(1, kScoresBudgetElems / ((size_t)nq * kv_len));
    tq = std::min(tq, seq);
    int kb = kv_len;  // decode (bidirectional) reads all; encoder full ≈ all
    if (causal && !full)
        kb = std::min(kv_len, t.sliding_window + tq);  // sliding-window band

    // --- attention ---
    int tmp    = g.add("tmp",      (size_t)seq * H * B,       {hidden});
    std::vector<int> qkv_inputs{tmp};
    if (fused_int4_attn) {
        size_t fused_dim = (size_t)nq * hd + (full ? 1 : 2) * (size_t)nkv * hd;
        int fused = g.add(full ? "aQK_fused" : "aQKV_fused",
                          (size_t)seq * fused_dim * B, {tmp});
        qkv_inputs = {fused};
    }
    int aQ     = g.add("aQ",       (size_t)seq * nq * hd * B, qkv_inputs);
    int aK     = g.add("aK",       direct_decode_kv ? 0 : (size_t)seq * nkv * hd * B, qkv_inputs);
    int aV     = g.add("aV",       direct_decode_kv ? 0 : (size_t)seq * nkv * hd * B, qkv_inputs);
    int aQhm   = g.add("aQ_hm",    (size_t)nq * tq * hd * B,  {aQ});
    int scores = g.add("aScores",  (size_t)nq * tq * kb * B,  {aQhm, aK});
    int actxhm = g.add("aCtx_hm",  (size_t)nq * tq * hd * B,  {scores, aV});
    int actxtm = g.add("aCtx_tm",  (size_t)seq * nq * hd * B, {actxhm});
    int attnout= g.add("attn_out", (size_t)seq * H * B,       {actxtm});

    // --- dual FFN ---
    int x1 = g.add("x1", (size_t)seq * H * B, {hidden, attnout});  // also frees attn_out
    int x2 = g.add("x2", (size_t)seq * H * B, {hidden});
    int rn = g.add("rn", (size_t)seq * H * B, {hidden});

    // dense MLP
    int inter = t.intermediate_size;
    int mlp;
    if (nvfp4) {
        int xpk = g.add("dense_xpacked", (size_t)seq * H / 2,        {x1});
        int xsc = g.add("dense_xscale",  (size_t)seq * (H / 16),     {x1});
        int gu  = g.add("dense_gate_up", (size_t)seq * 2 * inter * B,{xpk, xsc});
        int da  = g.add("dense_act",     (size_t)seq * inter * B,    {gu});
        mlp = g.add("mlp_out",           (size_t)seq * H * B,        {da});
    } else {
        int dg = g.add("dense_gate",     (size_t)seq * inter * B,    {x1});
        int du = g.add("dense_up",       (size_t)seq * inter * B,    {x1});
        int da = g.add("dense_act",      (size_t)seq * inter * B,    {dg, du});
        mlp = g.add("mlp_out",           (size_t)seq * H * B,        {da});
    }

    // MoE: router + expert pipeline (total_rows estimate per eager tail/hot
    // layout; data-dependent hot overflow omitted — this is the tail-cap floor).
    int E = t.num_experts, mi = t.moe_intermediate_size;
    int Tcap = std::min(64, round_up(seq, 8));
    int total_rows = E * Tcap;
    int rsc = g.add("router_scores", (size_t)seq * E * B,             {rn});
    int Xe  = g.add("moe_Xe",        (size_t)total_rows * H * B,      {x2});
    int Ye;
    if (nvfp4) {
        int xpk = g.add("moe_xpacked",   (size_t)total_rows * H / 2,       {Xe});
        int xsc = g.add("moe_xscale",    (size_t)total_rows * (H / 16),    {Xe});
        int mgu = g.add("moe_gu",        (size_t)total_rows * 2 * mi * B,  {xpk, xsc});
        int mac = g.add("moe_act",       (size_t)total_rows * mi * B,      {mgu});
        int apk = g.add("moe_actpacked", (size_t)total_rows * mi / 2,      {mac});
        int asc = g.add("moe_actscale",  (size_t)total_rows * (mi / 16),   {mac});
        Ye = g.add("moe_Ye",             (size_t)total_rows * H * B,       {apk, asc});
    } else {
        int mgu = g.add("moe_gu",        (size_t)total_rows * 2 * mi * B, {Xe});
        int mac = g.add("moe_act",       (size_t)total_rows * mi * B,     {mgu});
        Ye = g.add("moe_Ye",             (size_t)total_rows * H * B,      {mac});
    }
    int moe = g.add("moe_out",       (size_t)seq * H * B,             {Ye, x2});

    // residual write-back (in place into hidden): frees mlp_out, moe_out.
    g.add("ffn_resid", 0, {mlp, moe, rsc, hidden});
}

Graph build(const DiffConfig& cfg, int seq, int kv_len, bool causal, bool head) {
    const auto& t = cfg.text;
    Graph g;
    int hidden = g.add("hidden", (size_t)seq * t.hidden_size * B);

    if (head) {
        int H = t.hidden_size, inter = t.intermediate_size;
        int scn = g.add("selfcond_norm",    (size_t)seq * H * B);
        int scg = g.add("selfcond_gate_up", (size_t)seq * 2 * inter * B, {scn});
        int sca = g.add("selfcond_act",     (size_t)seq * inter * B,     {scg});
        int sco = g.add("selfcond_out",     (size_t)seq * H * B,         {sca});
        g.add("selfcond_resid", 0, {hidden, sco});
    }

    for (int l = 0; l < t.num_hidden_layers; ++l)
        add_block(g, cfg, l, hidden, seq, kv_len, causal);

    if (head) {
        int V = t.vocab_size, H = t.hidden_size;
        int logits = g.add("logits",     (size_t)seq * V * B, {hidden});
        int pbf16  = g.add("probs_bf16", (size_t)seq * V * B, {logits});
        g.add("sample", 0, {logits});
        g.add("soft_next", (size_t)seq * H * B, {pbf16});
    }
    return g;
}

}  // namespace

size_t plan(Graph& g, bool verbose) {
    int n = (int)g.nodes.size();
    std::vector<int> remaining(n, 0);
    for (auto& node : g.nodes)
        for (int in : node.inputs) ++remaining[in];

    DynArena a;
    std::vector<long> off(n, -1);
    for (int i = 0; i < n; ++i) {
        if (g.nodes[i].bytes > 0) {
            off[i] = (long)a.alloc(g.nodes[i].bytes);
            g.nodes[i].offset = off[i];
            if (verbose)
                std::printf("  [%3d] alloc %-14s %8.2f MB @ %ld  (peak %.2f MB)\n",
                            i, g.nodes[i].name.c_str(), g.nodes[i].bytes / 1048576.0,
                            off[i], a.peak / 1048576.0);
        }
        for (int in : g.nodes[i].inputs)
            if (--remaining[in] == 0 && g.nodes[in].bytes > 0)
                a.release((size_t)off[in], g.nodes[in].bytes);
    }
    return a.peak;
}

size_t no_reuse_bytes(const Graph& g) {
    size_t s = 0;
    for (const auto& node : g.nodes) s += node.bytes;
    return s;
}

Graph build_decode_graph(const DiffConfig& cfg, int seq, int enc_len) {
    return build(cfg, seq, enc_len + seq, /*causal=*/false, /*head=*/true);
}

Graph build_prefill_graph(const DiffConfig& cfg, int chunk, int past_len) {
    return build(cfg, chunk, past_len + chunk, /*causal=*/true, /*head=*/false);
}

}  // namespace memplan
