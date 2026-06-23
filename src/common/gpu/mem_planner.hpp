#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include "../../diffusion_gemma/config.hpp"

// ---------------------------------------------------------------------------
// Additive memory planner — a tensor-graph model of the diffusion forward pass.
// The eager runtime uses this to pre-size the per-GPU activation arena, and the
// benchmark can still print it as an analysis view.  It re-encodes the same
// tensor shapes and lifetimes, then runs a ggml-gallocr-style liveness pass:
// every tensor carries a use-count (its consumers); storage is taken from a
// free-list arena and returned the moment a tensor's last consumer runs, so
// later tensors with disjoint lifetimes reuse it.  The result is the peak of
// simultaneously-live tensors — what a real graph allocator would need — versus
// the current scheme's "sum of stable named buffers".
// ---------------------------------------------------------------------------
namespace memplan {

struct Graph {
    struct Node {
        std::string name;
        size_t bytes = 0;            // 0 = bookkeeping-only (consumes, no storage)
        std::vector<int> inputs;     // node ids this node consumes
        long offset = -1;            // assigned by plan()
    };
    std::vector<Node> nodes;

    int add(std::string name, size_t bytes, std::vector<int> inputs = {}) {
        nodes.push_back({std::move(name), bytes, std::move(inputs), -1});
        return (int)nodes.size() - 1;
    }
};

// Run the liveness allocation; returns peak bytes and fills Node::offset.
// If `verbose`, prints a per-node alloc/free trace.
size_t plan(Graph& g, bool verbose = false);

// Sum of every node's storage (no reuse) — the "stable named buffer" baseline.
size_t no_reuse_bytes(const Graph& g);

// One decode step: canvas of `seq` positions attending to `enc_len` encoder KV,
// all decoder layers + the LM head.  Mirrors decode_step().
Graph build_decode_graph(const DiffConfig& cfg, int seq, int enc_len);

// One prefill chunk of `chunk` positions at absolute offset `past_len`
// (causal encoder layers, no head).  Mirrors encode_block().
Graph build_prefill_graph(const DiffConfig& cfg, int chunk, int past_len);

}  // namespace memplan
