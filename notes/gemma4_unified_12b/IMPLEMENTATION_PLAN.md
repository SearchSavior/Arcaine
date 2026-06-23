# Gemma4 Unified BF16 Inference — Implementation Plan
## oneDNN 3.12 + SYCL on Intel GPU

**Target**: Full multimodal autoregressive inference in BF16 — text, image, and audio.  
Vision and audio are encoder-free (lightweight projections, no separate towers).

---

## Confirmed Architecture (config.json + safetensors header)

| Parameter | Value |
|---|---|
| `hidden_size` | 3840 |
| `intermediate_size` | 15360 |
| `vocab_size` | 262,144 |
| `num_hidden_layers` | 48 |
| `tie_word_embeddings` | true (lm_head = embed_tokens.T) |
| `final_logit_softcapping` | 30.0 |
| `rms_norm_eps` | 1e-6 |
| Model file | `model.safetensors` (23 GB, 677 tensors, all BF16) |

### Attention Layers

**Sliding** (40 of 48 — indices not divisible by 6):

| Weight | Shape | |
|---|---|---|
| `q_proj.weight` | [4096, 3840] | 16 Q heads × 256 head_dim |
| `k_proj.weight` | [2048, 3840] | 8 KV heads × 256 |
| `v_proj.weight` | [2048, 3840] | |
| `o_proj.weight` | [3840, 4096] | |
| `q_norm.weight` / `k_norm.weight` | [256] | per-head QK RMSNorm |
| Sliding window | 1024 | |
| RoPE θ | 10,000 | full rotary |

**Full** (8 of 48 — indices 5,11,17,23,29,35,41,47):

| Weight | Shape | |
|---|---|---|
| `q_proj.weight` | [8192, 3840] | 16 Q heads × 512 global_head_dim |
| `k_proj.weight` | [512, 3840] | 1 KV head × 512 — **K=V, no v_proj** |
| `o_proj.weight` | [3840, 8192] | |
| `q_norm.weight` / `k_norm.weight` | [512] | |
| RoPE θ | 1,000,000 | partial_rotary_factor = 0.25 (first 128 of 512 dims) |

### FFN (all 48 layers)

```
FFN(x) = (GELU(x @ gate_proj.T) * (x @ up_proj.T)) @ down_proj.T
```

| Weight | Shape |
|---|---|
| `gate_proj.weight` | [15360, 3840] |
| `up_proj.weight` | [15360, 3840] |
| `down_proj.weight` | [3840, 15360] |

### Per-Layer Norms + Scalar (all 48 layers)

All RMSNorm, shape [3840]:  
`input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`  
`layer_scalar`: shape [1] BF16

### Vision Embedder Weights (CPU+GPU pipeline)

| Weight | Shape | Step |
|---|---|---|
| `vision_embedder.patch_ln1.{weight,bias}` | [6912] | LayerNorm on raw patches |
| `vision_embedder.patch_dense.{weight,bias}` | [3840, 6912] | Dense projection |
| `vision_embedder.patch_ln2.{weight,bias}` | [3840] | LayerNorm post-dense |
| `vision_embedder.pos_embedding` | [1120, 2, 3840] | Factorized 2D positional table |
| `vision_embedder.pos_norm.{weight,bias}` | [3840] | LayerNorm post-posemb |
| `embed_vision.embedding_projection.weight` | [3840, 3840] | Final linear to text space |
| (pre-projection RMSNorm) | — | Not in safetensors → identity scale (1.0) |

### Audio Embedder Weights

| Weight | Shape | Step |
|---|---|---|
| `embed_audio.embedding_projection.weight` | [3840, 640] | Linear from 640 raw-waveform dims |
| (pre-projection RMSNorm) | — | Not in safetensors → identity scale (1.0) |

### Residual Formula (Gemma2 dual-norm style)

```
attn_in  = input_layernorm(h)
attn_out = attention(attn_in)
attn_out = post_attention_layernorm(attn_out) * layer_scalar
h        = h + attn_out

ffn_in   = pre_feedforward_layernorm(h)
ffn_out  = ffn(ffn_in)
ffn_out  = post_feedforward_layernorm(ffn_out) * layer_scalar
h        = h + ffn_out
```

---

## Directory Layout

```
/workspace/
├── CMakeLists.txt               (extend existing)
├── third_party/
│   ├── nlohmann/json.hpp        (single-header JSON — vendor v3.11.3)
│   ├── mio/mio.hpp              (single-header mmap)
│   └── stb/stb_image.h         (single-header image decode — PNG/JPEG)
└── src/
    ├── main.cpp                 (CLI: --model, --prompt, --image, --audio, ...)
    │
    ├── config.hpp               (ModelConfig parsed from config.json)
    ├── safetensors.hpp          (mmap parser → name→{dtype, shape, byte_offset})
    ├── tokenizer.hpp/.cpp       (BPE from tokenizer.json)
    │
    ├── gpu/
    │   ├── engine.hpp           (dnnl::engine + dnnl::stream + sycl::queue singleton)
    │   ├── buffer.hpp           (GpuBuffer<T>: USM device alloc + host↔GPU copy)
    │   └── ops.hpp              (oneDNN primitive wrappers: matmul_bf16, layer_norm, softmax)
    │
    ├── kernels/
    │   ├── rms_norm.hpp         (SYCL kernel: RMSNorm in BF16)
    │   ├── layer_norm.hpp       (SYCL kernel: LayerNorm in BF16 — for vision pipeline)
    │   ├── rope.hpp             (SYCL kernel: RoPE — standard and partial)
    │   ├── attention_mask.hpp   (SYCL kernel: causal + sliding-window mask)
    │   ├── embedding.hpp        (SYCL kernel: gather + scale by sqrt(H))
    │   ├── scatter.hpp          (SYCL kernel: masked_scatter for multimodal tokens)
    │   └── elementwise.hpp      (SYCL kernels: gelu_tanh, mul, softcap, scale)
    │
    ├── preprocess/
    │   ├── image_proc.hpp/.cpp  (CPU: resize, patchify, patches_merge, position_ids)
    │   └── audio_proc.hpp/.cpp  (CPU: waveform → 640-sample frames)
    │
    └── model/
        ├── weights.hpp          (LayerWeights, GlobalWeights, VisionWeights, AudioWeights)
        ├── weights_loader.cpp   (mmap safetensors → upload BF16 to GPU)
        ├── kv_cache.hpp         (per-layer pre-allocated GPU buffers)
        ├── attention.hpp/.cpp   (sliding + full attention forward)
        ├── ffn.hpp/.cpp         (FFN forward)
        ├── vision_embedder.hpp/.cpp  (GPU forward for vision pipeline)
        ├── audio_embedder.hpp/.cpp   (GPU forward for audio pipeline)
        ├── decoder_layer.hpp    (single decoder layer)
        └── model.hpp/.cpp       (Gemma4Model: full forward + generate)
```

---

## Phase 0 — Infrastructure

### 0a. Vendored headers

Place in `third_party/`:
- `nlohmann/json.hpp` — JSON parsing (config, tokenizer, safetensors header)
- `mio/mio.hpp` — memory-mapped file I/O for the 23GB safetensors file
- `stb/stb_image.h` — decode PNG/JPEG images to raw pixels (single header, define `STB_IMAGE_IMPLEMENTATION` in one .cpp)

### 0b. CMakeLists.txt extension

```cmake
include_directories(third_party)

add_executable(gemma4
    src/main.cpp
    src/tokenizer.cpp
    src/preprocess/image_proc.cpp
    src/preprocess/audio_proc.cpp
    src/model/weights_loader.cpp
    src/model/attention.cpp
    src/model/ffn.cpp
    src/model/vision_embedder.cpp
    src/model/audio_embedder.cpp
    src/model/model.cpp
)
target_compile_options(gemma4 PRIVATE -fsycl -O2)
target_link_options(gemma4   PRIVATE -fsycl)
target_link_libraries(gemma4 PRIVATE DNNL::dnnl)
```

### 0c. safetensors.hpp

Safetensors format: `[8-byte LE u64 header_len][header JSON][data blob]`

```cpp
struct TensorView {
    std::string            dtype;   // "BF16"
    std::vector<int64_t>   shape;
    const void*            data;    // pointer into mmap region
    size_t                 nbytes;
};

class SafetensorsFile {
    mio::mmap_source mmap_;
    std::unordered_map<std::string, TensorView> tensors_;
public:
    explicit SafetensorsFile(const std::string& path);
    const TensorView& get(const std::string& name) const;
    bool              has(const std::string& name) const;
};
```

Parse steps:
1. `mio::mmap_source(path)` — zero-copy map of 23GB file
2. Read first 8 bytes as `uint64_t header_len`
3. `nlohmann::json::parse(base + 8, base + 8 + header_len)`
4. For each key (skip `__metadata__`): record `{dtype, shape, data = base + 8 + header_len + offsets[0]}`

### 0d. config.hpp

```cpp
struct TextConfig {
    int   hidden_size        = 3840;
    int   intermediate_size  = 15360;
    int   vocab_size         = 262144;
    int   num_hidden_layers  = 48;
    int   num_attn_heads     = 16;
    int   num_kv_heads       = 8;           // sliding layers
    int   head_dim           = 256;         // sliding layers
    int   global_head_dim    = 512;         // full attention layers
    int   num_global_kv_heads = 1;
    bool  attention_k_eq_v   = true;
    int   sliding_window     = 1024;
    float rms_norm_eps       = 1e-6f;
    float final_logit_softcapping = 30.0f;
    std::vector<bool> is_full_attention;    // [48] derived from layer_types
};

struct ModelConfig {
    TextConfig  text;
    std::string model_dir;
    static ModelConfig from_dir(const std::string& dir);
};
```

---

## Phase 1 — GPU Engine & Buffer

### 1a. engine.hpp — singleton

```cpp
struct GpuEngine {
    dnnl::engine engine;   // gpu, index 0
    dnnl::stream stream;
    sycl::queue  queue;    // dnnl::sycl_interop::get_queue(stream)

    static GpuEngine& get();
};
```

### 1b. buffer.hpp — typed USM allocation

```cpp
using bf16 = uint16_t;  // BF16 stored as raw bits

template<typename T>
class GpuBuffer {
    T*     ptr_  = nullptr;
    size_t count_ = 0;
public:
    GpuBuffer() = default;
    explicit GpuBuffer(size_t n);   // sycl::malloc_device<T>
    ~GpuBuffer();                   // sycl::free

    T*     data()  const { return ptr_; }
    size_t count() const { return count_; }

    void upload(const T* host, size_t n);  // queue.memcpy + wait
    void download(T* host, size_t n) const;
    void zero();
};
```

---

## Phase 2 — SYCL Kernels

All kernels use USM device pointers and are submitted to `GpuEngine::get().queue`.  
BF16 is accumulated in FP32 inside kernels for numerical stability.

### 2a. rms_norm.hpp

```cpp
// out = x / rms(x) * weight   (no bias, no mean subtraction)
// (seq_len, H) in-place or out-of-place
void rms_norm(
    sycl::queue& q,
    const bf16* x, const bf16* weight,
    bf16* out,
    int seq_len, int H, float eps
);
```

Work-group per token position. Reduce over H in local memory, broadcast rms scalar.

### 2b. layer_norm.hpp

```cpp
// Standard LayerNorm with weight and bias — for vision pipeline only
// (N, D) → normalized over D
void layer_norm(
    sycl::queue& q,
    const bf16* x, const bf16* weight, const bf16* bias,
    bf16* out,
    int N, int D, float eps
);
```

### 2c. embedding.hpp

```cpp
// out[i] = embed_table[ids[i]] * scale   (scale = sqrt(hidden_size))
void embedding_lookup(
    sycl::queue& q,
    const bf16* table,   // (vocab_size, H)
    const int32_t* ids,  // (seq_len,) device ptr
    bf16* out,           // (seq_len, H)
    int seq_len, int H, float scale
);
```

### 2d. rope.hpp

```cpp
// Apply RoPE in-place to Q and K.
// rotary_dims = floor(head_dim * partial_rotary_factor)
//   sliding: partial_rotary_factor=1.0, theta=10000
//   full:    partial_rotary_factor=0.25, theta=1e6
// q: (seq_len, nq_heads, head_dim), k: (seq_len, nkv_heads, head_dim)
// offset: past_seq_len (position in the full sequence)
void apply_rope(
    sycl::queue& q,
    bf16* q_ptr, bf16* k_ptr,
    int seq_len, int offset,
    int nq_heads, int nkv_heads, int head_dim,
    float rope_theta, float partial_rotary_factor
);
```

`inv_freq` is computed **inline per thread** in the kernel — `theta` and `rotary_dims` are scalar constants passed as kernel arguments. 
```
// inside SYCL kernel, for dim pair index i (i < rotary_dims/2):
float inv_freq = 1.0f / sycl::pown(rope_theta, 2*i / (float)rotary_dims);
float angle    = (float)(seq_offset + token_pos) * inv_freq;
float c = sycl::cos(angle), s = sycl::sin(angle);
// apply 2D rotation to (x[2i], x[2i+1])
float x0 = x[2*i], x1 = x[2*i+1];
x[2*i]   = x0*c - x1*s;
x[2*i+1] = x0*s + x1*c;
// dims >= rotary_dims: pass through unchanged
```

### 2e. attention_mask.hpp

```cpp
// Fill float mask (q_len, kv_len) with 0.0 or -INFINITY.
// Positions where kv_j > qi are masked (causal).
// For sliding: additionally mask kv_j < qi_pos - window + 1.
// past_offset: first KV position index in the cache.
void fill_causal_mask(
    sycl::queue& q,
    float* mask,
    int q_len, int kv_len,
    int past_offset,
    int sliding_window   // INT_MAX for full attention
);
```

### 2f. scatter.hpp

```cpp
// Replace positions where mask[i] == true in `seq_embeds` with rows from `modal_embeds`.
// seq_embeds: (total_seq, H) — all token embeddings (text + placeholders)
// modal_embeds: (num_modal_tokens, H) — vision or audio projected embeddings
// mask: (total_seq,) bool
void masked_scatter_bf16(
    sycl::queue& q,
    bf16* seq_embeds,
    const bf16* modal_embeds,
    const bool* mask,
    int total_seq, int H
);
```

### 2g. elementwise.hpp

```cpp
void gelu_tanh_inplace(sycl::queue& q, bf16* x, int n);
void mul_inplace(sycl::queue& q, bf16* a, const bf16* b, int n);  // a *= b
void scale_inplace(sycl::queue& q, bf16* x, int n, float s);
void softcap_inplace(sycl::queue& q, float* x, int n, float cap); // tanh clamp
```

### 2h. ops.hpp — oneDNN wrappers

```cpp
// BF16 matmul: C (M,N) = A (M,K) @ B^T (N,K)
// Weight is stored row-major (N,K); transposed in oneDNN via format_tag::ba on wei_md.
void matmul_bf16(
    const bf16* A, int M, int K,
    const bf16* B, int N,
    bf16* C
);

// FP32 softmax in-place over last axis: (batch, seq_len)
void softmax_f32(float* x, int batch, int seq_len);
```

oneDNN matmul setup (cache primitives keyed on {M,K,N} shape):
- `src_md`:  `[M, K]`, `bf16`, `ab`
- `wei_md`:  `[N, K]`, `bf16`, `ba`  (transposed)
- `dst_md`:  `[M, N]`, `bf16`, `ab`
- Scale factor: 1.0 (scaling done manually where needed)

---

## Phase 3 — Weight Loading

### 3a. weights.hpp

```cpp
struct SlidingAttnWeights {
    GpuBuffer<bf16> q_proj, k_proj, v_proj, o_proj;  // (4096,3840), (2048,3840)×2, (3840,4096)
    GpuBuffer<bf16> q_norm, k_norm;                   // (256,)
};

struct FullAttnWeights {
    GpuBuffer<bf16> q_proj, k_proj, o_proj;  // (8192,3840), (512,3840), (3840,8192)
    GpuBuffer<bf16> q_norm, k_norm;          // (512,)
    // No v_proj: K=V — k_proj serves as both
};

struct FfnWeights {
    GpuBuffer<bf16> gate_proj, up_proj, down_proj;  // (15360,3840), (15360,3840), (3840,15360)
};

struct LayerWeights {
    bool     is_full;
    std::variant<SlidingAttnWeights, FullAttnWeights> attn;
    FfnWeights ffn;
    GpuBuffer<bf16> input_ln, post_attn_ln, pre_ffn_ln, post_ffn_ln;  // (3840,) each
    float    layer_scalar;  // BF16 weight converted to float at load time
};

struct VisionWeights {
    // vision_embedder.*
    GpuBuffer<bf16> patch_ln1_w, patch_ln1_b;   // (6912,)
    GpuBuffer<bf16> patch_dense_w, patch_dense_b; // (3840, 6912)
    GpuBuffer<bf16> patch_ln2_w, patch_ln2_b;   // (3840,)
    GpuBuffer<bf16> pos_embedding;               // (1120, 2, 3840)
    GpuBuffer<bf16> pos_norm_w, pos_norm_b;      // (3840,)
    // embed_vision.*
    GpuBuffer<bf16> proj_w;                      // (3840, 3840) — final linear
    // pre_proj RMSNorm: identity scale (all 1s) — no weight stored
};

struct AudioWeights {
    GpuBuffer<bf16> proj_w;  // (3840, 640) — embed_audio.embedding_projection.weight
    // pre_proj RMSNorm: identity — no weight stored
};

struct GlobalWeights {
    GpuBuffer<bf16>     embed_tokens;   // (262144, 3840)
    GpuBuffer<bf16>     final_norm;     // (3840,)
    std::vector<LayerWeights> layers;   // [48]
    VisionWeights       vision;
    AudioWeights        audio;
};
```

### 3b. weights_loader.cpp

```cpp
GlobalWeights load_weights(const std::string& model_dir, const ModelConfig& cfg);
```

1. Open `SafetensorsFile(model_dir + "/model.safetensors")` — zero-copy mmap
2. For each tensor: `GpuBuffer<bf16>(n_elements)` then `gpu_buf.upload(mmap_ptr, n_elements)`
3. Submit all uploads to the oneDNN stream; call `stream.wait()` once at end
4. `layer_scalar`: read BF16 from mmap, convert to float: `float s = bf16_to_float(*(uint16_t*)ptr)`

BF16→float conversion (CPU utility):
```cpp
inline float bf16_to_float(uint16_t v) {
    uint32_t u = static_cast<uint32_t>(v) << 16;
    float f; std::memcpy(&f, &u, 4);
    return f;
}
```

---

## Phase 4 — KV Cache

```cpp
struct LayerKvCache {
    GpuBuffer<bf16> k;   // (max_seq_len, num_kv_heads, head_dim)
    GpuBuffer<bf16> v;   // same, or same pointer as k for K=V layers
    int filled = 0;
};

class KvCache {
    std::vector<LayerKvCache> layers_;
    int max_seq_len_;
public:
    // Allocate all layers upfront
    KvCache(const ModelConfig& cfg, int max_seq_len);

    void  append(int layer, const bf16* k_new, const bf16* v_new, int new_tokens);
    const bf16* k(int layer) const;
    const bf16* v(int layer) const;
    int         len(int layer) const;
    void        reset();
};
```

Allocation per layer:
- **Sliding** (40 layers): `(max_seq_len, 8, 256)` for K and V
- **Full** (8 layers): `(max_seq_len, 1, 512)` — K only; V is the same buffer (K=V)

Total KV cache at `max_seq_len=2048`:
- Sliding: 40 × 2 × (2048 × 8 × 256 × 2 bytes) ≈ 1.6 GB
- Full:    8 × 1 × (2048 × 1 × 512 × 2 bytes) ≈ 0.016 GB
- Total: ~1.6 GB

---

## Phase 5 — Attention Forward Pass

### Sliding attention

```
Given: hidden (seq, 3840), past_len, layer weights

1. Q = hidden @ q_proj.T          → (seq, 4096)
2. K = hidden @ k_proj.T          → (seq, 2048)
3. V = hidden @ v_proj.T          → (seq, 2048)

4. Reshape: Q→(seq,16,256), K→(seq,8,256), V→(seq,8,256)

5. RMSNorm Q per head (q_norm [256]):  for each of 16 heads
   RMSNorm K per head (k_norm [256]):  for each of 8 heads

6. RoPE(Q, K, offset=past_len, theta=10000, partial=1.0)

7. Append K,V to cache → cache holds (past_len+seq, 8, 256)
   kv_len = past_len + seq

8. GQA expand: each KV head group_size=2 serves 2 Q heads.
   Logically treat K,V as (kv_len, 16, 256) by repeating each head.
   Implement as index arithmetic: Q head h → KV head h/2.

9. scores = Q[qi,h,:] · K[kj,h/2,:] / sqrt(256)
            → (seq, 16, kv_len)  computed as batched matmul

10. Fill causal+sliding mask (window=1024) into float32 buffer (seq, kv_len)

11. scores += mask, softmax over kv_len → (seq, 16, kv_len)

12. ctx = softmax_scores @ V → (seq, 16, 256)   batched matmul

13. Reshape ctx → (seq, 4096)

14. attn_out = ctx @ o_proj.T → (seq, 3840)
```

### Full attention (K=V, 1 global KV head)

```
1. Q = hidden @ q_proj.T          → (seq, 8192)
2. K = hidden @ k_proj.T          → (seq, 512)
   V = K  (same buffer — K=V)

3. Reshape: Q→(seq,16,512), K→(seq,1,512)

4. RMSNorm Q per head (q_norm [512])
   RMSNorm K      (k_norm [512])

5. RoPE(Q, K, offset=past_len, theta=1e6, partial=0.25)
   Rotary dims = floor(512 * 0.25) = 128

6. Append K to cache (V is same pointer — no separate append)
   kv_len = past_len + seq

7. Broadcast K,V to all 16 Q heads via index: Q head h → KV head 0

8. scores = Q[qi,h,:] · K[kj,0,:] / sqrt(512)  → (seq, 16, kv_len)

9. Causal mask (no sliding window)

10. Softmax → (seq, 16, kv_len)

11. ctx = softmax_scores @ V → (seq, 16, 512)

12. Reshape → (seq, 8192)

13. attn_out = ctx @ o_proj.T → (seq, 3840)
```

**Batched matmul for attention**: Use separate `matmul_bf16` calls per head, or construct a batched matmul with correct strides. For the first implementation, loop over heads on CPU scheduling parallel SYCL submits. Optimise to a single strided batched matmul in the second pass.

---

## Phase 6 — FFN Forward Pass

```
gate = hidden @ gate_proj.T      → (seq, 15360)
up   = hidden @ up_proj.T        → (seq, 15360)
gelu_tanh_inplace(gate)
mul_inplace(gate, up)            gate *= up
out  = gate @ down_proj.T        → (seq, 3840)
```

`gate_proj` and `up_proj` matmuls both read `hidden` — submit them back-to-back before the GELU so the GPU can pipeline.

---

## Phase 7 — Decoder Layer

```cpp
void decoder_layer_forward(
    const LayerWeights& w,
    bf16* hidden,       // (seq, 3840) in-place
    LayerKvCache& kv,
    int seq_len, int past_len,
    const ModelConfig& cfg,
    GpuBuffer<bf16>& tmp   // (seq, 3840) scratch
);
```

```
tmp     = rms_norm(hidden, w.input_ln)
attn    = attention(tmp)                    [dispatch sliding or full]
attn    = rms_norm(attn, w.post_attn_ln)
scale_inplace(attn, seq*H, w.layer_scalar)
hidden += attn

tmp     = rms_norm(hidden, w.pre_ffn_ln)
ffn_out = ffn(tmp)
ffn_out = rms_norm(ffn_out, w.post_ffn_ln)
scale_inplace(ffn_out, seq*H, w.layer_scalar)
hidden += ffn_out
```

---

## Phase 8 — Vision Pipeline (Encoder-Free)

The vision pipeline is split into a **CPU preprocessing** step and a **GPU embedding** step.

### 8a. CPU Preprocessing — image_proc.hpp/.cpp

Input: decoded image pixels `(H, W, 3)` uint8 or float

**Step 1: Aspect-ratio-preserving resize**
```cpp
std::pair<int,int> compute_target_size(
    int h, int w,
    int patch_size = 16,
    int pooling_k  = 3,          // model_patch_size = 48
    int max_soft_tokens = 280    // max model patches
);
```
- `max_teacher_patches = max_soft_tokens * pooling_k * pooling_k = 2520`
- `target_px = max_teacher_patches * patch_size^2 = 2520 × 256 = 645,120`
- `factor = sqrt(target_px / (H × W))`
- Round down to nearest multiple of `patch_size × pooling_k = 48` in both dims

**Step 2: Patchify to 16×16 teacher patches**
```cpp
// Returns (num_teacher_patches, 768) float32
// num_teacher_patches = (H/16) × (W/16) = up to 2520
std::vector<float> patchify(
    const float* pixels, int H, int W,
    int patch_size = 16
);
// Also returns (num_teacher_patches, 2) position (x,y) array
std::vector<std::array<int,2>> patch_positions(int H, int W, int ps);
```

Reshape `(C, H, W)` → interleave into `(ph, pw, ps, ps, C)` → reshape to `(num_patches, 768)`.  
Rescale: divide raw uint8 pixels by 255.0 (no mean/std normalization for this model).

**Step 3: patches_merge — 3×3 spatial pooling**
```cpp
struct MergedPatches {
    std::vector<float>            data;     // (num_model_patches, 6912)
    std::vector<std::array<int,2>> positions; // (num_model_patches, 2) — (x/3, y/3)
};

MergedPatches patches_merge(
    const std::vector<float>& teacher_patches,   // (L, 768)
    const std::vector<std::array<int,2>>& pos_xy, // (L, 2)
    int target_length                             // L / 9 = num_model_patches
);
```

Algorithm (see architecture doc §Implementation Guide):
```
k = 3  (pooling kernel size)

For each patch at (x, y):
  kernel_idx   = (x/k, y/k)               <- which 3×3 block it belongs to
  max_x        = max(pos_xy[:,0]) + 1

  num_from_top_left = k*k * kernel_idx.x + k * max_x * kernel_idx.y
  within_kernel = (x%k, y%k)
  num_in_kernel = within_kernel.x + within_kernel.y * k

  target_order[i] = num_in_kernel + num_from_top_left

argsort(target_order) → perm         <- groups each 3×3 block contiguously

ordered = teacher_patches[perm]      <- (L, 768)
reshape ordered → (target_length, 9, 16, 16, 3)
reshape →         (target_length, 3, 3, 16, 16, 3)
permute axes (0,1,3,2,4,5)         → (target_length, 3,16, 3,16, 3)
reshape →         (target_length, 48, 48, 3) = (target_length, 6912)

new_pos = min(pos_xy[perm].reshape(target_length, 9, 2), axis=1) / k
```

**Step 4: Pad to max_soft_tokens**
```cpp
// Pad data with zeros to (280, 6912); pad positions with -1 for padded slots
void pad_patches(MergedPatches& mp, int target = 280);
```

**Step 5: Output**
```cpp
struct ImageInput {
    std::vector<float>            pixel_values;    // (280, 6912) padded
    std::vector<std::array<int,2>> position_ids;  // (280, 2)  -1 for padding
    int                           num_valid_patches;
};
```

### 8b. GPU Vision Embedder — vision_embedder.hpp/.cpp

```cpp
// Returns GPU buffer of shape (num_valid_patches, 3840)
GpuBuffer<bf16> vision_embedder_forward(
    const VisionWeights& w,
    const ImageInput& img,        // CPU-side, uploaded inside this function
    GpuEngine& gpu
);
```

Step-by-step:

```
Upload pixel_values (280, 6912) to GPU as BF16    [cpu float32 → bf16 conversion on upload]
Upload position_ids (280, 2) to GPU as int32

x = layer_norm(pixel_values, patch_ln1_w, patch_ln1_b)   [LayerNorm on (280, 6912)]

x = x @ patch_dense_w.T + patch_dense_b                   [(280, 3840)]
x = layer_norm(x, patch_ln2_w, patch_ln2_b)               [(280, 3840)]

# Factorized 2D positional embedding
# pos_embedding shape: (1120, 2, 3840)
# For each patch i, axis a in {0,1}: lookup pos_embedding[position_ids[i,a], a, :]
# Sum over axis: pos_emb[i] = pos_embedding[px, 0, :] + pos_embedding[py, 1, :]
# Zero out padded patches (position_ids == -1)
pos_emb = gather_pos_embedding(pos_embedding, position_ids)  [(280, 3840)]
x = layer_norm(x + pos_emb, pos_norm_w, pos_norm_b)

# Multimodal embedder: RMSNorm (identity scale) → Linear
# pre_proj RMSNorm: weights not in safetensors → use scale=1.0 (just normalize)
x = rms_norm(x, scale=1.0)                                   [(280, 3840)]
x = x @ proj_w.T                                             [(280, 3840)]

# Strip padding
valid_mask = (position_ids != -1).all(axis=-1)   [bool, (280,)]
x = x[valid_mask]                                 [(num_valid, 3840)]

return x
```

**SYCL kernel for positional embedding gather**:
```
gather_pos_embedding(
    const bf16* table,          // (1120, 2, 3840)
    const int32_t* pos_ids,     // (280, 2) — (px, py) per patch; -1=padding
    bf16* out,                  // (280, 3840)
    int num_patches, int table_size  // 280, 1120
):
    for each patch i:
        px = pos_ids[i, 0]; py = pos_ids[i, 1]
        valid = (px != -1)
        for each hidden dim d:
            out[i,d] = valid ? table[px, 0, d] + table[py, 1, d] : 0.0
```

### 8c. Scatter vision embeddings into token sequence

After building the full prompt embedding sequence `(seq_len, 3840)`:

```
1. Identify image placeholder positions: bool mask where input_ids == IMAGE_TOKEN_ID (258880)
2. Count positions: must equal num_valid_patches
3. masked_scatter(seq_embeds, vision_embeds, image_mask)
```

---

## Phase 9 — Audio Pipeline (Encoder-Free)

The audio pipeline is even simpler — just frame chunking and a linear projection.

### 9a. CPU Preprocessing — audio_proc.hpp/.cpp

```cpp
struct AudioInput {
    std::vector<float> frames;     // (num_frames, 640) — raw waveform chunks
    int                num_frames;
};

AudioInput preprocess_audio(
    const float* waveform, int num_samples,
    int samples_per_frame = 640   // 40ms @ 16kHz
);
```

1. Pad `waveform` to nearest multiple of 640 with zeros
2. Reshape to `(num_frames, 640)`
3. No mel-spectrogram, no FFT, no normalization — raw float values passed through

### 9b. GPU Audio Embedder — audio_embedder.hpp/.cpp

```cpp
GpuBuffer<bf16> audio_embedder_forward(
    const AudioWeights& w,
    const AudioInput& audio,
    GpuEngine& gpu
);
```

```
Upload frames (num_frames, 640) to GPU as BF16

# Pre-projection RMSNorm — identity scale (weights not stored)
x = rms_norm(frames, scale=1.0)         [(num_frames, 640)]

# Linear projection
x = x @ proj_w.T                        [(num_frames, 3840)]

return x   # (num_frames, 3840)
```

### 9c. Scatter audio embeddings

Same pattern as vision:
```
audio_mask = (input_ids == AUDIO_TOKEN_ID)  # 258881
masked_scatter(seq_embeds, audio_embeds, audio_mask)
```

---

## Phase 10 — Full Model Forward Pass

```cpp
class Gemma4Model {
    ModelConfig     cfg_;
    GlobalWeights   weights_;
    KvCache         kv_cache_;
    GpuBuffer<bf16> hidden_;      // (max_seq, 3840)
    GpuBuffer<bf16> tmp_;         // (max_seq, 3840) scratch
    GpuBuffer<float> logits_buf_; // (vocab_size,)

public:
    Gemma4Model(const std::string& model_dir, int max_seq_len = 2048);

    // Returns float logits (vocab_size,) for the last token
    std::vector<float> forward(
        const std::vector<int>&    token_ids,
        int                        past_len,
        const std::vector<ImageInput>* images = nullptr,  // optional
        const std::vector<AudioInput>* audio  = nullptr   // optional
    );
};
```

Forward pass (with multimodal):

```
1. Upload token_ids to GPU (int32)

2. Embedding lookup + scale
   hidden = embed_tokens[token_ids] * sqrt(3840)    (seq, 3840)

3. If images provided:
     vision_embeds = vision_embedder_forward(image)   (num_valid, 3840)
     masked_scatter(hidden, vision_embeds, image_mask)

4. If audio provided:
     audio_embeds = audio_embedder_forward(audio)     (num_frames, 3840)
     masked_scatter(hidden, audio_embeds, audio_mask)

5. For each of 48 decoder layers:
     decoder_layer_forward(weights_.layers[i], hidden, kv[i], seq, past_len)

6. last = hidden[seq-1, :]                   (1, 3840)
   norm  = rms_norm(last, weights_.final_norm)
   logits_bf16 = norm @ embed_tokens.T       (1, vocab_size)  — tied weights
   convert logits_bf16 → logits_f32 on GPU (SYCL kernel)
   softcap_inplace(logits_f32, vocab_size, cap=30.0)
   download logits_f32 to host

return logits_f32
```

---

## Phase 11 — Tokenizer

### BPE from tokenizer.json

```cpp
class Tokenizer {
    std::unordered_map<std::string,int>     vocab_;       // piece → id
    std::vector<std::string>                id_to_piece_; // id → piece
    std::unordered_map<std::string,int>     merge_rank_;  // "A B" → priority
public:
    static Tokenizer from_json(const std::string& path);
    std::vector<int> encode(const std::string& text, bool add_bos = true) const;
    std::string      decode(const std::vector<int>& ids) const;
};
```

`tokenizer.json` structure:
- `model.vocab`: `{"<piece>": id, ...}` — 262,144 entries
- `model.merges`: `["A B", ...]` — ordered list; index = priority

BPE encode:
1. Apply byte-level pre-tokenization (map each byte to a symbol if char not in vocab)
2. Initialize token list from individual characters/bytes
3. Repeat: find the merge with the lowest rank (highest priority) among adjacent pairs → apply it → until no more merges apply

Special token IDs hardcoded:
```cpp
static constexpr int BOS_ID       = 2;
static constexpr int EOS_ID       = 1;
static constexpr int PAD_ID       = 0;
static constexpr int IMAGE_TOK_ID = 258880;
static constexpr int AUDIO_TOK_ID = 258881;
static constexpr int VIDEO_TOK_ID = 258884;
```

---

## Phase 12 — Sampler

```cpp
int sample_token(
    const float* logits,   // (vocab_size,) — already softcapped
    int vocab_size,
    float temperature,     // default 1.0
    int top_k,             // default 64
    float top_p,           // default 0.95
    std::mt19937& rng
);
```

1. Divide all logits by `temperature`
2. `std::partial_sort` to find top-K largest logits (O(V log K))
3. Softmax over top-K only
4. Sort top-K by probability descending; accumulate until ≥ top_p; zero the rest (nucleus)
5. Renormalize; `std::discrete_distribution` → sample index

---

## Phase 13 — Generation Loop & CLI

```cpp
// main.cpp
int main(int argc, char** argv) {
    // Parse: --model, --prompt, --image, --audio,
    //        --max-tokens (200), --temp (1.0), --top-k (64), --top-p (0.95)

    auto cfg   = ModelConfig::from_dir(model_dir);
    auto model = Gemma4Model(model_dir, /*max_seq=*/2048);
    auto tok   = Tokenizer::from_json(model_dir + "/tokenizer.json");

    auto tokens = tok.encode(prompt, /*add_bos=*/true);

    // Load image if provided
    std::optional<ImageInput> img;
    if (!image_path.empty()) {
        // Insert IMAGE_TOK_ID placeholders into tokens (num_valid_patches times)
        // Preprocess image
        img = preprocess_image(image_path);
        // Insert placeholder tokens at image position in `tokens`
    }

    // Prefill
    auto logits = model.forward(tokens, 0,
                                img ? &std::vector{*img} : nullptr);
    int past = tokens.size();

    std::mt19937 rng(42);
    std::vector<int> generated;

    for (int step = 0; step < max_new_tokens; ++step) {
        int next = sample_token(logits.data(), logits.size(),
                                temp, top_k, top_p, rng);
        if (next == EOS_ID || next == 106 /* alt EOS */) break;
        generated.push_back(next);
        logits = model.forward({next}, past);
        past++;
    }

    std::cout << tok.decode(generated) << "\n";
}
```

---

## Build Milestones

| Milestone | Tests |
|---|---|
| **M1 — Foundation** | safetensors parses all 677 tensors; config loads; tokenizer round-trips |
| **M2 — Weight Load** | All tensors on GPU without OOM; spot-check byte equality |
| **M3 — Ops** | rms_norm / layer_norm / matmul match CPU reference (≤1% relative error in BF16) |
| **M4 — Vision CPU** | patches_merge output matches Python reference on a test image |
| **M5 — Single Layer** | One sliding + one full layer forward; outputs finite |
| **M6 — Full Forward** | Prefill 16 tokens + multimodal; logits finite and softcapped |
| **M7 — Generation** | "The capital of France is" → coherent completion; tokens/sec logged |

---

## oneDNN Usage Summary

| Operation | Approach |
|---|---|
| Linear layers (Q/K/V/O, gate/up/down, vision proj, audio proj, lm_head) | `dnnl::matmul` BF16 |
| Attention softmax | `dnnl::softmax_forward` FP32 (or custom SYCL) |
| RMSNorm (LM layers, vision pre-proj) | Custom SYCL kernel |
| LayerNorm (vision pipeline: patch_ln1/2, pos_norm) | Custom SYCL kernel |
| GELU, mul, scale, softcap, embedding gather | Custom SYCL kernels |
| RoPE, attention masking, masked_scatter | Custom SYCL kernels |

---

## Memory Budget

| Component | Size |
|---|---|
| Model weights | ~23 GB |
| KV cache (max_seq=2048) | ~1.6 GB |
| Activations / scratch (seq=2048) | ~0.06 GB |
| Vision patch buffer (280 × 6912 × 2) | ~3.9 MB |
| **Total** | **~24.7 GB** |

GPU is Arc B70 32GB VRAM.

---

## Second-Pass Optimizations (post-MVP)

1. **Fused SDPA** — replace manual QK/softmax/@V with `dnnl::sdpa` primitive
2. **Ring-buffer KV cache** — proper O(window) memory for sliding layers at long contexts
3. **Fused post-ops** — attach GELU/scale as oneDNN matmul post-ops to reduce kernel launches
4. **Batched strided matmul** — single batched matmul for all attention heads instead of per-head dispatch
5. **Video pipeline** — per-frame patchify + frame-level position tracking (uses same vision embedder code path)
6. **INT4/INT8 quantization** — for sub-16 GB VRAM deployment
