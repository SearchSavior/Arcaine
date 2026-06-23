# Gemma4 Unified 12B — C++ inference engine (oneDNN + SYCL)

Inference-only implementation of **Gemma4 Unified 12B** in C++ targeting Intel
GPUs via oneDNN v3.12 + SYCL. Runs on CPU as a fallback. Supports text, image,
and audio inputs.

## Architecture notes

| Dimension | Value |
|---|---|
| Layers | 48 (40 sliding-window + 8 full-attention) |
| Hidden size | 3840 |
| Attention heads | 16 Q / 8 KV (sliding), 16 Q / 1 KV (full) |
| Head dim | 256 (sliding), 512 (full) |
| Sliding window | 1024 tokens |
| FFN intermediate | 15 360 |
| Vocab size | 262 144 |
| Dtype | BF16 weights + BF16 matmul via oneDNN |

### Non-obvious implementation details

- **Attention scale = 1.0** — Q and K are RMSNorm-ed per head (`q_norm` /
  `k_norm`), so the standard `1/sqrt(d)` factor is omitted.
- **v_norm** — V is separately unit-RMSNorm-ed (no learned scale) from the
  same `k_proj` output as K, before `k_norm` is applied. Full attention:
  `K = k_norm(k_proj(h))`, `V = v_norm(k_proj(h))`.
- **layer_scalar** — applied to the *full* hidden state after both attention
  and FFN residuals (`h *= scalar`), not to the deltas individually.
- **RoPE** — rotate-half convention (non-interleaved pairs `(x[i],
  x[i+d/2])`), freq denominator = `head_dim`. Sliding: `theta=1e4,
  partial=1.0`; full: `theta=1e6, partial=0.25` (64 active pairs of 256).

## Project layout

```
src/
├── config.hpp                  # ModelConfig parsed from config.json
├── tokenizer.{hpp,cpp}         # SentencePiece tokenizer
├── sampler.hpp                 # Greedy / top-k / top-p sampling
├── main.cpp                    # CLI, chat template, generation loop
├── safetensors.hpp             # Zero-copy safetensors mmap loader
├── gpu/
│   ├── buffer.hpp              # SYCL USM buffer wrapper (GpuBuffer<T>)
│   ├── engine.hpp              # Singleton sycl::queue
│   └── ops.hpp                 # oneDNN matmul wrappers (BF16)
├── kernels/
│   ├── rms_norm.hpp            # RMSNorm + rms_norm_no_scale (v_norm)
│   ├── rope.hpp                # rotate-half RoPE (sliding + full)
│   ├── embedding.hpp           # Token embedding lookup + scale
│   ├── elementwise.hpp         # add_inplace, scale_inplace, softmax, softcap
│   ├── attention_mask.hpp      # Causal mask with sliding window
│   └── scatter.hpp             # Vision/audio token scatter
├── model/
│   ├── weights.hpp             # Weight struct hierarchy
│   ├── weights_loader.cpp      # Safetensors → GPU upload
│   ├── kv_cache.hpp            # Per-layer KV cache (separate K and V)
│   ├── attention.{hpp,cpp}     # Sliding + full attention forward
│   ├── decoder_layer.hpp       # Single transformer layer
│   ├── ffn.{hpp,cpp}           # GeGLU FFN
│   ├── model.{hpp,cpp}         # Gemma4Model (embed → layers → lm_head)
│   ├── vision_embedder.{hpp,cpp}
│   └── audio_embedder.{hpp,cpp}
└── preprocess/
    ├── image_proc.{hpp,cpp}    # Image → patch tensors
    └── audio_proc.{hpp,cpp}    # Audio → mel features
```

## Build

```bash
# Inside the dev container (see .devops/ for setup)
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j"$(nproc)"
```

## Usage

```bash
./build/gemma4 --model /path/to/gemma-4-12B-it --prompt "What is the capital of France?"

# Optional flags
  --image <path>       PNG/JPEG image
  --audio <path>       Raw float32 PCM at 16 kHz
  --max-tokens <N>     Default: 200
  --temp <T>           Temperature (0.0 = greedy). Default: 1.0
  --top-k <K>          Default: 64
  --top-p <P>          Default: 0.95
  --seed <N>           RNG seed. Default: 42
```

The binary applies the Gemma4 IT chat template automatically and suppresses
thinking tokens (`enable_thinking=False`). EOS tokens: 1 (`<eos>`), 106
(`<turn|>`), 50 (`<|tool_response>`).

## Container setup

```bash
export RENDER_GID=$(getent group render | cut -d: -f3)
docker compose build
docker compose run --rm dev   # interactive shell in /workspace
```

Host requirements: Linux, Intel GPU, `i915`/`xe` driver, `/dev/dri` present,
user in `render` group.

If 0 GPU engines are detected (`sycl-ls` / `clinfo`), the runtime falls back
to CPU automatically.

## Notes

- oneDNN is built from source with `-DDNNL_CPU_RUNTIME=SYCL
  -DDNNL_GPU_RUNTIME=SYCL`. Mixing the binary distribution causes symbol
  conflicts — source build is required.
- A Rust layer is planned above the C++ engine. The image ships a Rust
  toolchain; expose a narrow `extern "C"` ABI from C++ and bind it from Rust
  via bindgen/cxx. Use `corrosion` to drive both from one `cmake --build`.
