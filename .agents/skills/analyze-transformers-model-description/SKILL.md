---
name: analyze-transformers-model-description
description: >
  Analyze a Hugging Face transformers model and implement its inference pipeline in SYCL.
  Use this skill whenever the task involves porting, implementing, or optimizing a
  transformers architecture (LLM, VLM, encoder, or any modality) as SYCL kernels or a
  SYCL runtime — including when the user mentions oneAPI, DPC++, Intel GPUs, Level Zero,
  a "custom inference engine", or asks to run a Hugging Face checkpoint without Python.
  Also use it at the start of such tasks to query the available hardware and device
  architecture so the implementation is hardware-aware from the beginning
---

---
name: analyze-transformers-model-description
description: >
  Analyze a Hugging Face transformers model and implement its inference pipeline in SYCL.
  Use this skill whenever the task involves porting, implementing, or optimizing a
  transformers architecture (LLM, VLM, encoder, or any modality) as SYCL kernels or a
  SYCL runtime — including when the user mentions oneAPI, DPC++, Intel GPUs, Level Zero,
  a "custom inference engine", or asks to run a Hugging Face checkpoint without Python.
  Builds on the hardware report from the check-available-hardware skill so the
  implementation is hardware-aware from the beginning.
---

# Transformers → SYCL implementation

This skill covers the full path from a Hugging Face checkpoint to a correct SYCL
implementation: reviewing the hardware report, architecture analysis, weight loading (including
quantized checkpoints), a correctness-first kernel plan, and validation against the
reference implementation.

Work through the phases in order. Do not start writing kernels before Phases 1–3 are
done — most failed ports come from guessing at the architecture or the hardware instead
of reading both.

## Phase 1 — Review the hardware report

Device discovery is handled by the `check-available-hardware` skill, which should
already have run before this one. The target is always an Intel Xe GPU on the Level
Zero backend. Do not re-enumerate devices or write probe programs; read the report that
skill produced and pull out the values that will drive kernel decisions:

- **Xe generation** (Xe-HPG, Xe-HPC, …) and Xe-core/EU count.
- **Sub-group sizes** available (typically 8/16/32) and max work-group size.
- **Local (shared) memory per work-group** and max USM allocation size.
- **fp16 / bf16 support** and whether XMX is usable via
  `sycl::ext::oneapi::experimental::matrix::joint_matrix`.
- **oneMKL / oneDNN availability** — prefer library GEMMs over hand-written ones unless
  the task explicitly requires custom kernels.

If the hardware report is missing, run `check-available-hardware` first rather than
probing manually. Kernel choices later (tile sizes, sub-group size, whether to use
joint_matrix, whether activations fit in local memory) must reference the reported
values rather than assumed ones.

## Phase 2 — Analyze the model from its files

Assume the weights and Hugging Face files for the target model are available locally
(config, tokenizer files, safetensors/GGUF shards, processor configs). Inventory them
first:

- `config.json` — the single source of truth for the architecture.
- `model.safetensors.index.json` + shards (or a single `model.safetensors`, or a GGUF
  file) — enumerate every tensor name, shape, and dtype.
- `tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json`.
- For multimodal models: `preprocessor_config.json`, `processor_config.json`,
  `chat_template.json`, vision/audio tower configs (often nested inside `config.json`
  as `vision_config`, `audio_config`, `text_config`).
- `generation_config.json` — default sampling parameters and EOS token ids.

**Never hardcode a value that is present in `config.json`; always read it from the
file.** This includes hidden size, layer count, head counts (`num_attention_heads` vs
`num_key_value_heads` — GQA/MQA), intermediate size, vocab size, RoPE parameters
(`rope_theta`, `rope_scaling` and its type: linear / dynamic / yarn / llama3),
normalization epsilon, activation function, sliding-window sizes, tied embeddings
(`tie_word_embeddings`), and quantization metadata (`quantization_config`). The loader
should fail loudly if an expected key is missing rather than silently substituting a
default — silent defaults are the most common source of "almost right" outputs.

Dump the full tensor listing (name → shape → dtype) to a file. Cross-check it against
`config.json`: does the number of layer blocks match `num_hidden_layers`? Do projection
shapes match `hidden_size`, `num_key_value_heads * head_dim`, `intermediate_size`?
Any mismatch means the architecture understanding is wrong — resolve it before coding.

## Phase 3 — Study the reference transformers code

Define the architecture and its data flow from the actual `modeling_*.py` for this
model class (in the installed `transformers` package or the model repo's
`trust_remote_code` files), not from memory of a similar model. Families differ in
small ways that break correctness: pre- vs post-norm, RMSNorm vs LayerNorm, gated MLP
(SwiGLU/GeGLU) vs plain, parallel vs sequential attention+MLP blocks, QK-norm,
attention softcapping, partial rotary dimensions, attention sinks, MoE routing.

Produce a written data-flow document (`architecture.md`) covering:

1. **Tokenization / preprocessing** for every modality the model supports — text
   tokenizer algorithm, image preprocessing (resize, normalization constants, patching),
   audio frontend, and how modalities are merged into the embedding sequence
   (e.g., image token placeholders replaced by projected vision features).
2. **Per-layer forward pass**, as explicit math with tensor shapes at each step,
   including exactly where residual adds and norms occur, RoPE application, KV-cache
   read/write, and the attention masking scheme (causal, sliding-window, block-diagonal
   for packed multimodal inputs).
3. **The head(s)**: final norm, LM head (tied or not), and any generation-time details
   (logit softcapping, temperature/top-k/top-p defaults from `generation_config.json`).
4. **Numerics**: which parts the reference computes in fp32 even when weights are
   fp16/bf16 (typically softmax, norms, RoPE, logits accumulation). Match these choices
   — they matter for correctness tolerances.

**Implement all modalities the target model supports.** If it is a VLM, the vision
tower, projector, and multimodal merging are in scope, not just the language decoder.
If a modality genuinely cannot be implemented (e.g., missing preprocessing dependency),
state this explicitly and get confirmation rather than silently shipping text-only.

## Phase 4 — Weight loading and quantization

- Write a loader that maps checkpoint tensor names to your implementation's parameters
  mechanically (a name-mapping table), validating shape and dtype for every tensor.
  Refuse to run with missing or unconsumed tensors.
- **The checkpoint may be quantized — use those weights as-is.** Read
  `quantization_config` (GPTQ, AWQ, bitsandbytes, FP8, compressed-tensors) or the GGUF
  tensor types, and implement the corresponding dequantization or, preferably,
  quantized compute:
  - Understand the exact packing: group size, zero-point representation
    (symmetric/asymmetric), scale dtype and layout, per-channel vs per-group, and any
    permutation the format applies (GPTQ `g_idx`/act-order reordering is a classic
    correctness trap).
  - For a first correct version it is acceptable to dequantize to fp16/fp32 on load and
    run dense kernels; keep on-the-fly quantized kernels as a later optimization, but
    verify the dequantization itself against the reference library's dequant output
    tensor-by-tensor before anything else.
  - Do not "re-quantize" or convert precision in ways that change the numbers the
    checkpoint defines.
- Respect the checkpoint's dtypes elsewhere too: bf16 weights should not be silently
  read as fp16.

## Phase 5 — Plan kernels: correctness before optimization

Decide where custom SYCL kernels are needed based on **correctness requirements, not
early optimization**. Concretely:

- Enumerate the operator set the data-flow document requires (embedding lookup, GEMM,
  norm, RoPE, attention, activation, elementwise adds, sampling, plus modality-specific
  ops like conv/patchify for vision).
- For each operator, choose the simplest implementation that is correct on the target
  hardware from Phase 1: a oneMKL/oneDNN call, a naive-but-clear SYCL kernel, or (only
  when the op doesn't exist elsewhere) a custom kernel. Record the choice and the reason.
- Fused kernels, joint_matrix/XMX paths, flash-attention-style tiling, and quantized
  matmuls are *optimizations*: they come only after the naive path matches the
  reference. Keep the naive path in the codebase behind a flag so every optimization
  can be A/B-verified against it.
- Match the reference's fp32 accumulation points identified in Phase 3.

## Phase 6 — Validate against the reference

Correctness is defined as matching the Hugging Face implementation, layer by layer:

1. Run the reference model in Python once and dump intermediate activations
   (embeddings out, each block's output, final norm, logits) for a small fixed prompt
   — and for each modality (a fixed test image/audio clip).
2. Compare your implementation's tensors at the same points. Compare in order from
   input to output so the first diverging layer localizes the bug. Use tolerances
   appropriate to the dtype (e.g., fp32 vs fp32 should be tight; fp16 accumulation
   differences are expected — compare relative error and top-k logit agreement, not
   bitwise equality).
3. End-to-end: greedy decoding must produce the same token sequence as the reference
   (greedy removes sampling nondeterminism). Then verify KV-cache decode matches
   no-cache prefill recomputation.
4. Only after all of the above, benchmark and optimize — re-running the comparison
   suite after every kernel change.

## Working style

- Keep the two artifacts this skill owns current as understanding evolves:
  `architecture.md` and the tensor-mapping table. Together with the hardware report
  from `check-available-hardware`, they are the contract between analysis and
  implementation.
- When the transformers code and your expectation disagree, the code wins; when the
  checkpoint files and the docs disagree, the files win.
- Surface every assumption you were forced to make (unsupported extension, missing
  file, ambiguous config key) in the final summary instead of burying it.