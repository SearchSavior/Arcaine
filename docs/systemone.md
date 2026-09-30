# POST /v1/systemone — structured decisions on DiffusionGemma

Jev-format structured decision reads on Arcaine's DiffusionGemma backend.
Each question is answered from **one denoiser pass** over a compiled answer
canvas whose answer slot is filled with fresh uniform noise per read; the slot
row is scored at temperature 1 against the full vocabulary and the allowed
label set. Repeated reads average the per-read conditional label
distributions.

This is a **format-compatible, locally computed approximation** of Jev's
`/v1/systemone`. It does not load TypeSafe's weights and does not claim equal
predictions, calibrated probabilities, or equal performance.

## Validation status

[systemone-results.md](systemone-results.md) is the requirement-to-evidence
table with exact commands, environment, and results.

**Implemented and executed on this checkout**

- `scripts/test_systemone.py` — HTTP shapes, validation, isolation,
  concurrency, lifecycle counts, graph-bypass counters.
- `scripts/check_compiler_parity.py` + `decision_compile_export.cpp` — exact
  prompt/canvas/slot/label-id agreement between Arcaine's compiler and an
  independent checkpoint-tokenizer/template construction, over Choice
  2/26/27/128/255, Score 2/10, Noul, Unicode, and structured descriptions.
- `scripts/check_scaffold_cases.py` — complete/absent/partial completed and
  short-partial/duplicated rejected, through the real compiler.
- `decision_hash_check.cpp` — content-stream golden vectors and invariance.
- `decision_validation_check.cpp` — direct-API bounds before GPU submission.
- `slot_score_check.cpp` — conditional probabilities, 255 labels, FP32
  underflow, nonfinite logits.
- `scripts/test_systemone_recovery.py` — fault injection inside prefill and
  decode, error-path drain, graph bypass.
- `scripts/eval_systemone_behavior.py` — labeled Noul/Choice/Score behavior.

**Implemented, not executed here (dependency or hardware gate)**

- `scripts/check_sdk_compat.py` — official TypeSafe/Jev SDK request/response.
  The `typesafe_sdk` package is not installed in this environment.
- Width execution at 17/32/64 and the full fixed/auto read boundary matrix.
  These need a direct-model harness; the compiled canvas width is fixed by the
  compiler, not by an HTTP field.
- `scripts/check_tokenizer_parity.py` — raw-encoding companion; superseded by
  the compiler-parity check above.
- Cross-engine comparison against the pinned vLLM PR. No equivalent
  hardware/checkpoint is available here.

**Not established**

- Exact Jev confidence parity (formula unpublished), calibrated probabilities,
  and equal predictions/throughput.
- Cross-engine parity remains unverified.

Guide-vs-schema difference: the Choice guide permits structured option
descriptions and the narrower HTTP schema documents strings only — the
structured form is accepted here, serialized deterministically, and the
official SDK was verified to parse both structured Choice descriptions and
structured Score `legend` levels. The standard response keeps only Jev answer
fields; local diagnostics live under the namespaced `arcaine` object, which
the SDK ignores.

## Request

```json
{
  "model": "arcaine-diffusiongemma",
  "state": {"ticket": "Everything is down and we have a demo at noon."},
  "questions": {
    "urgent":   {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}},
    "route":    {"type": "choice", "instructions": "...", "criteria": {"support": "...", "engineering": "..."}},
    "severity": {"type": "score",  "instructions": "...", "criteria": ["Minor", "Serious", "Critical"]}
  }
}
```

- `state`: string, object, or array (structured values are serialized
  deterministically, meaning preserved).
- `questions`: a map keyed by caller-provided strings; duplicate JSON keys are
  rejected (422). Question IDs are response keys only — they never enter
  inference or random seeds.
- Choice `criteria`: 2–255 named options; descriptions may be strings, null,
  or structured content. Option keys are sorted canonically for the label
  mapping.
- Score `criteria`: 2–10 ordered level descriptions (string or structured).
- Noul `criteria`: optional `true`/`false` descriptions.
- `instructions`: string or structured content.
- Unknown request fields and control fields (`seed`, `steps`, `canvas`,
  `samples`, `think`, ...) are rejected with 422 — canvas, seed, sampling, and
  denoising are internal settings. The optional namespaced extension
  `"arcaine": {"diagnostics": true}` adds a per-question diagnostics object
  (label mass, vocabulary entropy, per-read stats, timing) to the response.

## Response

```json
{
  "model": "arcaine-diffusiongemma",
  "answers": {
    "urgent":   {"type": "noul", "noul": 0.91},
    "route":    {"type": "choice", "choice": "engineering",
                 "probabilities": {"engineering": 0.8, "support": 0.2},
                 "confidence": 0.2781},
    "severity": {"type": "score", "score": 1.2,
                 "legend": {"0": "Minor", "1": "Serious", "2": "Critical"},
                 "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3},
                 "confidence": 0.1827}
  },
  "usage": {"input_tokens": 312, "output_tokens": 3}
}
```

- Noul: `noul` is the probability of **yes**; no separate confidence field.
- Choice: `choice` is the highest-probability original option key (ties
  resolve to canonical key order); `probabilities` covers all options.
- Score: `score` is the zero-based index expectation, `legend` maps index
  strings to the level descriptions as given, `probabilities` is indexed.
- `usage` is **local work accounting**, not Jev billing parity:
  `input_tokens` counts the actual encoded prompt tokens summed over the
  isolated questions (a question's prompt is counted once, not per read);
  `output_tokens` counts scored answer-slot positions (one per read).

## Compatibility table

| Behavior | Jev documented | This backend |
|---|---|---|
| Request/response JSON shape | as documented | matched; official SDK parses string and structured Choice descriptions and structured Score legend levels |
| Noul answer | probability of yes | matched shape |
| Choice answer | key + probabilities + confidence | matched shape; confidence differs (below) |
| Score answer | zero-based expectation + legend + probabilities + confidence | matched shape; confidence differs (below) |
| Question isolation | questions cannot see each other | matched: separate prompt + canvas per question, sequential execution under the model lock |
| Parallel execution / near-constant latency | documented | **not reproduced**: questions run sequentially |
| Confidence | distribution concentration, formula not published | **approximation**: `1 - H(p)/log(K)` on the final averaged distribution; marked by response headers |
| Score level evaluation | levels described as evaluated separately | **disclosed difference**: the label scorer sees the full rubric |
| Token accounting | hosted billing | local work counts (above) |
| Modalities | text + images | **text only**; `images`/multipart are rejected |

Every response (including errors) carries:

```
X-Arcaine-Confidence-Method: normalized-entropy-v1
X-Arcaine-Compatibility: jev-format-approximate-confidence
```

## Status codes

- `401` invalid/missing bearer token (when `ARCAINE_API_KEY` is set).
- `404` unknown model name.
- `422` request validation failure (schema, labels, limits).
- `500` internal error (including nonfinite model scores).

No rate limiting is implemented, so 429/529 are never emitted; concurrent
requests serialize on the generation mutex.

## Model name

`model` must equal the server's `--served-model-name`. Setting
`ARCAINE_SYSTEMONE_MODEL_ALIAS=jev-latest` additionally accepts that alias;
the response's `model` always reports the actual local model.

## Local limits (Arcaine, not Jev)

- Questions per request: 64 (`ARCAINE_SYSTEMONE_MAX_QUESTIONS`).
- Choice options: 2–255 (single-token label codebook, verified per request
  against the loaded tokenizer).
- Score levels: 2–10.
- Context per question: the compiled prompt **plus** canvas must fit `--max-seq`
  (default 2048). A question is rejected when the two together exceed the limit.
- Canvas width: the compiled answer template rounded up to a multiple of 16, at
  most the model's configured canvas capacity (`canvas_capacity()`, the width
  the activation arena is planned for). The capacity is fixed at load time; the
  former unconditional 256 value is not guaranteed.
- Allocation mode: structured reads require the pooled activation arena.
  `DIFF_ARENA=off` / `DISABLE_SCRATCH=1` are rejected with a clear error before
  GPU work, because that mode frees device memory immediately while submitted
  kernels can still read it.

## Read policy (deployment settings, not Jev-derived)

- Default: **1 read per question**.
- `ARCAINE_SYSTEMONE_READS=N`: fixed reads, 1–32.
- `ARCAINE_SYSTEMONE_READS_AUTO=1`: start with one read, escalate per question
  up to `ARCAINE_SYSTEMONE_READS_AUTO_MAX` (default 4) while the aggregate's
  normalized label entropy exceeds
  `ARCAINE_SYSTEMONE_AUTO_ENTROPY_THRESHOLD` (default 0.5), the mean label
  mass is below `ARCAINE_SYSTEMONE_AUTO_MIN_LABEL_MASS` (default 0.0), or the
  global argmax is not an allowed label. The first read is always included.
- `ARCAINE_GPU_WATCHDOG_S=N` (default 0 = off, range 0–3600): if a structured
  read makes no GPU progress for N seconds, the server writes the stalled stage
  to the log. Then the server stops with code 70. In this way, the model lock is
  released. A hung GPU can need a host reboot. The watchdog prevents only a
  silent, permanent hang.

## Implementation notes

- Sources: `src/utils/decision_schema.{hpp,cpp}` (validation, compilation,
  aggregation), `src/modeling/diffusion_gemma/structured_read.cpp` (read
  driver), `src/modeling/diffusion_gemma/fusions/logits.hpp`
  (`slot_score_rows`), `src/modeling/diffusion_gemma/model.{hpp,cpp}`
  (`read_decisions`, structured `decode_forward` mode),
  `src/apps/server/routes/systemone.cpp` (route).
- The decoder processes all canvas positions bidirectionally; only the final
  scoring reads the slot row. There is no autoregressive position shift and
  no EOS truncation of scored positions.
- Structured reads hold an explicit per-call eager policy that disables **all**
  NVFP4 SYCL graph capture/replay (whole-layer sessions and nested per-kernel
  micro-captures) on both devices until keying, address lifetime, and
  invalidation are proven. `arcaine.graph_captures_delta` /
  `graph_replays_delta` in the diagnostics extension confirm 0 for a read even
  with `DIFF_NVFP4_SYCL_GRAPH=1`. Ordinary chat keeps the existing capture path.
- The canvas prefix (empty thought scaffold) is derived from the **actual**
  rendered chat-template suffix, not an assumption; a partial/duplicated
  scaffold is rejected, and answer labels are verified as ordinary (non-special)
  single tokens including the closing token.
- Noise draws: counter-based splitmix64 hash of (question content, server
  seed, read index, counter) with rejection sampling (no modulo bias); the
  stream is independent of question IDs, request order, and other questions.
- Scoring: final softcap applied once in FP32; logprobs normalized over the
  full vocabulary; label mass, vocabulary entropy, and global argmax are
  computed on device and only compact results are downloaded. The LM-head
  projection is BF16 (the model's retained projection precision); reductions
  are FP32. Raw nonfinite logits are flagged on device before softcap.
