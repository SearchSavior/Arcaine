# SystemOne validation results and provenance

Requirement-to-evidence record for `/v1/systemone`. It separates implemented,
executed, and incomplete checks, and gives reproduction commands. The endpoint
reference is [systemone.md](systemone.md).

## Environment

| Item | Value |
|---|---|
| Executable | `arcaine_server` |
| Checkpoint | `diffusiongemma-26B-A4B-it-AWQ-INT4` |
| GPU | Intel Arc Pro B70 (Battlemage G31), one device |
| Driver | Level-Zero V2 `20.2.0 [1.15.38646+7]` |
| Host kernel | `7.2.6-1-cachyos` |
| Toolchain | oneAPI `icpx`, `-fsycl-targets=intel_gpu_bmg_g31`, oneDNN |
| Host Python | 3.14.7, `transformers`, `tokenizers` |
| Official SDK | `typesafe_sdk` 0.7.0 (`httpx2` 2.13.0) in an isolated venv |
| Build | `cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DARCAINE_SYCL_TARGETS=intel_gpu_bmg_g31 && cmake --build build --target arcaine_server -j4` |
| Server | `ARCAINE_API_KEY=local ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/arcaine_server --model <ckpt> --served-model-name arcaine-diffusiongemma --host 0.0.0.0 --port 7461 --max-seq 8192` |

## Requirement-to-evidence table

| Spec area | Implementation | Fixture / harness | Executed evidence | Status |
|---|---|---|---|---|
| §2 public endpoint, auth, unsupported backend | `systemone.cpp` | `test_systemone.py` | 401/404/422/200 cases pass | Executed |
| §2 response shape, confidence headers, usage | `decision_schema.cpp` | `test_systemone.py`, `check_sdk_compat.py` | shape/header/usage checks pass; SDK parses | Executed |
| §2 SDK request construction | `check_sdk_compat.py` | official `TypeSafeClient` | SDK 0.7.0 request + response pass, structured legend values preserved | Executed |
| §3 compile, label codebook 2–255 | `decision_schema.cpp` | `systemone_compile_cases.json`, `decision_compile_export.cpp`, `check_compiler_parity.py` | 30 exact prompt/canvas/slot/label comparisons; Choice 2/26/27/128/255, Score 2/10, Noul, Unicode, structured | Executed |
| §3.4 thought-scaffold suffix | `detect_scaffold_prefix` | `check_scaffold_cases.py` | complete/absent/partial completed; short-partial/duplicated rejected | Executed |
| §3 content stream hash | `decision_schema.cpp` | `decision_hash_check.cpp` | collision, control-char, boundary, rename/reorder, distractor, golden vectors | Executed |
| §3 tokenizer/template parity | `chat_template.cpp`, `decision_schema.cpp` | `check_compiler_parity.py` | minja/HF whitespace bug fixed; prompts now match exactly | Executed |
| §4 independent API validation | `decision_validation.hpp` | `decision_validation_check.cpp` | bounds/capacity/overflow-before-narrowing pass | Executed |
| §4 numerical scoring | `fusions/logits.hpp` | `slot_score_check.cpp` | conditional/255/underflow/ties/tail/nonfinite pass | Executed |
| §4 short canvases / graph bypass | `structured_read.cpp`, `model.cpp` | `test_systemone.py`, `test_systemone_recovery.py` | eager scope; 0 captures/0 replays | Partly executed (widths 17/32/64 not executed) |
| §4 lifecycle atomicity | driver + `decode_forward` | `test_systemone.py` | 1 prefill + N decode per question | Executed |
| §5 fixed/adaptive reads | `structured_read.cpp` | `eval_systemone_behavior.py` | fixed 1 and 4 reads executed | Partly executed (reads 32, auto boundaries not executed) |
| §7.4 error recovery | `arena.hpp`, `structured_read.cpp`, `model.cpp` | `test_systemone_recovery.py` | fault at prefill, in-decode, after-decode; next read + chat pass | Executed |
| §7.5 behavior | `decision_schema.cpp` | `eval_systemone_behavior.py` | accuracy/Brier/coverage/repeatability, 1 and 4 reads | Executed (small labeled set) |
| §7.7 performance | benchmark harness | `benchmark_systemone.py` | prompt sweep, reads 1 and 4, measured tokens | Executed |

## Measured results

### Compiler/tokenizer parity (`check_compiler_parity.py`)

`compiler parity: 30 artifact comparisons matched (16 fixtures, exact
token/position agreement)`; fixtures 13/14/15 rejected as expected
(1-option Choice, 1-level Score, 256-option Choice). Each question compares the
full prompt IDs, canvas IDs, slot position, and label IDs against an independent
Hugging-Face construction from the checkpoint `tokenizer.json` and
`chat_template.jinja`.

A blocking defect was found and fixed during this work: `build_chat_prompt` for
`ChatTemplateMessage` wrapped plain text as single-element content-parts, so the
template's sequence branch appended a trailing space after the system content.
The rendered prompt differed from the checkpoint by one space token before
resolution. Passing string content restores exact parity.

### Scaffold cases (`check_scaffold_cases.py`)

| Case | Result |
|---|---|
| complete | accepted; canvas has no scaffold prefix |
| absent | accepted; canvas prepends the full scaffold |
| partial (`<|channel>thought\n`) | completed; canvas appends the close token |
| short partial (`<|channel>`) | rejected: "unsupported partial" |
| duplicated | rejected: "duplicated" |

### Content-stream hash (`decision_hash_check.cpp`)

All pass: historical concatenation-collision example now differs;
control-character and field-boundary cases differ; rename/reorder keeps
canonical-content seeds and execution order; an unrelated question does not
change another's seed. Golden vectors: `q1=17734819047203551209`,
`q2=9788230640868820864` for the documented FNV-1a-64 over little-endian
length-prefixed fields (stream version `arcaine-systemone-v1`).

### Numerical scoring (`slot_score_check.cpp`)

`all numerical checks passed` (conditional probabilities, 255 labels, FP32
underflow, ties, tail labels, nonfinite raw logits).

### Direct API validation (`decision_validation_check.cpp`)

`all validation checks passed`, including canvas/label sizes checked before any
narrowing, and rejection of `DIFF_ARENA=off`/`DISABLE_SCRATCH` for structured
reads.

### Error recovery (`test_systemone_recovery.py`)

| Fault point | Result |
|---|---|
| `after_prefill` | first request 500, next read 200, chat 200 |
| `in_decode_layer` | first request 500, next read 200, chat 200 |
| `after_first_decode_submit` | first request 500, next read 200, chat 200 |
| `in_decode_layer` + `DIFF_NVFP4_SYCL_GRAPH=1` | first 500, next read 200 with 0 captures / 0 replays |

### HTTP acceptance (`test_systemone.py`)

All checks passed: shapes, headers, validation, isolation (solo vs group,
rename/reorder/distractor), Choice 26/27/128/255, Score 10, Unicode, diagnostics,
lifecycle, graph bypass, empty IDs, concurrent and mixed chat/structured callers,
ordinary chat.

### Behavior (`eval_systemone_behavior.py`, 10 labelled cases)

| Reads | Accuracy | Brier | Coverage (conf≥0.5, mass≥0.5) | Repeatable |
|---|---|---|---|---|
| 1 | 0.889 (8/9 scored) | 0.207 | 0.900 | yes |
| 4 | 0.889 (8/9 scored) | 0.212 | 0.900 | yes |

The one miss is `score_moderate` (gold 1, predicted 0.04). Noul cases include
negation and an unknown case (no gold, counted only in coverage). Ordinary chat
generation returned empty text for every probe, so an ordinary-vs-structured
comparison could not be measured; this is reported, not hidden. The set is small;
it detects gross behavior problems, not calibration.

### Performance (`benchmark_systemone.py`, iters=5, warmup=1)

Reads=1:

| Target | Measured tokens | canvas | reads | p50 ms | p95 ms | prefill ms | decode ms | reads/s | dec/s |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 518–526 | 16 | 15 | 2153.5 | 2163.6 | 462.5 | 93.0 | 1.4 | 1.4 |
| 1024 | 1011–1019 | 16 | 15 | 3289.7 | 3311.9 | 675.2 | 93.8 | 0.9 | 0.9 |
| 2048 | 2031–2039 | 16 | 15 | 7676.2 | 7769.7 | 1232.5 | 93.4 | 0.4 | 0.4 |
| 4096 | 4071–4079 | 16 | 15 | 24798.3 | 24881.0 | 2355.8 | 94.1 | 0.1 | 0.1 |

Reads=4:

| Target | Measured tokens | canvas | reads | p50 ms | p95 ms | prefill ms | decode ms | reads/s | dec/s |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 518–526 | 16 | 60 | 2462.2 | 2493.3 | 468.9 | 363.4 | 4.9 | 1.2 |
| 1024 | 1011–1019 | 16 | 60 | 3684.3 | 3716.3 | 672.7 | 369.7 | 3.3 | 0.8 |
| 2048 | 2031–2039 | 16 | 60 | 8588.8 | 8603.7 | 1231.7 | 365.8 | 1.4 | 0.3 |
| 4096 | 4071–4079 | 16 | 60 | 27448.5 | 27652.4 | 2358.8 | 367.3 | 0.4 | 0.1 |

`reads/s` counts scored slots; `dec/s` counts completed question answers. At
4096 tokens p50 is dominated by host prompt compilation/tokenization of three
long questions, not by the decoder. Four reads add only the decode time
(~270 ms at 512).

These numbers are measured after the single-tokenization compiler fix. Before
that fix the scaffold-suffix check tokenized every prompt twice, and the same
harness gave 4096 p50 57106 ms (reads=1), 512 p50 8781.8 ms. The native
tokenizer is the slow part; minja rendering is ~0.2 s per render.

### Official SDK (`check_sdk_compat.py`, typesafe_sdk 0.7.0)

All pass, including an official `TypeSafeClient.system_one(...)` request built
from `Noul`/`Choice`/`Score` objects, and preservation of a structured Score
legend value `{"level":"Serious","rank":1}`. Guide-vs-schema note: the Choice
guide allows structured descriptions and the SDK accepts them; Arcaine accepts
and preserves them.

## Exact reproduction commands

```bash
# Server (container arcaine-dev-1), default read policy
ARCAINE_API_KEY=local ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/arcaine_server \
  --model models/diffusiongemma-26B-A4B-it-AWQ-INT4 --served-model-name arcaine-diffusiongemma \
  --host 0.0.0.0 --port 7461 --max-seq 8192

# HTTP acceptance
python3 scripts/test_systemone.py --base http://127.0.0.1:7461 --key local
# Recovery (start the server with one ARCAINE_SYSTEMONE_FAULT=<point> per leg)
python3 scripts/test_systemone_recovery.py --base http://127.0.0.1:7461 --key local
# Behavior (server read policy 1, then ARCAINE_SYSTEMONE_READS=4)
python3 scripts/eval_systemone_behavior.py --base http://127.0.0.1:7461 --key local --ordinary
# Performance
python3 scripts/benchmark_systemone.py -p 512,1024,2048,4096 --iters 5 --warmup 1

# Native harnesses (build in the container)
docker exec arcaine-dev-1 sh -c 'cd /workspace && icpx -fsycl -O2 -fsycl-targets=intel_gpu_bmg_g31 \
  src/modeling/diffusion_gemma/benchmarks/<harness>.cpp <sources> -I src -I third_party \
  -I third_party/minja/include -I/opt/onednn/include \
  -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate -o /tmp/<harness>'

# Compiler/tokenizer parity (host + container exporter)
python3 scripts/check_compiler_parity.py --model /home/dwmcqueen/models/diffusiongemma-26B-A4B-it-AWQ-INT4 \
  --native-cmd "docker exec -i arcaine-dev-1 /tmp/decision_compile_export \
    /workspace/models/diffusiongemma-26B-A4B-it-AWQ-INT4 \
    /workspace/scripts/fixtures/systemone_compile_cases.json --seed 42 --max-seq 8192"
python3 scripts/check_scaffold_cases.py --model /home/dwmcqueen/models/diffusiongemma-26B-A4B-it-AWQ-INT4 \
  --container-model /workspace/models/diffusiongemma-26B-A4B-it-AWQ-INT4

# Official SDK (isolated venv)
/tmp/opencode/sdkvenv/bin/python3 scripts/check_sdk_compat.py --base http://127.0.0.1:7461 --key local
```

## Incomplete gates

- **Canvas-width execution 17/32/64.** The compiled canvas width is fixed by
  the answer template (16 here), not by a request field. Executing arbitrary
  widths needs a direct-model harness that constructs a `CompiledDecisionTemplate`
  with a chosen canvas length and calls `read_decisions`. Not built; HTTP
  cannot express this setting.
- **Read caps 1/4/32 and auto-policy boundaries.** Fixed 1 and 4 are executed.
  32 reads and the entropy/label-mass/out-of-label-argmax escalation boundaries
  require a direct-model harness or per-leg server restarts; only 1 and 4 were
  run here.
- **Slot-isolation capture.** Input-level isolation is proven at the compiled
  artifact and content-stream level (compiler parity, hash invariance). Directly
  capturing on-device canvas inputs per read and showing only the registered
  slot changes was not instrumented.
- **Mixed graph-mode chat.** Chat under the experimental
  `DIFF_NVFP4_SYCL_GRAPH=1` fails in this tree independently of this feature
  (reproduced on the base commit); structured reads bypass capture (0/0).
- **Cross-engine parity.** No pinned-vLLM comparison; equivalent
  hardware/checkpoint unavailable. Unverified.
- **Confidence parity.** `normalized-entropy-v1` approximation, not exact Jev.
