#!/usr/bin/env python3
"""Compiler/tokenizer parity: Arcaine's real compiler vs the checkpoint.

Arcaine artifacts come from `decision_compile_export`, which calls the same
`parse_decision_request` + `compile_decision_questions` path the HTTP route uses.
Expected artifacts are built here independently with Hugging Face
`transformers`/`tokenizers` from the checkpoint `tokenizer.json` and
`chat_template.jinja`.

Build the exporter in the container (once):

  docker exec arcaine-dev-1 sh -c 'cd /workspace && icpx -fsycl -O2 \\
      -fsycl-targets=intel_gpu_bmg_g31 \\
      src/modeling/diffusion_gemma/benchmarks/decision_compile_export.cpp \\
      src/utils/decision_schema.cpp src/utils/chat.cpp \\
      src/preprocessing/tokenizer.cpp src/preprocessing/unicode.cpp \\
      src/preprocessing/unicode-data.cpp src/preprocessing/chat_template.cpp \\
      -I src -I third_party -I third_party/minja/include -I/opt/onednn/include \\
      -Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate \\
      -o /tmp/decision_compile_export'

Run on the host (needs the Python `transformers` and `tokenizers` packages):

  python3 scripts/check_compiler_parity.py \\
      --model /home/dwmcqueen/models/diffusiongemma-26B-A4B-it-AWQ-INT4 \\
      --native-cmd "docker exec -i arcaine-dev-1 /tmp/decision_compile_export \\
        /workspace/models/diffusiongemma-26B-A4B-it-AWQ-INT4 \\
        /workspace/scripts/fixtures/systemone_compile_cases.json \\
        --seed 42 --max-seq 8192 --canvas-step 16 --canvas-max 256"

A missing Python dependency or a skipped comparison is INCOMPLETE (exit 2), not a
pass. A slot-affecting mismatch is a release-blocking FAIL (exit 1).
"""
import argparse
import json
import os
import subprocess
import sys

EXPECTED_ERROR_INDEXES = {13, 14, 15}


def run_native(cmd):
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("native exporter failed:", proc.stderr.decode()[:800])
        sys.exit(2)
    return json.loads(proc.stdout)


def system_text(q, content_text=None):
    s = ("Answer a question about the state the user provides. The question lists "
         "its allowed answers; reply with exactly one label.\n")
    if q["instructions"]:
        s += "\nQuestion: " + q["instructions"] + "\n"
    for j, label in enumerate(q["label_texts"]):
        desc = q["option_descriptions"][j]
        s += "  " + label + ":"
        if q["type"] == "noul":
            if desc:
                s += " " + desc
        elif q["type"] == "choice":
            s += " " + q["option_keys"][j]
            if desc:
                s += " (" + desc + ")"
        else:
            if desc:
                s += " " + desc
        s += "\n"
    s += "\nReply with exactly one label, formatted as \"answer: label\"."
    return s


def tokenize_prompt(hf, msgs, tmpl):
    extra = {"enable_thinking": False}
    if tmpl is not None:
        extra["chat_template"] = tmpl
    out = hf.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **extra)
    if isinstance(out, list):
        return list(out)
    try:
        return list(out["input_ids"])
    except Exception:
        return list(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--native-cmd", required=True)
    ap.add_argument("--fixtures",
                    default=os.path.join(os.path.dirname(__file__),
                                         "fixtures", "systemone_compile_cases.json"))
    args = ap.parse_args()

    try:
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer
    except ImportError:
        print("[INCOMPLETE] Python 'transformers'/'tokenizers' not installed; "
              "independent expected artifacts cannot be built. Not a pass.")
        sys.exit(2)

    model = args.model.rstrip("/")
    ref = Tokenizer.from_file(model + "/tokenizer.json")
    hf = AutoTokenizer.from_pretrained(model)
    jinja = model + "/chat_template.jinja"
    tmpl = open(jinja).read() if os.path.exists(jinja) else None

    fixtures = json.load(open(args.fixtures))
    native = run_native(args.native_cmd)
    settings = native["settings"]
    pad = settings["pad_token_id"]
    turn_close = settings["turn_close_id"]
    close = "<channel|>"
    close_ids = ref.encode(close, add_special_tokens=False).ids
    scaffold_ids = ref.encode("<|channel>thought\n<channel|>", add_special_tokens=False).ids

    def cont(text):
        ids = ref.encode(close + text, add_special_tokens=False).ids
        if ids[:len(close_ids)] != close_ids:
            raise RuntimeError("close token is not a clean prefix of the continuation")
        return ids[len(close_ids):]

    failures, checked = 0, 0
    for r in native["results"]:
        idx = r["index"]
        if "error" in r:
            if idx in EXPECTED_ERROR_INDEXES:
                print(f"[PASS] fixture {idx}: compiler rejected as expected")
            else:
                failures += 1
                print(f"[FAIL] fixture {idx}: unexpected compiler error: {r['error'][:120]}")
            continue
        if idx in EXPECTED_ERROR_INDEXES:
            failures += 1
            print(f"[FAIL] fixture {idx}: expected a compiler error, got artifacts")
            continue
        what_base = f"fixture {idx}"
        state_text = r["state_text"]
        for q in r["questions"]:
            what = f"{what_base} q={q['external_id']}"
            msgs = [{"role": "system", "content": system_text(q)},
                    {"role": "user", "content": state_text}]
            try:
                exp_prompt = tokenize_prompt(hf, msgs, tmpl)
            except Exception as e:  # noqa: BLE001
                print(f"[INCOMPLETE] {what}: HF template render failed: {e}")
                sys.exit(2)
            if exp_prompt != q["prompt_ids"]:
                failures += 1
                print(f"[FAIL] {what}: prompt ids differ "
                      f"(native {len(q['prompt_ids'])}, hf {len(exp_prompt)})")
                for i, (a, b) in enumerate(zip(q["prompt_ids"], exp_prompt)):
                    if a != b:
                        print(f"   first diff at {i}: native {a} hf {b}")
                        break
            else:
                checked += 1

            encs = [cont("answer: " + lt) for lt in q["label_texts"]]
            if len({len(e) for e in encs}) != 1:
                failures += 1
                print(f"[FAIL] {what}: labels are not equal-length continuations")
                continue
            L = len(encs[0])
            pos = [i for i in range(L) if any(e[i] != encs[0][i] for e in encs)]
            if len(pos) != 1:
                failures += 1
                print(f"[FAIL] {what}: labels differ at {len(pos)} token positions")
                continue
            pos = pos[0]
            exp_label_ids = [e[pos] for e in encs]
            if exp_label_ids != q["label_ids"]:
                failures += 1
                print(f"[FAIL] {what}: label ids {q['label_ids']} vs hf {exp_label_ids}")

            slot = q["slot_position"]
            prefix_len = slot - pos
            canvas = q["canvas"]
            if prefix_len < 0 or canvas[prefix_len:prefix_len + L] != encs[0]:
                failures += 1
                print(f"[FAIL] {what}: canvas label encoding mismatch "
                      f"(slot {slot}, label pos {pos})")
                continue
            if canvas[:prefix_len] not in ([], scaffold_ids):
                failures += 1
                print(f"[FAIL] {what}: canvas prefix is neither empty nor the scaffold")
            after = prefix_len + L
            if after >= len(canvas) or canvas[after] != turn_close:
                failures += 1
                print(f"[FAIL] {what}: expected turn-close {turn_close} after the label")
            elif any(t != pad for t in canvas[after + 1:]):
                failures += 1
                print(f"[FAIL] {what}: non-pad tokens after the turn-close")
            else:
                checked += 1

    print()
    if failures:
        print(f"{failures} compiler parity FAILURES")
        sys.exit(1)
    print(f"compiler parity: {checked} artifact comparisons matched "
          f"({len(fixtures)} fixtures, exact token/position agreement)")
    print("NOTE: this establishes native-vs-checkpoint tokenizer/template/artifact "
          "parity. Behavior quality and cross-engine parity are separate gates.")


if __name__ == "__main__":
    main()
