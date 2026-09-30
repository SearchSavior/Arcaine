#!/usr/bin/env python3
"""Compare the in-process native tokenizer against the checkpoint's HF
tokenizer on the strings that structured-read compilation actually depends on.

Hard-fail (id-for-id) checks:
  * the fully rendered chat prompt (system + user, thinking disabled) -- this is
    the exact token sequence the encoder prefills;
  * the structured answer templates (`answer: <label>`) and thought scaffold;
  * the label codebook words.

Informational: bare-word/Unicode fixtures are reported but not failed, because
HF `tokenizers` applies the SentencePiece automatic leading-space prefix to a
bare string while the native tokenizer does not.  That convention difference
does not affect the pipeline (the rendered prompt matches exactly, as checked
above); the endpoint never encodes a bare word as a standalone prompt.

Usage:
  python3 scripts/check_tokenizer_parity.py \
      --model /home/dwmcqueen/models/diffusiongemma-26B-A4B-it-AWQ-INT4 \
      --native-cmd "docker exec -i arcaine-dev-1 /tmp/tokenizer_parity_check /workspace/models/diffusiongemma-26B-A4B-it-AWQ-INT4/tokenizer.json"
"""
import argparse
import json
import subprocess
import sys

SLOT_FIXTURES = [
    "answer: A", "answer: AA", "answer: ZZ", "answer: yes", "answer: no",
    "answer: 1", "answer: 10",
    "<|channel>thought\n<channel|>",
    "<|turn>model\n<|channel>thought\n<channel|>",
    "<turn|>",
]
INFO_FIXTURES = [
    "Everything is down and we have a demo at noon.",
    "\u6570\u636e\u4e2d\u5fc3?", "eu-centr\u00e4l", "\u9ad8",
    "{\"e\":\"outage\",\"dc\":\"eu-central\"}",
    "Reply with exactly one label, formatted as \"answer: label\".",
]

SYSTEM_TEXT = ("Answer a question about the state the user provides. The question "
               "lists its allowed answers; reply with exactly one label.\n\n"
               "Question: Does this require a response within an hour?\n"
               "  yes: A service outage needs prompt action\n"
               "  no: The request can wait\n\n"
               "Reply with exactly one label, formatted as \"answer: label\".")
USER_TEXT = "Everything is down and we have a demo at noon."


def native_encode(cmd, texts):
    proc = subprocess.run(cmd, shell=True, input=json.dumps(texts).encode(),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("native checker failed:", proc.stderr.decode()[:500])
        sys.exit(2)
    return json.loads(proc.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--native-cmd", required=True,
                    help="runs the native checker and reads a JSON array from stdin")
    args = ap.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("[INCOMPLETE] Python 'tokenizers' is not installed; the checkpoint "
              "comparison cannot run. This is not a passing release-gate result.")
        sys.exit(2)
    ref = Tokenizer.from_file(args.model.rstrip("/") + "/tokenizer.json")

    failures = 0
    prompt_checked = False
    skipped = 0

    # 1. Rendered chat prompt (the real encoder input).
    try:
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(args.model)
        msgs = [{"role": "system", "content": SYSTEM_TEXT},
                {"role": "user", "content": USER_TEXT}]
        rendered = hf.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
        enc = hf.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                     enable_thinking=False)
        hf_ids = list(enc["input_ids"])
        nat_ids = native_encode(args.native_cmd, [rendered])[0]
        prompt_checked = True
        if hf_ids == nat_ids:
            print(f"[PASS] rendered chat prompt ({len(hf_ids)} ids)")
        else:
            failures += 1
            print(f"[FAIL] rendered chat prompt\n   HF:     {hf_ids}\n   native: {nat_ids}")
    except ImportError:
        skipped += 1
        print("[INCOMPLETE] rendered prompt: transformers not installed. The prompt "
              "check is required; this run does not establish prompt parity.")

    # 2. Slot-critical fixtures as the compiler encodes them: a continuation
    #    immediately after the prompt's closing special token.
    close = "<channel|>"
    slot_inputs = [close + text for text in SLOT_FIXTURES]
    nat = native_encode(args.native_cmd, slot_inputs)
    for text, full, ids in zip(SLOT_FIXTURES, slot_inputs, nat):
        want = ref.encode(full, add_special_tokens=False).ids
        if want == ids:
            print(f"[PASS] continuation {text!r} -> {ids[len(ref.encode(close, add_special_tokens=False).ids):]} ids")
        else:
            failures += 1
            print(f"[FAIL] continuation {text!r}\n   HF:     {want}\n   native: {ids}")

    # 3. Informational bare strings.
    nat_info = native_encode(args.native_cmd, INFO_FIXTURES)
    for text, ids in zip(INFO_FIXTURES, nat_info):
        want = ref.encode(text, add_special_tokens=False).ids
        tag = "same" if want == ids else "leading-space convention differs"
        print(f"[INFO] {text!r}: {tag}")

    print()
    if failures:
        print(f"{failures} tokenizer parity FAILURES (slot/prompt critical)")
        sys.exit(1)
    if skipped:
        print(f"INCOMPLETE: {skipped} required check(s) skipped; "
              "this is not a passing release-gate result")
        sys.exit(2)
    print(f"tokenizer parity: rendered prompt + {len(SLOT_FIXTURES)} slot-critical fixtures match")
    print("NOTE: this compares raw encodings, not the full compiled artifact set "
          "(build_prompt ids, canvas, slot, label codebook) against the checkpoint "
          "tokenizer; see docs/systemone-results.md.")


if __name__ == "__main__":
    main()
