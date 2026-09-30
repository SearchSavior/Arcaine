#!/usr/bin/env python3
"""Scaffold-classification fixtures through the real compiler.

`detect_scaffold_prefix` is exercised with controlled `chat_template.jinja`
variants that emit a complete, absent, partial, short-partial, or duplicated
thought scaffold in the assistant generation suffix. Each variant is a real
model directory (tokenizer files symlinked to the checkpoint), so the compiler
path is identical to HTTP.

Usage (host; needs the container + the built exporter):

  python3 scripts/check_scaffold_cases.py \
      --model /home/dwmcqueen/models/diffusiongemma-26B-A4B-it-AWQ-INT4 \
      --container-model /workspace/models/diffusiongemma-26B-A4B-it-AWQ-INT4
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}"
    "{{ '<|turn>' + m['role'] + '\\n' }}{{ m['content'] }}{{ '<turn|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|turn>model\\n' }}" + "{{ SCAFFOLD }}" +
    "{% endif %}"
)

CASES = {
    "complete":     "<|channel>thought\n<channel|>",
    "absent":       "",
    "partial_open": "<|channel>thought\n",
    "short_partial": "<|channel>",
    "duplicated":   ("<|channel>thought\n<channel|>"
                     "<|channel>thought\n<channel|>"),
}
ERROR_CASES = {"short_partial": "unsupported partial",
               "duplicated": "duplicated"}

REQUEST = {
    "model": "x", "state": "Pick one.",
    "questions": {"q": {"type": "choice", "instructions": "Which?",
                        "criteria": {"a": "first", "b": "second"}}},
}


def expected_scaffold(model):
    from tokenizers import Tokenizer
    ref = Tokenizer.from_file(model.rstrip("/") + "/tokenizer.json")
    full = ref.encode("<|channel>thought\n<channel|>", add_special_tokens=False).ids
    open_t = ref.encode("<|channel>thought\n", add_special_tokens=False).ids
    return ref, full, open_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--container-model", required=True)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--container-root", default="/workspace")
    ap.add_argument("--export-cmd", default="/tmp/decision_compile_export")
    args = ap.parse_args()

    try:
        ref, full, open_t = expected_scaffold(args.model)
    except ImportError:
        print("[INCOMPLETE] Python 'tokenizers' not installed; expected scaffold "
              "tokens cannot be built. Not a pass.")
        sys.exit(2)
    open_id, close_id = ref.token_to_id("<|channel>"), ref.token_to_id("<channel|>")

    tmp = os.path.join(args.repo, ".scaffold_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    failures = 0
    try:
        for case, scaffold in CASES.items():
            d = os.path.join(tmp, case)
            os.makedirs(d)
            for f in ("tokenizer.json", "tokenizer_config.json", "config.json"):
                src = args.model.rstrip("/") + "/" + f
                if os.path.exists(src):
                    # Symlink target is resolved inside the container.
                    os.symlink(args.container_model.rstrip("/") + "/" + f,
                               os.path.join(d, f))
            with open(os.path.join(d, "chat_template.jinja"), "w") as fh:
                fh.write(TEMPLATE.replace("{{ SCAFFOLD }}", scaffold))
            fx = os.path.join(tmp, case + ".json")
            json.dump([REQUEST], open(fx, "w"))

            cont_dir = args.container_root + "/.scaffold_tmp/" + case
            cont_fx = args.container_root + "/.scaffold_tmp/" + case + ".json"
            cmd = (f"docker exec -i arcaine-dev-1 {args.export_cmd} {cont_dir} "
                   f"{cont_fx} --seed 42 --max-seq 8192 --canvas-step 16 --canvas-max 256")
            proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
            if proc.returncode != 0:
                print(f"[INCOMPLETE] {case}: exporter failed: {proc.stderr.decode()[:200]}")
                sys.exit(2)
            res = json.loads(proc.stdout)["results"][0]
            if case in ERROR_CASES:
                ok = "error" in res and ERROR_CASES[case] in res["error"]
                print(f"[{'PASS' if ok else 'FAIL'}] {case}: rejected "
                      f"({res.get('error','')[:80]})")
                failures += 0 if ok else 1
                continue
            if "error" in res:
                print(f"[FAIL] {case}: unexpected error {res['error'][:100]}")
                failures += 1
                continue
            # Verify the canvas prefix matches the expected scaffold treatment.
            q = res["questions"][0]
            if case == "complete":
                ok = q["canvas"][0] not in (open_id, close_id)
            elif case == "absent":
                ok = q["canvas"][:len(full)] == full
            elif case == "partial_open":
                expected = full[len(open_t):]
                ok = q["canvas"][:len(expected)] == expected
            print(f"[{'PASS' if ok else 'FAIL'}] {case}: canvas prefix "
                  f"{q['canvas'][:5]}")
            failures += 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"{failures} scaffold-case FAILURES")
        sys.exit(1)
    print(f"scaffold cases: {len(CASES)} classifications matched "
          "(complete/absent/partial completed; short-partial/duplicated rejected)")


if __name__ == "__main__":
    main()
