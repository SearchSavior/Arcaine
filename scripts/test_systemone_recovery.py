#!/usr/bin/env python3
"""Focused error-recovery + graph-bypass check for /v1/systemone.

Start arcaine_server with one fault-injection point, for example:

  ARCAINE_SYSTEMONE_FAULT=in_decode_layer ./build/arcaine_server ...
  ARCAINE_SYSTEMONE_FAULT=after_prefill ./build/arcaine_server ...
  ARCAINE_SYSTEMONE_FAULT=after_first_decode_submit ./build/arcaine_server ...

The hook throws once per process while GPU work is submitted and buffers are
live.  This script verifies that:

  1. the first structured request fails cleanly (HTTP 500) with the injected
     fault, exercising the error-path queue drain + KV reset;
  2. a following structured request succeeds (state did not stick);
  3. its diagnostics report zero graph captures/replays, proving the explicit
     eager policy bypasses capture (meaningful with DIFF_NVFP4_SYCL_GRAPH=1);
  4. an ordinary chat request still works afterwards.

Pass --graph-mode when the server runs with DIFF_NVFP4_SYCL_GRAPH=1.  The chat
check is then skipped: chat under that experimental graph mode fails in the
current tree independently of this feature (verified on the base commit).

Usage: python3 scripts/test_systemone_recovery.py --base http://127.0.0.1:7461 --key local
"""
import argparse
import json
import urllib.error
import urllib.request

FAILURES = []


def request(base, path, body, key=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data,
                                 headers={"content-type": "application/json"})
    if key:
        req.add_header("authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read())
        except Exception:
            payload = None
        return e.code, payload


def check(name, cond, detail=""):
    print(("[PASS] " if cond else "[FAIL] ") + name + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:7461")
    ap.add_argument("--key", default="local")
    ap.add_argument("--graph-mode", action="store_true",
                    help="server runs with DIFF_NVFP4_SYCL_GRAPH=1; skip the chat check")
    args = ap.parse_args()

    st, models = request(args.base, "/v1/models", None, args.key)
    model = models["data"][0]["id"]
    body = {
        "model": model,
        "state": {"ticket": "Everything is down and we have a demo at noon."},
        "questions": {
            "urgent": {"type": "noul",
                       "instructions": "Does this require a response within an hour?",
                       "criteria": {"true": "An outage needs prompt action", "false": "It can wait"}},
        },
        "arcaine": {"diagnostics": True},
    }

    st1, b1 = request(args.base, "/v1/systemone", body, args.key)
    check("fault injection: first structured request fails cleanly (500)", st1 == 500,
          f"got {st1}: {str(b1)[:200]}")

    st2, b2 = request(args.base, "/v1/systemone", body, args.key)
    check("recovery: next structured request succeeds (200)", st2 == 200,
          f"got {st2}: {str(b2)[:300]}")
    if st2 == 200:
        ext = b2.get("arcaine", {})
        check("graph bypass: 0 captures / 0 replays (DIFF_NVFP4_SYCL_GRAPH=1)",
              ext.get("graph_captures_delta") == 0 and ext.get("graph_replays_delta") == 0,
              json.dumps(ext))

    if args.graph_mode:
        print("[SKIP] chat under DIFF_NVFP4_SYCL_GRAPH=1 fails independently of "
              "this feature (verified on the base commit)")
    else:
        st3, b3 = request(args.base, "/v1/chat/completions",
                          {"model": model,
                           "messages": [{"role": "user", "content": "Say hi in one word."}],
                           "max_tokens": 16}, args.key)
        check("recovery: ordinary chat still works (200)",
              st3 == 200 and b3.get("choices"), f"got {st3}: {str(b3)[:200]}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:", *FAILURES, sep="\n  - ")
        raise SystemExit(1)
    print("recovery + graph-bypass checks passed")


if __name__ == "__main__":
    main()
