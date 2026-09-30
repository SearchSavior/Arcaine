#!/usr/bin/env python3
"""Focused /v1/systemone validation against a running arcaine_server.

Covers the acceptance behaviors from the implementation spec that are
observable over HTTP: response shapes, validation errors (422), model-name
handling (404), auth (401), determinism, and question isolation.

Usage: python3 scripts/test_systemone.py [--base http://127.0.0.1:7461] [--key local]
"""
import argparse
import concurrent.futures
import json
import sys
import urllib.error
import urllib.request

FAILURES = []


def post(base, path, body, key=None, raw=False):
    data = body if raw else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"content-type": "application/json"})
    if key:
        req.add_header("authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, dict(r.headers), json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read())
        except Exception:
            payload = None
        return e.code, dict(e.headers), payload


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def approx_eq(a, b, tol=1e-6):
    return abs(a - b) <= tol


def answers_eq(a, b, tol=1e-4):
    # confidence = 1 - H/ln K amplifies probability noise by ~ln(1/(1-p))/ln K
    # near saturation, so it gets a wider (documented) tolerance.
    conf_tol = tol * 10
    if a is None or b is None or set(a) != set(b):
        return False
    for k in a:
        ja, jb = json.dumps(a[k], sort_keys=True), json.dumps(b[k], sort_keys=True)
        if ja == jb:
            continue
        # compare numerically field by field
        if a[k].get("type") != b[k].get("type"):
            return False
        for f in ("noul", "score"):
            if f in a[k] and not approx_eq(a[k][f], b[k][f], tol):
                return False
        if "confidence" in a[k] and not approx_eq(a[k]["confidence"], b[k]["confidence"], conf_tol):
            return False
        for f in ("choice",):
            if f in a[k] and a[k][f] != b[k][f]:
                return False
        pa, pb = a[k].get("probabilities"), b[k].get("probabilities")
        if (pa is None) != (pb is None):
            return False
        if pa is not None:
            if set(pa) != set(pb):
                return False
            if any(not approx_eq(pa[x], pb[x], tol) for x in pa):
                return False
    return True


GOLDEN = {
    "model": None,  # filled from /v1/models
    "state": {"ticket": "Everything is down and we have a demo at noon."},
    "questions": {
        "urgent": {"type": "noul",
                   "instructions": "Does this require a response within an hour?",
                   "criteria": {"true": "A service outage needs prompt action",
                                "false": "The request can wait"}},
        "route": {"type": "choice",
                  "instructions": "Choose the responsible team.",
                  "criteria": {"support": "Customer support",
                               "engineering": "Service outages and software faults"}},
        "severity": {"type": "score",
                     "instructions": "Rate the incident.",
                     "criteria": ["Minor", "Serious", "Critical"]},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:7461")
    ap.add_argument("--key", default="local")
    args = ap.parse_args()
    base, key = args.base, args.key

    st, _, models = post(base, "/v1/models", None, key) if False else (None, None, None)
    req = urllib.request.Request(base + "/v1/models")
    req.add_header("authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=30) as r:
        models = json.loads(r.read())
    model = models["data"][0]["id"]
    print("served model:", model)
    GOLDEN["model"] = model

    # --- golden request -----------------------------------------------------
    st, hdr, body = post(base, "/v1/systemone", GOLDEN, key)
    check("golden: HTTP 200", st == 200, f"got {st}: {body}")
    if st != 200:
        print("aborting: golden request failed")
        sys.exit(1)
    check("golden: confidence header",
          hdr.get("X-Arcaine-Confidence-Method") == "normalized-entropy-v1")
    check("golden: compatibility header",
          hdr.get("X-Arcaine-Compatibility") == "jev-format-approximate-confidence")
    a = body.get("answers", {})
    check("golden: answer keys", set(a) == {"urgent", "route", "severity"}, str(a))
    u = a.get("urgent", {})
    check("noul shape", u.get("type") == "noul" and isinstance(u.get("noul"), (int, float))
          and 0.0 <= u["noul"] <= 1.0 and "confidence" not in u, json.dumps(u))
    r = a.get("route", {})
    check("choice shape", r.get("type") == "choice" and r.get("choice") in ("support", "engineering")
          and set(r.get("probabilities", {})) == {"support", "engineering"}
          and approx_eq(sum(r["probabilities"].values()), 1.0, 1e-3)
          and 0.0 <= r.get("confidence", -1) <= 1.0, json.dumps(r))
    s = a.get("severity", {})
    check("score shape", s.get("type") == "score" and isinstance(s.get("score"), (int, float))
          and s.get("legend") == {"0": "Minor", "1": "Serious", "2": "Critical"}
          and set(s.get("probabilities", {})) == {"0", "1", "2"}
          and approx_eq(sum(s["probabilities"].values()), 1.0, 1e-3), json.dumps(s))
    usage = body.get("usage", {})
    check("usage ints", isinstance(usage.get("input_tokens"), int)
          and isinstance(usage.get("output_tokens"), int)
          and usage["input_tokens"] > 0 and usage["output_tokens"] >= 3, json.dumps(usage))
    check("model echo", body.get("model") == model)
    print("  golden answers:", json.dumps(a))

    # --- determinism --------------------------------------------------------
    # The engine's MoE/attention reductions are not bit-deterministic across
    # runs (ordinary chat shows the same jitter); the measured noise floor on
    # this B70 + INT4-AWQ checkpoint is ~2e-5 on label probabilities.
    _, _, body2 = post(base, "/v1/systemone", GOLDEN, key)
    check("determinism: identical repeat", answers_eq(a, body2.get("answers"), 1e-4))

    # --- isolation: alone vs multi-question ---------------------------------
    for qid in GOLDEN["questions"]:
        solo = {"model": model, "state": GOLDEN["state"],
                "questions": {qid: GOLDEN["questions"][qid]}}
        _, _, solo_body = post(base, "/v1/systemone", solo, key)
        check(f"isolation: {qid} alone == in group",
              answers_eq({qid: a[qid]}, solo_body.get("answers"), 1e-4),
              f"{json.dumps(a[qid])} vs {json.dumps((solo_body.get('answers') or {}).get(qid))}")

    # --- isolation: rename + reorder + contradictory distractor -------------
    renamed = {"model": model, "state": GOLDEN["state"], "questions": {
        "zz_unrelated": {"type": "noul", "instructions": "Is the sky green?"},
        "renamed_route": GOLDEN["questions"]["route"],
        "aa_urgent": GOLDEN["questions"]["urgent"],
        "sev2": GOLDEN["questions"]["severity"],
    }}
    _, _, ren = post(base, "/v1/systemone", renamed, key)
    ra = ren.get("answers", {})
    check("isolation: renamed ids keep answers",
          answers_eq({"route": a["route"]}, {"route": ra.get("renamed_route")}, 1e-4))
    check("isolation: distractor does not move urgent",
          answers_eq({"urgent": a["urgent"]}, {"urgent": ra.get("aa_urgent")}, 1e-4))

    # --- validation errors ---------------------------------------------------
    dup_raw = b'{"model":"%s","state":"s","questions":{"q":{"type":"noul"},"q":{"type":"noul"}}}' % model.encode()
    st, _, b = post(base, "/v1/systemone", dup_raw, key, raw=True)
    check("duplicate question key -> 422", st == 422, f"got {st}")

    st, _, b = post(base, "/v1/systemone",
                    dict(GOLDEN, seed=1234), key)
    check("control field seed -> 422", st == 422, f"got {st}")

    st, _, b = post(base, "/v1/systemone", dict(GOLDEN, model="jev-latest"), key)
    check("unknown model -> 404 (alias unset)", st == 404, f"got {st}")

    st, _, _ = post(base, "/v1/systemone", GOLDEN, key="wrong")
    check("bad bearer -> 401", st == 401, f"got {st}")

    st, _, b = post(base, "/v1/systemone",
                    {"model": model, "questions": {"q": {"type": "noul"}}}, key)
    check("missing state -> 422", st == 422, f"got {st}")

    st, _, b = post(base, "/v1/systemone",
                    {"model": model, "state": "x",
                     "questions": {"q": {"type": "bogus"}}}, key)
    check("unknown type -> 422", st == 422, f"got {st}")

    # --- limits ----------------------------------------------------------------
    crit = {f"opt{i:03d}": f"Option number {i}" for i in range(255)}
    st, _, b = post(base, "/v1/systemone",
                    {"model": model, "state": "Pick option 7.",
                     "questions": {"big": {"type": "choice",
                                           "instructions": "Which option?",
                                           "criteria": crit}}}, key)
    ok255 = st == 200 and len(b.get("answers", {}).get("big", {}).get("probabilities", {})) == 255
    check("choice 255 options -> 200 with 255 probabilities", ok255, f"got {st}: {str(b)[:300]}")

    crit = {f"opt{i:03d}": None for i in range(256)}
    st, _, _ = post(base, "/v1/systemone",
                    {"model": model, "state": "x",
                     "questions": {"big": {"type": "choice", "criteria": crit}}}, key)
    check("choice 256 options -> 422", st == 422, f"got {st}")

    # Explicit intermediate Choice sizes.
    for n in (26, 27, 128):
        crit = {f"opt{i:03d}": f"Option {i}" for i in range(n)}
        st, _, b = post(base, "/v1/systemone",
                        {"model": model, "state": "Pick an option.",
                         "questions": {"q": {"type": "choice",
                                             "instructions": "Which option?",
                                             "criteria": crit}}}, key)
        got = b.get("answers", {}).get("q", {}).get("probabilities", {}) if b else {}
        check(f"choice {n} options -> 200 with {n} probabilities",
              st == 200 and len(got) == n, f"got {st}: {str(b)[:250]}")

    st, _, b = post(base, "/v1/systemone",
                    {"model": model, "state": "Rate it.",
                     "questions": {"s": {"type": "score", "instructions": "Rate.",
                                         "criteria": [f"level {i}" for i in range(10)]}}}, key)
    ok10 = st == 200 and len(b.get("answers", {}).get("s", {}).get("probabilities", {})) == 10
    check("score 10 levels -> 200", ok10, f"got {st}: {str(b)[:300]}")

    st, _, _ = post(base, "/v1/systemone",
                    {"model": model, "state": "x",
                     "questions": {"s": {"type": "score",
                                         "criteria": [f"l{i}" for i in range(11)]}}}, key)
    check("score 11 levels -> 422", st == 422, f"got {st}")

    # --- structured/unicode content -------------------------------------------
    st, _, b = post(base, "/v1/systemone",
                    {"model": model,
                     "state": [{"e": "outage", "dc": "eu-centräl"}, {"sev": "高"}],
                     "questions": {"u": {"type": "choice", "instructions": {"ask": "数据中心?"},
                                         "criteria": {"东京": None, "法兰克福": "EU central"}}}}, key)
    check("unicode options/structured state -> 200", st == 200, f"got {st}: {str(b)[:300]}")

    # --- diagnostics extension --------------------------------------------------
    st, _, b = post(base, "/v1/systemone", dict(GOLDEN, arcaine={"diagnostics": True}), key)
    d = (b.get("arcaine") or {}).get("diagnostics", {})
    # reads depends on the server policy (ARCAINE_SYSTEMONE_READS*). Accept any
    # positive count and require the read-noise fields to be null only at N=1.
    ok = (st == 200 and set(d) == set(GOLDEN["questions"])
          and all("label_mass" in v and "vocab_entropy" in v and v.get("reads", 0) >= 1
                  and ((v.get("selected_label_stderr") is None) == (v.get("reads") == 1))
                  for v in d.values()))
    check("diagnostics extension", ok, f"got {st}: {str(b)[:400]}")
    if ok:
        print("  diagnostics:", json.dumps(d, indent=1)[:800])

    ok_life = st == 200 and all(
        v.get("prefill_calls") == 1 and v.get("decode_calls") == v.get("reads")
        for v in d.values())
    check("lifecycle: 1 prefill + N decode per question", ok_life,
          json.dumps({k: (v.get("prefill_calls"), v.get("decode_calls"), v.get("reads"))
                      for k, v in d.items()}))

    ext = b.get("arcaine", {}) if st == 200 else {}
    check("graph bypass: 0 captures / 0 replays in a read",
          ext.get("graph_captures_delta") == 0 and ext.get("graph_replays_delta") == 0,
          json.dumps(ext))

    st, _, b = post(base, "/v1/systemone", GOLDEN, key)
    check("diagnostics absent by default", st == 200 and "arcaine" not in b)

    # --- arbitrary / empty question IDs are accepted ---------------------------
    st, _, b = post(base, "/v1/systemone",
                    {"model": model, "state": "Is the sky green?",
                     "questions": {"": {"type": "noul",
                                        "instructions": "Is the sky green?"}}}, key)
    ok_empty = (st == 200 and "" in b.get("answers", {})
                and b["answers"][""].get("type") == "noul")
    check("empty question id accepted as a response key", ok_empty, f"got {st}: {str(b)[:200]}")

    # --- concurrent callers serialize safely ----------------------------------
    def one_call(_):
        return post(base, "/v1/systemone", GOLDEN, key)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(one_call, range(6)))
    ok_conc = all(st == 200 and answers_eq(a, body.get("answers"), 1e-4)
                  for (st, _, body) in results)
    check("concurrent callers: all 200 and match isolated answers", ok_conc,
          str([(st, (body or {}).get("error")) for (st, _, body) in results]))

    # --- mixed chat + structured concurrency ----------------------------------
    def mixed(kind):
        if kind == "chat":
            return post(base, "/v1/chat/completions",
                        {"model": model,
                         "messages": [{"role": "user", "content": "Say hello in one word."}],
                         "max_tokens": 16}, key)
        return post(base, "/v1/systemone", GOLDEN, key)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        mixed_results = list(ex.map(mixed, ["structured", "chat"] * 3))
    check("mixed chat/structured concurrency: all 200",
          all(st == 200 for (st, _, _) in mixed_results),
          str([(st, (body or {}).get("error")) for (st, _, body) in mixed_results]))

    # --- ordinary chat unaffected ------------------------------------------------
    st, _, b = post(base, "/v1/chat/completions",
                    {"model": model,
                     "messages": [{"role": "user", "content": "Say hello in one word."}],
                     "max_tokens": 32}, key)
    ok = st == 200 and b.get("choices")
    check("chat completions still works", ok, f"got {st}: {str(b)[:200]}")
    if ok:
        print("  chat reply:", json.dumps(b["choices"][0]["message"].get("content"))[:120])

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:", *FAILURES, sep="\n  - ")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
