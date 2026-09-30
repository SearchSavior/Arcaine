#!/usr/bin/env python3
"""Small labeled behavior evaluation for /v1/systemone.

Runs a fixed labeled case set through the structured endpoint and, optionally,
through ordinary chat generation, then reports accuracy, Brier score, coverage
at confidence/label-mass thresholds, and repeatability. This is a small fixture
set: it detects gross behavior problems , not calibration.

Scoring rules (defined before results):
  * unknown/ambiguous cases have no gold label and count only toward coverage;
  * Noul Brier uses (p_yes - y)^2 with y in {0,1};
  * Choice/Score Brier is the multiclass Brier sum_j (p_j - onehot_j)^2;
  * a case is "covered" when confidence >= --conf-threshold and label mass >=
    --mass-threshold (Noul has no confidence, so it uses label mass only).

Usage:
  python3 scripts/eval_systemone_behavior.py --base http://127.0.0.1:7461 --key local
  python3 scripts/eval_systemone_behavior.py --base ... --ordinary
"""
import argparse
import json
import urllib.error
import urllib.request

# gold: index into the answer keys for choice/score, or 1/0 for noul; None = unknown.
CASES = [
    {"id": "noul_outage", "state": "Everything is down and the demo is at noon.",
     "q": {"type": "noul", "instructions": "Is there an outage?",
           "criteria": {"true": "a service outage", "false": "no service outage"}}, "gold": 1},
    {"id": "noul_normal", "state": "All systems report normal and fast.",
     "q": {"type": "noul", "instructions": "Is there an outage?",
           "criteria": {"true": "a service outage", "false": "no service outage"}}, "gold": 0},
    {"id": "noul_negation", "state": "There is no outage; every check passed.",
     "q": {"type": "noul", "instructions": "Is there an outage?",
           "criteria": {"true": "a service outage", "false": "no service outage"}}, "gold": 0},
    {"id": "noul_unknown", "state": "The incident report is still pending.",
     "q": {"type": "noul", "instructions": "Is there an outage?",
           "criteria": {"true": "a service outage", "false": "no service outage"}}, "gold": None},
    {"id": "choice_eng", "state": "Disk is full and writes are failing on the primary.",
     "q": {"type": "choice", "instructions": "Which team owns this?",
           "criteria": {"support": "customer billing", "engineering": "software faults"}},
     "gold_key": "engineering"},
    {"id": "choice_support", "state": "The customer asks how to update the billing address.",
     "q": {"type": "choice", "instructions": "Which team owns this?",
           "criteria": {"support": "customer billing", "engineering": "software faults"}},
     "gold_key": "support"},
    {"id": "score_critical", "state": "Everything is down and the demo is at noon.",
     "q": {"type": "score", "instructions": "Rate the severity.",
           "criteria": ["minor", "moderate", "critical"]}, "gold": 2},
    {"id": "score_minor", "state": "A tooltip is misspelled in the settings page.",
     "q": {"type": "score", "instructions": "Rate the severity.",
           "criteria": ["minor", "moderate", "critical"]}, "gold": 0},
    {"id": "score_moderate", "state": "One replica is slow but the service is up.",
     "q": {"type": "score", "instructions": "Rate the severity.",
           "criteria": ["minor", "moderate", "critical"]}, "gold": 1},
    {"id": "num_boundary", "state": "Record 150 was inserted after record 120.",
     "q": {"type": "score", "instructions": "Which numeric range applies?",
           "criteria": ["below 100", "100 to 199", "200 or more"]}, "gold": 1},
]


def post(base, path, body, key, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json",
                                          "authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ordered_keys(ans, qtype):
    if qtype == "noul":
        return ["no", "yes"]
    if qtype == "choice":
        return sorted(ans["probabilities"])
    return sorted(ans["probabilities"], key=int)


def answer_probs(ans, qtype):
    return [ans["probabilities"][k] for k in ordered_keys(ans, qtype)] \
        if qtype != "noul" else [1.0 - ans["noul"], ans["noul"]]


def gold_index(case, keys):
    if "gold_key" in case:
        return keys.index(case["gold_key"])
    return case["gold"]


def run_structured(base, key, model):
    out = {}
    body = {"model": model, "state": None,
            "questions": {c["id"]: c["q"] for c in CASES},
            "arcaine": {"diagnostics": True}}
    # One request per case to keep label mass diagnostics aligned and isolate errors.
    for c in CASES:
        b = dict(body)
        b["state"] = c["state"]
        b["questions"] = {c["id"]: c["q"]}
        payload = post(base, "/v1/systemone", b, key)
        ans = payload["answers"][c["id"]]
        d = payload["arcaine"]["diagnostics"][c["id"]]
        out[c["id"]] = {"ans": ans,
                        "mass": max(d["label_mass"]) if d.get("label_mass") else 0.0}
    return out


def metrics(results, conf_thr, mass_thr):
    acc_num = acc_den = 0
    brier = []
    covered = 0
    for c in CASES:
        r = results[c["id"]]["ans"]
        probs = answer_probs(r, c["q"]["type"])
        mass = results[c["id"]]["mass"]
        conf = r.get("confidence", 1.0)
        if conf >= conf_thr and mass >= mass_thr:
            covered += 1
        if c.get("gold") is None and "gold_key" not in c:
            continue
        acc_den += 1
        if c["q"]["type"] == "noul":
            pred = 1 if r["noul"] >= 0.5 else 0
        else:
            pred = max(range(len(probs)), key=lambda j: probs[j])
        gi = gold_index(c, ordered_keys(r, c["q"]["type"]))
        acc_num += int(pred == gi)
        onehot = [1.0 if j == gi else 0.0 for j in range(len(probs))]
        brier.append(sum((p - o) ** 2 for p, o in zip(probs, onehot)))
    return {"acc": acc_num / acc_den if acc_den else float("nan"),
            "brier": sum(brier) / len(brier) if brier else float("nan"),
            "coverage": covered / len(CASES)}


def run_ordinary(base, key, model):
    acc_num = acc_den = nonempty = 0
    for c in CASES:
        if c.get("gold") is None and "gold_key" not in c:
            continue
        crit = c["q"]["criteria"]
        keys = ["no", "yes"] if c["q"]["type"] == "noul" else (
            sorted(crit) if isinstance(crit, dict)
            else [str(i) for i in range(len(crit))])
        gi = gold_index(c, keys)
        prompt = (f"State: {c['state']}\nQuestion: {c['q']['instructions']}\n"
                  f"Allowed answers: {keys}. Reply with exactly one answer.")
        payload = post(base, "/v1/chat/completions",
                       {"model": model, "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 8}, key)
        text = (payload["choices"][0]["message"].get("content") or "").strip().lower()
        gold = keys[gi] if 0 <= gi < len(keys) else ""
        acc_den += 1
        if text:
            nonempty += 1
        acc_num += int(gold in text)
    return (acc_num / acc_den if acc_den else float("nan")), nonempty, acc_den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:7461")
    ap.add_argument("--key", default="local")
    ap.add_argument("--conf-threshold", type=float, default=0.5)
    ap.add_argument("--mass-threshold", type=float, default=0.5)
    ap.add_argument("--ordinary", action="store_true")
    args = ap.parse_args()

    req = urllib.request.Request(args.base + "/v1/models")
    req.add_header("authorization", "Bearer " + args.key)
    with urllib.request.urlopen(req, timeout=30) as r:
        model = json.loads(r.read())["data"][0]["id"]

    first = run_structured(args.base, args.key, model)
    second = run_structured(args.base, args.key, model)
    m = metrics(first, args.conf_threshold, args.mass_threshold)
    repeat_ok = all(first[k]["ans"].get("choice") == second[k]["ans"].get("choice")
                    and abs(first[k]["ans"].get("score", 0) - second[k]["ans"].get("score", 0)) < 1e-3
                    for k in first)
    print(f"cases={len(CASES)} model={model}")
    print(f"structured accuracy={m['acc']:.3f} brier={m['brier']:.3f} "
          f"coverage(conf>={args.conf_threshold},mass>={args.mass_threshold})={m['coverage']:.3f}")
    print(f"repeatability (choice/score stable across two runs): {repeat_ok}")
    for c in CASES:
        r = first[c["id"]]["ans"]
        got = (r["choice"] if c["q"]["type"] == "choice"
               else round(r["score"], 3) if c["q"]["type"] == "score"
               else round(r["noul"], 3))
        gold = c.get("gold_key", c.get("gold"))
        print(f"  {c['id']:<16} gold={gold} got={got} "
              f"conf={r.get('confidence', float('nan')):.3f} mass={first[c['id']]['mass']:.3f}")
    if args.ordinary:
        oacc, nonempty, n = run_ordinary(args.base, args.key, model)
        print(f"ordinary generation: accuracy={oacc:.3f} on {n} cases; "
              f"{nonempty}/{n} replies were non-empty (empty replies cannot be parsed)")


if __name__ == "__main__":
    main()
