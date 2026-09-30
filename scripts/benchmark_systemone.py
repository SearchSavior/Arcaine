#!/usr/bin/env python3
"""End-to-end /v1/systemone benchmark on the running arcaine_server.

Sweeps the AGENTS.md prompt lengths (512,1024,2048,4096) with a filler that is
calibrated so the MEASURED prompt tokens land near each target. Reports p50/p95
request latency, the server-side synchronized prefill/decode split, the measured
prompt token counts and canvas widths (via arcaine diagnostics), reads/second,
and completed decisions/second.

Throughput terms:
  * reads/second      - scored answer slots per wall second (includes repeated
                        reads of the same question);
  * decisions/second  - completed question answers per wall second.
At four reads per question these differ by the number of questions.

The server's read policy is a deployment setting (ARCAINE_SYSTEMONE_READS /
ARCAINE_SYSTEMONE_READS_AUTO_*); run one server configuration per leg, e.g.:

  # leg A: 1 read/question (default)
  python3 scripts/benchmark_systemone.py -p 512,1024,2048,4096
  # leg B: 4 reads/question (server started with ARCAINE_SYSTEMONE_READS=4)
  python3 scripts/benchmark_systemone.py -p 512,1024,2048,4096

The structured canvas is the compiled answer template padded to the server's
canvas step (16 tokens here). It is not a caller-tunable width, so the sweep
reports the measured width instead of sweeping it. Canvas capacity is a load-time
model setting, not a request field.
"""
import argparse
import json
import statistics
import time
import urllib.request

UNIT = ("The datacenter report covers routine capacity, power, and cooling "
        "metrics for the quarter. ")


def make_request(model, filler_repeats):
    state = {"ticket": UNIT * max(0, filler_repeats) +
             " Everything is down and we have a demo at noon."}
    return {
        "model": model,
        "state": state,
        "questions": {
            "urgent": {"type": "noul",
                       "instructions": "Does this require a response within an hour?",
                       "criteria": {"true": "A service outage needs prompt action",
                                    "false": "The request can wait"}},
            "route": {"type": "choice",
                      "instructions": "Choose the responsible team.",
                      "criteria": {"support": "Customer support",
                                   "engineering": "Service outages and software faults"}},
            "severity": {"type": "score", "instructions": "Rate the incident.",
                         "criteria": ["Minor", "Serious", "Critical"]},
        },
        "arcaine": {"diagnostics": True},
    }


def post(base, key, body):
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json",
                                          "authorization": "Bearer " + key})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        payload = json.loads(r.read())
    return (time.perf_counter() - t0) * 1e3, payload


def measured_tokens(payload):
    d = payload.get("arcaine", {}).get("diagnostics", {})
    return max((v.get("prompt_tokens", 0) for v in d.values()), default=0)


def calibrate(base, key, model, target):
    """Pick a filler repeat count whose measured prompt tokens are near target."""
    def probe(rep):
        _, payload = post(base, key, make_request(model, rep))
        return measured_tokens(payload)

    r = max(1, target // 20)
    t = probe(r)
    if t <= 0:
        return r, t
    for _ in range(3):
        if abs(t - target) <= 0.05 * target:
            break
        r = max(1, round(r * target / t))
        t = probe(r)
    return r, t


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prompts", default="512,1024,2048,4096")
    ap.add_argument("--base", default="http://127.0.0.1:7461")
    ap.add_argument("--key", default="local")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--no-calibrate", action="store_true",
                    help="use a fixed filler heuristic instead of measured calibration")
    args = ap.parse_args()

    req = urllib.request.Request(args.base + "/v1/models")
    req.add_header("authorization", "Bearer " + args.key)
    with urllib.request.urlopen(req, timeout=30) as r:
        model = json.loads(r.read())["data"][0]["id"]

    print(f"model={model} iters={args.iters} warmup={args.warmup}")
    print(f"{'target':>7} {'q_toks':>8} {'canvas':>7} {'reads':>6} "
          f"{'p50_ms':>9} {'p95_ms':>9} {'prefill_ms':>11} {'decode_ms':>10} "
          f"{'reads/s':>9} {'dec/s':>7}")
    for plen in [int(x) for x in args.prompts.split(",")]:
        if args.no_calibrate:
            filler, toks = max(1, plen // 15), None
        else:
            filler, toks = calibrate(args.base, args.key, model, plen)
        body = make_request(model, filler)
        lat, prefill, decode = [], [], []
        q_toks_lo = q_toks_hi = 0
        canvas_lo = canvas_hi = 0
        total_reads = total_decisions = 0
        total_wall = 0.0
        for i in range(args.warmup + args.iters):
            ms, payload = post(args.base, args.key, body)
            if i < args.warmup:
                continue
            lat.append(ms)
            total_wall += ms / 1e3
            d = payload.get("arcaine", {}).get("diagnostics", {})
            # Aggregate the actual work across every timed request.
            total_decisions += len(d)
            total_reads += sum(v.get("reads", 0) for v in d.values())
            toks_list = [v.get("prompt_tokens", 0) for v in d.values()]
            canvas = [v.get("canvas_width", 0) for v in d.values()]
            q_toks_lo, q_toks_hi = min(toks_list), max(toks_list)
            canvas_lo, canvas_hi = min(canvas), max(canvas)
            prefill.append(sum(v["prefill_ms"] for v in d.values()))
            decode.append(sum(v["decode_ms"] for v in d.values()))
        reads_per_s = total_reads / total_wall if total_wall > 0 else float("nan")
        dec_per_s = total_decisions / total_wall if total_wall > 0 else float("nan")
        q_range = f"{q_toks_lo}-{q_toks_hi}" if q_toks_lo != q_toks_hi else str(q_toks_lo)
        c_range = f"{canvas_lo}-{canvas_hi}" if canvas_lo != canvas_hi else str(canvas_lo)
        print(f"{plen:>7} {q_range:>8} {c_range:>7} {total_reads:>6} "
              f"{statistics.median(lat):>9.1f} {pct(lat, 0.95):>9.1f} "
              f"{statistics.median(prefill):>11.1f} {statistics.median(decode):>10.1f} "
              f"{reads_per_s:>9.1f} {dec_per_s:>7.1f}")


if __name__ == "__main__":
    main()
