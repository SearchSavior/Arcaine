#!/usr/bin/env python3
"""Decode-path + NVFP4 benchmark sweep harness.

Stdlib-only. Invokes the already-built ``diffusion_bench`` binary with different
env vars / args, parses per-run ``[bench]`` lines into a SQLite DB. Resumable.

The harness changes no C++; it only drives the binary. See the plan
(super-valued-moccasin.md) and memory note ``diffusion_bench_harness.md`` for the
verified CLI/output contract.
"""
import argparse
import collections
import datetime
import itertools
import os
import re
import sqlite3
import subprocess
import sys
import time

# Directory this script lives in (bench/sweep/). DB + logs default to here so the
# harness is cwd-independent.
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Axis -> env mapping (see plan: "Axis -> env-var mapping").
#   - "device" sampler and "exact" soft_next mean the env var is UNSET, so the
#     harness explicitly drops every DIFF_* knob from the inherited env before
#     adding the axis-specific ones (see compose_env).
# ---------------------------------------------------------------------------

# All env vars the harness manages. These are stripped from the inherited env so
# that "unset" really means unset (no leakage from the parent shell).
MANAGED_ENV = [
    "DIFF_HOST_SAMPLER",
    "DIFF_SOFT_NEXT",
    "DIFF_DECODE_KV_DIRECT_CACHE",
    "DIFF_STAGE_DECODER_KV_KERNEL",
    "DIFF_ONEDNN_SDPA",
    "DIFF_FORCE_DENOISE_STEPS",
    "DIFF_SKIP_LAST_SOFT_NEXT",
    "DIFF_NVFP4_GROUPED_GEMM",
    "DIFF_NVFP4_GPU_LAYOUT",
    "DIFF_NVFP4_GPU_LAYOUT_MAX_SEQ",
    "ZE_AFFINITY_MASK",
]

KERN_DEFAULT = "default,hybrid,custom"


def env_gpu(v):
    return {"ZE_AFFINITY_MASK": "0" if v == "single" else "0,1"}


def env_sampler(v):
    return {} if v == "device" else {"DIFF_HOST_SAMPLER": "1"}


def env_soft_next(v):
    return {} if v == "exact" else {"DIFF_SOFT_NEXT": v}  # hard | topk:8


def env_s1(v):
    out = {}
    if v == "off":
        return out
    if v in ("decode_kv", "all"):
        out["DIFF_DECODE_KV_DIRECT_CACHE"] = "1"
    if v in ("stage_kv", "all"):
        out["DIFF_STAGE_DECODER_KV_KERNEL"] = "1"
    if v in ("onednn_sdpa", "all"):
        out["DIFF_ONEDNN_SDPA"] = "decode"
    return out


def env_force_denoise(v):
    return {} if v == "off" else {"DIFF_FORCE_DENOISE_STEPS": "1"}


def env_skip_last(v):
    return {} if v == "off" else {"DIFF_SKIP_LAST_SOFT_NEXT": "1"}


def env_grouped_gemm(v):
    # None (decode group, not an axis) and "off" both leave the knob unset.
    return {} if v in (None, "off") else {"DIFF_NVFP4_GROUPED_GEMM": "xe2"}


def env_gpu_layout(v):
    # gpu_layout=on also raises the decode-seq gate so the path actually engages
    # (canvas seq 256 > default gate 64). None (decode group) -> unset.
    if v in (None, "off"):
        return {}
    return {
        "DIFF_NVFP4_GPU_LAYOUT": "1",
        "DIFF_NVFP4_GPU_LAYOUT_MAX_SEQ": "512",
    }


# ---------------------------------------------------------------------------
# Launch plan
# ---------------------------------------------------------------------------

def _decode_axes(gpu, sampler, soft_next, s1, force, skip):
    return {
        "sweep_group": "decode",
        "gpu": gpu,
        "sampler": sampler,
        "soft_next": soft_next,
        "s1": s1,
        "force_denoise": force,
        "skip_last_soft_next": skip,
        "grouped_gemm": None,   # not an axis in this group
        "gpu_layout": None,
    }


def _nvfp4_axes(gpu, grouped_gemm, gpu_layout):
    return {
        "sweep_group": "nvfp4",
        "gpu": gpu,
        # Fixed "expected-best" decode config (plan, Group 2).
        "sampler": "device",
        "soft_next": "hard",
        "s1": "all",
        "force_denoise": "on",
        "skip_last_soft_next": "on",
        "grouped_gemm": grouped_gemm,
        "gpu_layout": gpu_layout,
    }


def _env_from_axes(axes):
    env = {}
    env.update(env_gpu(axes["gpu"]))
    env.update(env_sampler(axes["sampler"]))
    env.update(env_soft_next(axes["soft_next"]))
    env.update(env_s1(axes["s1"]))
    env.update(env_force_denoise(axes["force_denoise"]))
    env.update(env_skip_last(axes["skip_last_soft_next"]))
    env.update(env_grouped_gemm(axes["grouped_gemm"]))
    env.update(env_gpu_layout(axes["gpu_layout"]))
    return env


def build_plan(groups):
    """Return list of cells: {"axes": {...}, "env": {...}, "kernels": "..."}."""
    plan = []
    if "decode" in groups:
        for gpu, sampler, soft_next, s1, force, skip in itertools.product(
            ["single", "dual"],
            ["device", "host"],
            ["exact", "hard", "topk"],
            ["off", "decode_kv", "stage_kv", "onednn_sdpa", "all"],
            ["off", "on"],
            ["off", "on"],
        ):
            axes = _decode_axes(gpu, sampler, soft_next, s1, force, skip)
            plan.append({"axes": axes, "env": _env_from_axes(axes),
                         "kernels": KERN_DEFAULT})
    if "nvfp4" in groups:
        for gpu, gg, gl in itertools.product(
            ["single", "dual"], ["off", "xe2"], ["off", "on"]
        ):
            axes = _nvfp4_axes(gpu, gg, gl)
            plan.append({"axes": axes, "env": _env_from_axes(axes),
                         "kernels": KERN_DEFAULT})
    return plan


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

COLS = [
    "ts", "git_sha", "sweep_group", "gpu", "sampler", "soft_next", "s1",
    "force_denoise", "skip_last_soft_next", "grouped_gemm", "gpu_layout",
    "kernel", "n_prompt", "n_gen", "ds", "run_idx",
    "prefill_tps", "output_tps", "canvas_pos_s", "fwd_s", "passes",
    "kv_mb_per_tok", "arena_gb", "tok_per_fwd",
]

# Columns that uniquely identify a cell's expected output (exclude kernel + run_idx,
# which are swept in-process / per-timed-run). Used for resumability.
KEY_COLS = [
    "sweep_group", "gpu", "sampler", "soft_next", "s1", "force_denoise",
    "skip_last_soft_next", "grouped_gemm", "gpu_layout", "n_prompt", "n_gen", "ds",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id            INTEGER PRIMARY KEY,
  ts            TEXT,
  git_sha       TEXT,
  sweep_group   TEXT,
  gpu           TEXT,
  sampler       TEXT,
  soft_next     TEXT,
  s1            TEXT,
  force_denoise TEXT,
  skip_last_soft_next TEXT,
  grouped_gemm  TEXT,
  gpu_layout    TEXT,
  kernel        TEXT,
  n_prompt      INTEGER,
  n_gen         INTEGER,
  ds            INTEGER,
  run_idx       INTEGER,
  prefill_tps   REAL,
  output_tps    REAL,
  canvas_pos_s  REAL,
  fwd_s         REAL,
  passes        INTEGER,
  kv_mb_per_tok REAL,
  arena_gb      REAL,
  tok_per_fwd   REAL
);
CREATE INDEX IF NOT EXISTS idx_combo ON runs(
  gpu, sampler, soft_next, s1, force_denoise, skip_last_soft_next,
  grouped_gemm, gpu_layout, kernel
);
"""


def open_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def cell_key(cell, args):
    a = cell["axes"]
    return {
        "sweep_group": a["sweep_group"],
        "gpu": a["gpu"],
        "sampler": a["sampler"],
        "soft_next": a["soft_next"],
        "s1": a["s1"],
        "force_denoise": a["force_denoise"],
        "skip_last_soft_next": a["skip_last_soft_next"],
        "grouped_gemm": a["grouped_gemm"],
        "gpu_layout": a["gpu_layout"],
        "n_prompt": args.p,
        "n_gen": args.n,
        "ds": args.ds,
    }


def _where(key):
    # IS handles NULL grouped_gemm/gpu_layout for the decode group.
    return " AND ".join(f"{c} IS ?" for c in key)


def cell_count(cur, key):
    cur.execute(f"SELECT COUNT(*) FROM runs WHERE {_where(key)}",
                tuple(key.values()))
    return cur.fetchone()[0]


def delete_cell(cur, key):
    cur.execute(f"DELETE FROM runs WHERE {_where(key)}", tuple(key.values()))


def insert_row(cur, row):
    cur.execute(
        "INSERT INTO runs (" + ",".join(COLS) + ") VALUES (" +
        ",".join("?" * len(COLS)) + ")",
        tuple(row.get(c) for c in COLS),
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Matches diffusion_bench.cpp:292-298 (one line per timed run):
#   [bench] <kernel> p<N> n<N> run i/R: prefill X t/s | output Y t/s |
#   canvas Z pos/s | W fwd/s | P passes | kv K MB/token | arena A GB
BENCH_RE = re.compile(
    r"\[bench\]\s+(\S+)\s+p(\d+)\s+n(\d+)\s+run\s+(\d+)/(\d+):\s+"
    r"prefill\s+([-+0-9.]+)\s+t/s\s+\|\s+"
    r"output\s+([-+0-9.]+)\s+t/s\s+\|\s+"
    r"canvas\s+([-+0-9.]+)\s+pos/s\s+\|\s+"
    r"([-+0-9.]+)\s+fwd/s\s+\|\s+"
    r"(\d+)\s+passes\s+\|\s+"
    r"kv\s+([-+0-9.]+)\s+MB/token\s+\|\s+"
    r"arena\s+([-+0-9.]+)\s+GB"
)


def parse_bench_line(line):
    m = BENCH_RE.search(line)
    if not m:
        return None
    (kernel, p, n, run_i, run_r, prefill, output, canvas,
     fwd, passes, kv, arena) = m.groups()
    fwd_f = float(fwd)
    return {
        "kernel": kernel,
        "n_prompt": int(p),
        "n_gen": int(n),
        "run_idx": int(run_i),
        "prefill_tps": float(prefill),
        "output_tps": float(output),
        "canvas_pos_s": float(canvas),
        "fwd_s": fwd_f,
        "passes": int(passes),
        "kv_mb_per_tok": float(kv),
        "arena_gb": float(arena),
        "tok_per_fwd": (float(output) / fwd_f) if fwd_f else 0.0,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def combo_str(cell):
    a = cell["axes"]
    parts = ["grp=" + a["sweep_group"], "gpu=" + a["gpu"]]
    for c in ("sampler", "soft_next", "s1", "force_denoise",
              "skip_last_soft_next", "grouped_gemm", "gpu_layout"):
        parts.append(c + "=" + str(a[c]))
    return ",".join(parts)


def compose_env(axis_env):
    """Inherit the parent env (PATH/ONEAPI/LD_LIBRARY_PATH/...) but drop every
    knob the harness manages, then add the axis-specific ones."""
    env = dict(os.environ)
    for v in MANAGED_ENV:
        env.pop(v, None)
    env.update(axis_env)
    return env


def fmt_eta(secs):
    if secs is None or secs < 0 or secs != secs:  # nan check
        return "?"
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def git_sha(repo_dir):
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Run one process (one cell)
# ---------------------------------------------------------------------------

def run_cell(conn, cur, args, git_sha_val, cell, key, logf, failf, verbose):
    """Launch one process, parse + insert rows. Returns (n_parsed, rc, elapsed)."""
    env = compose_env(cell["env"])
    cmd = [
        args.binary, "--model", args.model,
        "-p", str(args.p), "-n", str(args.n), "-ds", str(args.ds),
        "-w", str(args.warmup), "-r", str(args.runs),
        "--kernels", cell["kernels"],
    ]
    t0 = time.monotonic()
    recent = collections.deque(maxlen=60)  # for failure diagnostics
    parsed = 0
    try:
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except FileNotFoundError:
        msg = f"[sweep] binary not found: {args.binary}"
        print(msg, file=sys.stderr)
        failf.write(f"--- {now_iso()} {combo_str(cell)} BINARY-MISSING\n")
        failf.flush()
        return 0, 127, time.monotonic() - t0

    with proc:
        for line in proc.stdout:
            recent.append(line.rstrip())
            if verbose:
                sys.stdout.write(line); sys.stdout.flush()
            rec = parse_bench_line(line)
            if rec is not None:
                row = dict(key)  # the axis/workload identifying columns
                row["ts"] = now_iso()
                row["git_sha"] = git_sha_val
                row["kernel"] = rec["kernel"]
                row["run_idx"] = rec["run_idx"]
                row["n_prompt"] = rec["n_prompt"]
                row["n_gen"] = rec["n_gen"]
                row["ds"] = args.ds
                for c in ("prefill_tps", "output_tps", "canvas_pos_s", "fwd_s",
                          "passes", "kv_mb_per_tok", "arena_gb", "tok_per_fwd"):
                    row[c] = rec[c]
                insert_row(cur, row)
                conn.commit()
                parsed += 1
        rc = proc.wait()
    elapsed = time.monotonic() - t0

    if rc != 0 or parsed == 0:
        failf.write(f"--- {now_iso()} {combo_str(cell)} rc={rc} parsed={parsed}\n")
        for ln in recent:
            failf.write("    " + ln + "\n")
        failf.flush()
    return parsed, rc, elapsed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def list_models():
    """--list-models: os.listdir('/workspace/models') (container path), with a
    graceful fallback to ./models so the flag also works on the host."""
    for cand in ("/workspace/models", os.path.join(SWEEP_DIR, "..", "..", "models")):
        if os.path.isdir(cand):
            entries = sorted(os.listdir(cand))
            print(f"# {cand} ({len(entries)} entries)")
            for e in entries:
                print(e)
            return 0
    print("no /workspace/models and no ./models directory found", file=sys.stderr)
    return 1


def dry_run(plan, args):
    print(f"# sweep plan: {len(plan)} process launches")
    cells = len(plan) * 3  # 3 kernels in-process
    runs = cells * args.runs
    print(f"# cells={cells} timed_runs={runs} (kernels in-process, runs={args.runs})")
    print("# group      gpu    sampler soft_next s1         force skip  grouped_gemm gpu_layout kernels")
    for cell in plan:
        a = cell["axes"]
        print(f"  {a['sweep_group']:10s} {a['gpu']:6s} {a['sampler']:7s} "
              f"{a['soft_next']:9s} {a['s1']:10s} {a['force_denoise']:5s} "
              f"{a['skip_last_soft_next']:4s} {str(a['grouped_gemm']):11s} "
              f"{str(a['gpu_layout']):10s} {cell['kernels']}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Decode-path + NVFP4 benchmark sweep harness.",
    )
    p.add_argument("--model", help="model directory (required unless --list-models)")
    p.add_argument("--binary", default="build/diffusion_bench",
                   help="path to diffusion_bench (default: build/diffusion_bench)")
    p.add_argument("--p", type=int, default=512, help="prompt tokens (default 512)")
    p.add_argument("--n", type=int, default=128, help="generated tokens (default 128)")
    p.add_argument("--ds", type=int, default=48, help="denoising steps (default 48)")
    p.add_argument("--warmup", "-w", type=int, default=2, help="warmup runs (default 2)")
    p.add_argument("--runs", "-r", type=int, default=3, help="timed runs (default 3)")
    p.add_argument("--groups", default="decode,nvfp4",
                   help="comma list of decode,nvfp4 (default: decode,nvfp4)")
    p.add_argument("--db", default=os.path.join(SWEEP_DIR, "results.db"),
                   help="SQLite DB path")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p.add_argument("--force", action="store_true",
                   help="re-run completed cells (deletes existing rows first)")
    p.add_argument("--list-models", action="store_true",
                   help="list /workspace/models and exit")
    p.add_argument("--verbose", action="store_true",
                   help="tee binary stdout to console")
    args = p.parse_args(argv)

    if args.list_models:
        return list_models()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    bad = [g for g in groups if g not in ("decode", "nvfp4")]
    if bad:
        p.error(f"unknown group(s): {bad} (valid: decode, nvfp4)")

    if not args.model:
        p.error("--model is required (or use --list-models)")

    plan = build_plan(groups)

    if args.dry_run:
        return dry_run(plan, args)

    if not os.path.isfile(args.binary):
        p.error(f"binary not found: {args.binary} (build it first)")

    conn = open_db(args.db)
    cur = conn.cursor()
    git_sha_val = git_sha(os.getcwd())
    expected_per_cell = 3 * args.runs  # 3 kernels in-process x timed runs

    log_path = os.path.join(SWEEP_DIR, "sweep.log")
    fail_path = os.path.join(SWEEP_DIR, "sweep.failures.log")
    logf = open(log_path, "a", buffering=1)
    failf = open(fail_path, "a", buffering=1)

    header = (f"[sweep] start {now_iso()} git={git_sha_val} "
              f"model={args.model} p={args.p} n={args.n} ds={args.ds} "
              f"w={args.warmup} r={args.runs} groups={','.join(groups)} "
              f"plan={len(plan)} db={args.db}")
    print(header, file=sys.stderr)
    logf.write(header + "\n")

    run_elapsed = []  # only actually-launched processes (for ETA)
    N = len(plan)
    skipped = 0
    failed = 0
    done_cells = 0
    for i, cell in enumerate(plan, 1):
        key = cell_key(cell, args)
        count = cell_count(cur, key)
        if args.force:
            if count:
                delete_cell(cur, key)
                conn.commit()
        elif count >= expected_per_cell:
            skipped += 1
            msg = (f"[sweep] {i}/{N} {combo_str(cell)} SKIP "
                   f"(have {count}/{expected_per_cell})")
            print(msg, file=sys.stderr)
            continue
        elif 0 < count < expected_per_cell:
            # partial -> delete and re-run to avoid duplicates
            delete_cell(cur, key)
            conn.commit()

        msg = f"[sweep] {i}/{N} {combo_str(cell)} running..."
        print(msg, file=sys.stderr)

        parsed, rc, elapsed = run_cell(
            conn, cur, args, git_sha_val, cell, key, logf, failf, args.verbose,
        )
        run_elapsed.append(elapsed)
        if parsed == 0 or rc != 0:
            failed += 1
        done_cells = i
        mean = sum(run_elapsed) / len(run_elapsed) if run_elapsed else 0.0
        eta = mean * (N - i)
        msg = (f"[sweep] {i}/{N} {combo_str(cell)} parsed={parsed} rc={rc} "
               f"elapsed={elapsed:.1f}s eta={fmt_eta(eta)} "
               f"(skipped={skipped} failed={failed})")
        print(msg, file=sys.stderr)
        logf.write(msg + "\n")

    summary = (f"[sweep] done {now_iso()} launched={done_cells - skipped}/{N} "
               f"skipped={skipped} failed={failed} db={args.db}")
    print(summary, file=sys.stderr)
    logf.write(summary + "\n")
    logf.close()
    failf.close()
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
