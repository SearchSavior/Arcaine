#!/usr/bin/env python3
"""Read the sweep SQLite DB and emit comparison PNGs (needs matplotlib).

Run:
    python3 bench/sweep/plot.py --db bench/sweep/results.db --out bench/sweep/plots
"""
import argparse
import math
import os
import sqlite3
import sys

# matplotlib is imported lazily (_ensure_mpl) so that --help and the pure
# helpers work without matplotlib installed (it is installed one-time inside
# the container, per the plan).
plt = None


def _ensure_mpl():
    global plt
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as _plt
        plt = _plt


SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))

GPUS = ["single", "dual"]
KERNELS = ["default", "hybrid", "custom"]
S1S = ["off", "decode_kv", "stage_kv", "onednn_sdpa", "all"]
SOFT_NEXTS = ["exact", "hard", "topk"]
FORCES = ["off", "on"]

SHORT = {
    "sampler": "sampler", "soft_next": "soft_next", "s1": "s1",
    "force_denoise": "force", "skip_last_soft_next": "skip",
    "kernel": "kernel", "grouped_gemm": "gg", "gpu_layout": "gl",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def mean_sd(vals):
    n = len(vals)
    if n == 0:
        return None, 0.0
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (n - 1)  # sample stddev
    return m, math.sqrt(var)


def where(filters):
    parts, params = [], []
    for k, v in filters.items():
        if v is None:
            parts.append(f"{k} IS NULL")
        else:
            parts.append(f"{k} = ?")
            params.append(v)
    return " AND ".join(parts), params


def note_str(filters, skip=("sweep_group",)):
    return " ".join(
        f"{SHORT.get(k, k)}={v}" for k, v in filters.items() if k not in skip
    )


def git_sha(cur):
    cur.execute("SELECT git_sha FROM runs ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    return row[0] if row and row[0] else "unknown"


def grouped_bar(ax, series, vary_vals, data, ylabel, plt):
    """data: {(series_val, vary_val): (mean, sd)}; bars grouped by vary_val."""
    n = len(series)
    width = 0.8 / max(n, 1)
    x = list(range(len(vary_vals)))
    for si, s in enumerate(series):
        means = [data.get((s, v), (float("nan"), 0.0))[0] for v in vary_vals]
        sds = [data.get((s, v), (float("nan"), 0.0))[1] for v in vary_vals]
        offs = [xi + (si - (n - 1) / 2) * width for xi in x]
        ax.bar(offs, means, width=width, yerr=sds, label=s, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in vary_vals])
    ax.set_ylabel(ylabel)
    ax.legend(title="gpu")
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def bar_plot(cur, outdir, fname, title, filters, vary_col, vary_vals, ylabel,
             sha, series=GPUS):
    where_sql, params = where(filters)
    sql = f"SELECT gpu, {vary_col}, output_tps FROM runs WHERE {where_sql}"
    cur.execute(sql, params)
    agg = {}
    for gpu, vval, out in cur.fetchall():
        agg.setdefault((gpu, vval), []).append(out)
    _ensure_mpl()
    data = {}
    for gpu in series:
        for vval in vary_vals:
            vals = agg.get((gpu, vval))
            if vals:
                m, sd = mean_sd(vals)
                data[(gpu, vval)] = (m, sd)
    fig, ax = plt.subplots(figsize=(7, 5))
    grouped_bar(ax, series, vary_vals, data, ylabel, plt)
    ax.set_title(f"{title}\ngit {sha}  | fixed: {note_str(filters)}", fontsize=10)
    fig.tight_layout()
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, len(data)


def plot_sampler(cur, outdir, sha):
    filters = {"sweep_group": "decode", "s1": "all", "soft_next": "exact",
               "force_denoise": "off", "skip_last_soft_next": "off",
               "kernel": "hybrid", "grouped_gemm": None, "gpu_layout": None}
    return bar_plot(cur, outdir, "sampler.png", "output t/s: device vs host sampler",
                    filters, "sampler", ["device", "host"], "output t/s", sha)


def plot_s1(cur, outdir, sha):
    filters = {"sweep_group": "decode", "sampler": "device", "soft_next": "exact",
               "force_denoise": "off", "skip_last_soft_next": "off",
               "kernel": "hybrid", "grouped_gemm": None, "gpu_layout": None}
    return bar_plot(cur, outdir, "s1_fusions.png", "output t/s: S1 fusion variants",
                    filters, "s1", S1S, "output t/s", sha)


def plot_soft_next(cur, outdir, sha):
    filters = {"sweep_group": "decode", "sampler": "device", "s1": "all",
               "force_denoise": "off", "skip_last_soft_next": "off",
               "kernel": "hybrid", "grouped_gemm": None, "gpu_layout": None}
    return bar_plot(cur, outdir, "soft_next.png",
                    "output t/s: soft_next modes", filters,
                    "soft_next", SOFT_NEXTS, "output t/s", sha)


def plot_force_denoise(cur, outdir, sha):
    filters = {"sweep_group": "decode", "sampler": "device", "s1": "all",
               "soft_next": "exact", "skip_last_soft_next": "off",
               "kernel": "hybrid", "grouped_gemm": None, "gpu_layout": None}
    return bar_plot(cur, outdir, "force_denoise.png",
                    "output t/s: force_denoise (stop-flag D2H cost)",
                    filters, "force_denoise", FORCES, "output t/s", sha)


def plot_nvfp4_kernels(cur, outdir, sha):
    # focus decode config from the decode group; grouped_gemm/gpu_layout NULL.
    filters = {"sweep_group": "decode", "sampler": "device", "s1": "all",
               "soft_next": "hard", "force_denoise": "on",
               "skip_last_soft_next": "on", "grouped_gemm": None,
               "gpu_layout": None}
    return bar_plot(cur, outdir, "nvfp4_kernels.png",
                    "output t/s: NVFP4 expert kernels", filters,
                    "kernel", KERNELS, "output t/s", sha)


def plot_nvfp4_micro(cur, outdir, sha):
    """grouped_gemm x gpu_layout heatmap per GPU; aggregate over kernel+run_idx."""
    filters = {"sweep_group": "nvfp4"}  # decode axes are fixed in this group
    where_sql, params = where(filters)
    sql = ("SELECT gpu, grouped_gemm, gpu_layout, output_tps FROM runs "
           f"WHERE {where_sql}")
    cur.execute(sql, params)
    agg = {}
    for gpu, gg, gl, out in cur.fetchall():
        agg.setdefault((gpu, gg, gl), []).append(out)
    _ensure_mpl()
    gg_vals = ["off", "xe2"]
    gl_vals = ["off", "on"]
    fig, axes = plt.subplots(1, len(GPUS), figsize=(10, 4.5), squeeze=False)
    n_cells = 0
    for ai, gpu in enumerate(GPUS):
        ax = axes[0][ai]
        mat = []
        for gg in gg_vals:
            row = []
            for gl in gl_vals:
                vals = agg.get((gpu, gg, gl))
                if vals:
                    m, _ = mean_sd(vals)
                    n_cells += 1
                else:
                    m = float("nan")
                row.append(m)
            mat.append(row)
        ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(gl_vals))); ax.set_xticklabels(gl_vals)
        ax.set_yticks(range(len(gg_vals))); ax.set_yticklabels(gg_vals)
        ax.set_xlabel("gpu_layout")
        ax.set_ylabel("grouped_gemm")
        ax.set_title(f"gpu={gpu}")
        for i in range(len(gg_vals)):
            for j in range(len(gl_vals)):
                v = mat[i][j]
                ax.text(j, i, f"{v:.1f}" if v == v else "n/a",
                        ha="center", va="center",
                        color="white" if v == v and v < 50 else "black")
    fig.suptitle(f"NVFP4 micro: output t/s (mean over kernels)\n"
                 f"git {sha}  | fixed: sampler=device s1=all soft_next=hard "
                 f"force=on skip=on", fontsize=10)
    fig.tight_layout()
    path = os.path.join(outdir, "nvfp4_micro.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, n_cells


def plot_summary_table(cur, outdir, sha):
    """Top-10 configs by mean output t/s overall."""
    # A "config" = gpu + all axis columns + kernel (workload fixed: p/n/ds).
    cols = ["gpu", "sampler", "soft_next", "s1", "force_denoise",
            "skip_last_soft_next", "grouped_gemm", "gpu_layout", "kernel"]
    sel = ", ".join(cols)
    cur.execute(f"SELECT {sel}, output_tps FROM runs")
    agg = {}
    for row in cur.fetchall():
        out = row[-1]
        key = row[:-1]
        agg.setdefault(key, []).append(out)
    rows = []
    for key, vals in agg.items():
        m, _ = mean_sd(vals)
        if m is not None:
            rows.append((m, key, len(vals)))
    rows.sort(key=lambda r: r[0], reverse=True)
    top = rows[:10]

    _ensure_mpl()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")
    header = ["rank", "output t/s", "n"] + [SHORT.get(c, c) for c in cols]
    table = [header]
    for i, (m, key, n) in enumerate(top, 1):
        table.append([str(i), f"{m:.1f}", str(n)] + [str(k) for k in key])
    tbl = ax.table(cellText=table[1:], colLabels=table[0], loc="center",
                  cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)
    ax.set_title(f"Top-10 configs by output t/s (overall)\ngit {sha}",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(outdir, "summary_table.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path, len(top)


PLOTS = [
    ("sampler.png", plot_sampler),
    ("s1_fusions.png", plot_s1),
    ("soft_next.png", plot_soft_next),
    ("force_denoise.png", plot_force_denoise),
    ("nvfp4_kernels.png", plot_nvfp4_kernels),
    ("nvfp4_micro.png", plot_nvfp4_micro),
    ("summary_table.png", plot_summary_table),
]


def main(argv=None):
    p = argparse.ArgumentParser(description="Plot sweep results to PNGs.")
    p.add_argument("--db", default=os.path.join(SWEEP_DIR, "results.db"))
    p.add_argument("--out", default=os.path.join(SWEEP_DIR, "plots"))
    args = p.parse_args(argv)

    if not os.path.isfile(args.db):
        print(f"db not found: {args.db}", file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    sha = git_sha(cur)

    cur.execute("SELECT COUNT(*) FROM runs")
    total = cur.fetchone()[0]
    print(f"# db={args.db} rows={total} git={sha} -> {args.out}")

    for fname, fn in PLOTS:
        try:
            path, n = fn(cur, args.out, sha)
            print(f"  wrote {path} ({n} cells)")
        except Exception as e:  # one missing dataset must not kill the rest
            print(f"  SKIP {fname}: {e}", file=sys.stderr)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
