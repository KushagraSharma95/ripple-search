"""Render result plots for the README from results/*.json."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
P = R / "plots"
P.mkdir(exist_ok=True)

C = {"ripple": "#d62728", "ripple_quality": "#d62728", "ripple_budget": "#ff9896",
     "hnsw": "#1f77b4", "brute_force": "#7f7f7f",
     "vp_tree": "#2ca02c", "adaptive": "#d62728", "frozen": "#7f7f7f"}


def load(name):
    f = R / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


# --- exp1: scaling in N ---
d = load("exp1_scaling_n")
if d:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ns = [r["n"] for r in d["rows"]]
    for eng in ["brute_force", "ripple_budget", "ripple_quality", "hnsw", "vp_tree"]:
        xs = [r["n"] for r in d["rows"] if eng in r["engines"]]
        if not xs:
            continue
        p50 = [r["engines"][eng]["p50_ms"] for r in d["rows"] if eng in r["engines"]]
        rec = [r["engines"][eng]["recall@10"] for r in d["rows"] if eng in r["engines"]]
        axes[0].plot(xs, p50, "o-", label=eng, color=C.get(eng))
        axes[1].plot(xs, rec, "o-", label=eng, color=C.get(eng))
    axes[0].set(xlabel="N documents", ylabel="query p50 (ms)", xscale="log",
                yscale="log", title="Latency vs N (single thread)")
    axes[1].set(xlabel="N documents", ylabel="recall@10", xscale="log",
                ylim=(0, 1.05), title="Quality vs N")
    for eng in ["ripple_budget", "ripple_quality"]:
        work = [r["engines"][eng]["visited_fraction_of_n"] for r in d["rows"]]
        axes[2].plot(ns, work, "o-", color=C[eng], label=f"{eng} visited/N")
    vx = [r["n"] for r in d["rows"] if "vp_tree" in r["engines"]]
    vw = [r["engines"]["vp_tree"]["dist_fraction_of_n"] for r in d["rows"] if "vp_tree" in r["engines"]]
    axes[2].plot(vx, vw, "s-", color=C["vp_tree"], label="vp-tree dists/N")
    axes[2].set(xlabel="N documents", ylabel="fraction of corpus touched",
                xscale="log", title="Work per query vs N")
    for a in axes:
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(P / "exp1_scaling_n.png", dpi=120)

# --- exp2: dimensionality ---
d = load("exp2_dimensionality")
if d:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for regime, ls in [("clustered", "-"), ("iid_gaussian", "--")]:
        rows = [r for r in d["rows"] if r["regime"] == regime]
        ds = [r["d"] for r in rows]
        axes[0].plot(ds, [r["vp_tree_dist_fraction"] for r in rows], "o" + ls,
                     color=C["vp_tree"], label=f"vp-tree {regime}")
        axes[1].plot(ds, [r["relative_contrast"] for r in rows], "o" + ls,
                     color="#9467bd", label=f"contrast {regime}")
        axes[2].plot(ds, [r["ripple_recall@10"] for r in rows], "o" + ls,
                     color=C["ripple"], label=f"ripple {regime}")
        axes[2].plot(ds, [r["hnsw_recall@10"] for r in rows], "^" + ls,
                     color=C["hnsw"], label=f"hnsw {regime}")
    axes[0].axhline(1.0, color="k", lw=0.5)
    axes[0].set(xlabel="dimensions", ylabel="distances computed / N", xscale="log",
                title="Triangle-inequality pruning vs D\n(claim: should FALL)")
    axes[1].set(xlabel="dimensions", ylabel="(dmax-dmin)/dmin", xscale="log",
                yscale="log", title="Distance concentration vs D")
    axes[2].set(xlabel="dimensions", ylabel="recall@10", xscale="log",
                ylim=(0, 1.05), title="Accuracy vs D (claim: should RISE)")
    for a in axes:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(P / "exp2_dimensionality.png", dpi=120)

# --- exp3: adaptation ---
d = load("exp3_hebbian_adaptation")
if d:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, wl in enumerate(d["workloads"]):
        wd = d["workloads"][wl]
        for name in ["adaptive", "frozen"]:
            ws = wd["windows"][name]
            xs = [w["window_start"] for w in ws]
            axes[0][col].plot(xs, [w["mean_visited"] for w in ws], "-",
                              color=C[name], label=name)
            axes[1][col].plot(xs, [w["recall@10"] for w in ws], "-",
                              color=C[name], label=name)
        axes[0][col].set(title=f"{wl}: work over time", xlabel="query #",
                         ylabel="mean nodes visited")
        axes[1][col].set(title=f"{wl}: recall over time", xlabel="query #",
                         ylabel="recall@10", ylim=(0, 1.05))
        for row in range(2):
            axes[row][col].legend()
            axes[row][col].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(P / "exp3_hebbian.png", dpi=120)

# --- exp4: caches ---
d = load("exp4_thermo_cache")
if d:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    policies = ["lru", "lfu", "lru_prefetch", "thermo_exact", "thermo_prefetch"]
    colors = ["#7f7f7f", "#bcbd22", "#1f77b4", "#ff7f0e", "#d62728"]
    for ax, wl in zip(axes, d["workloads"]):
        rows = d["workloads"][wl]
        caps = [r["capacity"] for r in rows]
        for pol, c in zip(policies, colors):
            ax.plot(caps, [r[pol] for r in rows], "o-", color=c, label=pol)
        ax.set(title=f"workload: {wl}", xlabel="cache capacity", xscale="log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("hit rate on needed top-10")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(P / "exp4_cache.png", dpi=120)

print("plots ->", P)
