"""Tables and figures for the 2026-10-19 overnight experiments (random
minefield, budget 2, 3 seeds) and the simulator benchmark.

    python scripts/analysis/overnight_report.py --root logs/vast --out paper/results

Reads, under --root (what `vast.sh pull` brings home into logs/vast):
  eval/<run>.json         scripts/eval_design.py --json, one per trained run
  queue/<run>.log         training logs (eval lines "step=.. k=v ..", design
                          lines "design it= ...")
  bench/results.jsonl     scripts/bench/*.py
Writes tables (markdown + LaTeX) and figures (PDF + PNG) to --out.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BUDGET = 2.0
CONDITIONS = [  # (key prefix, label)
    ("nom", "Nominal body"),
    ("cd_speed", "Co-design, speed ranking"),
    ("cd_ff", "Co-design, feasibility first"),
    ("fx_cd_speed", "Speed-ranked body, retrained"),
    ("fx_cd_ff", "Feasibility-first body, retrained"),
    ("cd_unc", "Unconstrained co-design"),
    ("fx_cd_unc", "Unconstrained body + constraint"),
]
COLORS = dict(zip([c for c, _ in CONDITIONS],
                  ["#555555", "#1f77b4", "#d62728", "#9ecae1", "#ff9896", "#2ca02c", "#98df8a"]))
METRICS = [  # (eval key, column, scale, decimals)
    ("reward", "Return", 1, 2),
    ("speed", "Speed (m/s)", 1, 2),
    ("cost", "Cost", 1, 2),
    ("hazard_steps", "Hazard steps", 1, 1),
    ("flipped", "Flipped (%)", 100, 1),
    ("out_of_bounds", "OOB (%)", 100, 1),
    ("hazard_free", "Hazard-free (%)", 100, 1),
    ("safe", "Under budget (%)", 100, 0),
]
GENES = ["hip len", "thigh len", "shin len", "hip rad", "thigh rad", "shin rad", "torso rad"]


def runs_of(prefix, evals):
    pat = re.compile(rf"^{prefix}_s\d+$")
    return sorted(k for k in evals if pat.match(k))


def mean_se(xs):
    xs = np.asarray(xs, dtype=float)
    if xs.size == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(xs.std(ddof=1) / np.sqrt(xs.size)) if xs.size > 1 else float("nan")


def parse_log(path):
    evals, designs = [], []
    if not path.exists():
        return evals, designs
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("step="):
            d = {}
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        d[k] = float(v)
                    except ValueError:
                        pass
            if "eval/episode_speed" in d:
                evals.append(d)
        elif line.startswith("design it="):
            m = re.search(r"step=\s*([\d,]+)", line)
            row = {"step": float(m.group(1).replace(",", ""))}
            for k in ("score", "comp", "arrive", "cost", "feas"):
                mm = re.search(rf"\b{k}=\s*([-+\d.]+)", line)
                if mm:
                    row[k] = float(mm.group(1))
            mm = re.search(r"mode=\[([^\]]*)\]", line)
            if mm:
                row["mode"] = [float(x) for x in mm.group(1).split()]
            designs.append(row)
    return evals, designs


def fmt(m, s, dec):
    if np.isnan(m):
        return "--"
    return f"{m:.{dec}f}" + (f" ± {s:.{dec}f}" if not np.isnan(s) else "")


def main_table(evals, out):
    rows_md = ["| Condition | n | " + " | ".join(c for _, c, _, _ in METRICS) + " |",
               "|---|---|" + "---|" * len(METRICS)]
    rows_tex = []
    for prefix, label in CONDITIONS:
        rs = runs_of(prefix, evals)
        if not rs:
            continue
        cells = []
        for key, _, scale, dec in METRICS:
            m, s = mean_se([evals[r]["pooled"][key]["mean"] * scale for r in rs])
            cells.append(fmt(m, s, dec))
        rows_md.append(f"| {label} | {len(rs)} | " + " | ".join(cells) + " |")
        rows_tex.append(f"{label} & {len(rs)} & " + " & ".join(c.replace('±', r'$\pm$') for c in cells) + r" \\")
    (out / "main_table.md").write_text(
        f"Random minefield, budget {BUDGET:g}. Mean ± SE over training seeds; each run evaluated on "
        "3 x 512 stochastic episodes (scripts/eval_design.py). Cost and under-budget are per environment.\n\n"
        + "\n".join(rows_md) + "\n")
    (out / "main_table.tex").write_text(
        "\\begin{tabular}{l" + "c" * (len(METRICS) + 1) + "}\n\\toprule\nCondition & $n$ & "
        + " & ".join(c.replace("%", r"\%") for _, c, _, _ in METRICS) + r" \\" + "\n\\midrule\n"
        + "\n".join(rows_tex) + "\n\\bottomrule\n\\end{tabular}\n")
    print("\n".join(rows_md))


def per_run_table(evals, out):
    lines = ["| Run | Body | Mass (kg) | Return | Speed | Cost | Hazard steps | Flipped (%) | OOB (%) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for prefix, _ in CONDITIONS:
        for r in runs_of(prefix, evals):
            p = evals[r]["pooled"]
            lines.append(
                f"| {r} | {evals[r]['body'][:60]} | {evals[r].get('mass_kg', float('nan')):.1f} | "
                f"{p['reward']['mean']:.2f} | {p['speed']['mean']:.2f} | {p['cost']['mean']:.2f} ± {p['cost']['se']:.2f} | "
                f"{p['hazard_steps']['mean']:.1f} | {100 * p['flipped']['mean']:.1f} | {100 * p['out_of_bounds']['mean']:.1f} |")
    (out / "per_run_table.md").write_text("\n".join(lines) + "\n")


def scatter(evals, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for prefix, label in CONDITIONS:
        rs = runs_of(prefix, evals)
        if not rs:
            continue
        c = np.array([evals[r]["pooled"]["cost"]["mean"] for r in rs])
        v = np.array([evals[r]["pooled"]["speed"]["mean"] for r in rs])
        ax.scatter(c, v, s=18, color=COLORS[prefix], alpha=0.55)
        cm, cs = mean_se(c)
        vm, vs = mean_se(v)
        ax.errorbar(cm, vm, xerr=0 if np.isnan(cs) else cs, yerr=0 if np.isnan(vs) else vs, fmt="o",
                    ms=7, color=COLORS[prefix], capsize=3, label=label, mec="k", mew=0.6)
    ax.set_xscale("symlog", linthresh=1.0)
    allc = [evals[r]["pooled"]["cost"]["mean"] for p, _ in CONDITIONS for r in runs_of(p, evals)]
    ax.set_xlim(0, max(allc + [BUDGET]) * 1.6)
    ax.axvline(BUDGET, ls="--", color="k", lw=0.8)
    ax.text(BUDGET, ax.get_ylim()[1], " budget", va="top", fontsize=8)
    ax.set_xlabel("Episode cost (symlog)")
    ax.set_ylabel("Speed (m/s)")
    ax.legend(fontsize=7, loc="best", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"speed_vs_cost.{ext}", dpi=200)
    plt.close(fig)


def design_curves(logs, out):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    for prefix, label in CONDITIONS:
        if not prefix.startswith("cd_"):
            continue
        for i, r in enumerate(sorted(k for k in logs if re.match(rf"^{prefix}_s\d+$", k))):
            d = logs[r][1]
            if not d:
                continue
            st = np.array([x["step"] for x in d]) / 1e9
            kw = dict(color=COLORS[prefix], lw=1, alpha=0.8, label=label if i == 0 else None)
            axes[0].plot(st, [x.get("score", np.nan) * 1 for x in d], **kw)
            axes[1].plot(st, [x.get("cost", np.nan) for x in d], **kw)
            axes[2].plot(st, [x.get("arrive", np.nan) for x in d], **kw)
    axes[0].set_ylabel("Design score (m/decision)")
    axes[1].set_ylabel("Episode cost, sampled designs")
    axes[1].axhline(BUDGET, ls="--", color="k", lw=0.8)
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[2].set_ylabel("Arrival rate")
    for a in axes:
        a.set_xlabel("Environment steps (B)")
        a.grid(alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"codesign_curves.{ext}", dpi=200)
    plt.close(fig)


def gene_trajectories(logs, out):
    runs = [k for k in sorted(logs) if re.match(r"^cd_(speed|ff|unc)_s\d+$", k) and logs[k][1]]
    if not runs:
        return
    fig, axes = plt.subplots(1, 7, figsize=(14, 2.4), sharey=True)
    for r in runs:
        prefix = r.rsplit("_s", 1)[0]
        d = [x for x in logs[r][1] if "mode" in x]
        st = np.array([x["step"] for x in d]) / 1e9
        modes = np.array([x["mode"] for x in d])
        scale = 1.0 + 0.4 * np.clip(modes, -1, 1)  # design -> scale factor on the nominal ant
        for g in range(min(7, scale.shape[1])):
            axes[g].plot(st, scale[:, g], color=COLORS[prefix], lw=0.9, alpha=0.8)
    for g, a in enumerate(axes):
        a.set_title(GENES[g], fontsize=8)
        a.axhline(1.0, color="k", lw=0.5, ls=":")
        a.set_ylim(0.55, 1.45)
        a.set_xlabel("steps (B)", fontsize=7)
        a.grid(alpha=0.3)
    axes[0].set_ylabel("scale (mode)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"gene_trajectories.{ext}", dpi=200)
    plt.close(fig)


def single_body_curves(logs, out):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    for prefix, label in CONDITIONS:
        if prefix.startswith("cd_"):
            continue
        for i, r in enumerate(sorted(k for k in logs if re.match(rf"^{prefix}_s\d+$", k))):
            e = logs[r][0]
            if not e:
                continue
            st = np.array([x["step"] for x in e]) / 1e6
            kw = dict(color=COLORS[prefix], lw=1, alpha=0.8, label=label if i == 0 else None)
            axes[0].plot(st, [x["eval/episode_speed"] for x in e], **kw)
            axes[1].plot(st, [x.get("eval/episode_cost", np.nan) for x in e], **kw)
    axes[0].set_ylabel("Eval speed (m/s)")
    axes[1].set_ylabel("Eval episode cost")
    axes[1].axhline(BUDGET, ls="--", color="k", lw=0.8)
    axes[1].set_yscale("symlog", linthresh=1.0)
    for a in axes:
        a.set_xlabel("Environment steps (M)")
        a.grid(alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"single_body_curves.{ext}", dpi=200)
    plt.close(fig)


def gene_table(evals, out):
    lines = ["| Run | " + " | ".join(GENES) + " | Mass (kg) |", "|---|" + "---|" * (len(GENES) + 1)]
    for prefix, _ in CONDITIONS:
        if not prefix.startswith("cd_"):
            continue
        for r in runs_of(prefix, evals):
            m = re.search(r"mode \[([^\]]*)\]", evals[r]["body"])
            if not m:
                continue
            mode = np.array([float(x) for x in m.group(1).split()])
            scale = 1.0 + 0.4 * np.clip(mode, -1, 1)
            lines.append(f"| {r} | " + " | ".join(f"{s:.2f}" for s in scale) + f" | {evals[r].get('mass_kg', float('nan')):.1f} |")
    (out / "evolved_bodies.md").write_text("Scale factors on the nominal Safety Gym ant (1.00 = nominal; box 0.6-1.4).\n\n"
                                           + "\n".join(lines) + "\n")


def bench(root, out):
    path = root / "bench" / "results.jsonl"
    if not path.exists():
        return
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ok = [r for r in recs if "steps_per_s" in r]
    lines = ["| System | Task | Variant | Envs | Env steps/s | Physics steps/s | Physics (dt x substeps, solver) |",
             "|---|---|---|---|---|---|---|"]
    best = {}
    for r in ok:
        key = (r["system"], r["impl"], r["task"], r["variant"])
        if key not in best or r["steps_per_s"] > best[key]["steps_per_s"]:
            best[key] = r
    for key in sorted(best, key=lambda k: (k[2], -best[k]["steps_per_s"])):
        r = best[key]
        sysname = r["system"] + ("" if r["impl"] in ("cpu",) else f" ({'MuJoCo Warp' if r['impl'] == 'warp' else 'MJX-JAX'})")
        lines.append(f"| {sysname} | {r['task']} | {r['variant']} | {r['n']} | {r['steps_per_s']:,.0f} | "
                     f"{r.get('physics_steps_per_s', float('nan')):,.0f} | {r.get('physics_dt')} x {r.get('substeps')}, {r.get('solver')} |")
    errs = [r for r in recs if "error" in r]
    (out / "bench_table.md").write_text(
        "Best throughput per configuration (random actions). One env step = 0.02 s simulated in every row "
        "except Safety-Gymnasium 'native' ant (0.1 s).\n\n" + "\n".join(lines) + "\n"
        + ("\nFailed configurations:\n" + "\n".join(f"- {r['system']} {r['task']} {r['variant']} n={r.get('n')}: {r['error'][:120]}" for r in errs) if errs else ""))
    print("\n".join(lines))
    for task in ("pointgoal2", "antgoal2", "minefield"):
        rows = [r for r in ok if r["task"] == task]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        groups = {}
        for r in rows:
            groups.setdefault((r["system"], r["impl"], r["variant"]), []).append(r)
        for (sysn, impl, var), rs in sorted(groups.items()):
            rs = sorted(rs, key=lambda x: x["n"])
            lab = f"{sysn} [{'Warp' if impl == 'warp' else 'MJX-JAX' if impl == 'jax' else 'CPU'}] {var}"
            if impl == "cpu":
                for x in rs:
                    ax.scatter([x["n"]], [x["steps_per_s"]], marker="s", s=30, label=f"{lab} ({x['n']} procs)")
            else:
                ax.plot([x["n"] for x in rs], [x["steps_per_s"] for x in rs], marker="o", ms=3, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Parallel environments")
        ax.set_ylabel("Environment steps / s")
        ax.set_title(task, fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=5.5, frameon=False, loc="lower right")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out / f"bench_{task}.{ext}", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("logs/vast"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("paper/results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evals = {p.stem: json.loads(p.read_text()) for p in sorted((args.root / "eval").glob("*.json"))}
    logs = {p.stem: parse_log(p) for p in sorted((args.root / "queue").glob("*.log"))
            if not p.stem.startswith(("eval_", "bench", "smoke"))}
    if evals:
        main_table(evals, args.out)
        per_run_table(evals, args.out)
        gene_table(evals, args.out)
        scatter(evals, args.out)
    if logs:
        design_curves(logs, args.out)
        gene_trajectories(logs, args.out)
        single_body_curves(logs, args.out)
    bench(args.root, args.out)
    print(f"wrote {sorted(p.name for p in args.out.iterdir())}")


if __name__ == "__main__":
    main()
