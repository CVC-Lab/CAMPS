"""Turn results/*.json into markdown + LaTeX tables and figures.

Usage: python -m experiments.analyze
Writes docs/results_tables.md, docs/tables/*.tex, docs/figures/*.png.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DOCS = os.path.join(ROOT, "docs")


def load(name: str) -> Dict[str, Any]:
    with open(os.path.join(RESULTS, f"{name}.json")) as f:
        return json.load(f)


def fmt_ms(q: Dict[str, float]) -> str:
    return f"{q['median']:.1f} [{q['q1']:.1f}, {q['q3']:.1f}]"


def fmt_e(x: float) -> str:
    if x is None:
        return "-"
    s = f"{x:.2e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")


def short_status(rec: Dict) -> str:
    if rec.get("status") == "exception":
        return "error"
    return "converged" if rec.get("converged") else f"DNC ({rec.get('status')})"


def total_matvecs(rec: Dict) -> str:
    mv = rec.get("matvecs", {})
    parts = [f"{k}:{v['matvec']}" for k, v in sorted(mv.items()) if v.get("matvec")]
    return " ".join(parts) if parts else "-"


def mem_str(rec: Dict) -> str:
    m = rec.get("memory", {})
    if not m:
        return "-"
    return f"{m['analytic_total_MiB']:.1f} / {m['measured_solve_peak_MiB']:.1f}"


# ---------------------------------------------------------------- markdown

def md_table(records: List[Dict], caption: str) -> str:
    hdr = ("| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | "
           "Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |\n"
           "|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for r in records:
        if r.get("status") == "exception":
            rows.append(f"| {r['method']} | - | - | - | - | error | - | - | - | - |")
            continue
        rows.append(
            f"| {r['method']} | {fmt_ms(r['setup_ms'])} | {fmt_ms(r['solve_ms'])} | "
            f"{r['total_ms_median']:.1f} | {r['outer_iterations']} | {short_status(r)} | "
            f"{fmt_e(r['rel_residual'])} | {fmt_e(r['forward_error'])} | "
            f"{mem_str(r)} | {total_matvecs(r)} |")
    return f"**{caption}**\n\n{hdr}" + "\n".join(rows) + "\n\n"


# ---------------------------------------------------------------- latex

def tex_escape(s: str) -> str:
    return (s.replace("<-", r"$\leftarrow$").replace("_", r"\_")
             .replace("%", r"\%"))


def tex_method(name: str) -> str:
    """Paper-style solver names: FGMRES64[IR64<-32[LU32]] ->
    $\\mathrm{FGMRES}_{64}[\\mathrm{IR}_{64\\leftarrow 32}[\\mathrm{LU}_{32}]]$."""
    import re
    s = name
    s = re.sub(r"IR(\d+|b16)<-(\d+|b16)", r"IRSUB{\1}{\2}", s)
    s = re.sub(r"([A-Za-z]+)(64|32|b16|16)", r"\\mathrm{\1}_{\2}", s)
    s = s.replace("IRSUB{", r"\mathrm{IR}_{")
    s = re.sub(r"\\mathrm\{IR\}_\{(\w+)\}\{(\w+)\}", r"\\mathrm{IR}_{\1\\leftarrow \2}", s)
    s = s.replace(" (direct)", r"\ \text{(direct)}")
    s = s.replace(" (plain IR)", r"\ \text{(plain IR)}")
    return f"${s}$"


def tex_table(records: List[Dict], caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{lrrrrlrr}", r"\hline",
        r"Method & Setup & Solve & Total & Outer & Status & $\|b-Ax\|_2/\|b\|_2$ & Mem (MiB) \\",
        r"\hline",
    ]
    for r in records:
        if r.get("status") == "exception":
            continue
        status = "converged" if r.get("converged") else f"DNC ({r.get('status')})"
        mem = r.get("memory", {}).get("analytic_total_MiB", 0.0)
        lines.append(
            f"{tex_method(r['method'])} & {r['setup_ms']['median']:.1f} & "
            f"{r['solve_ms']['median']:.1f} & {r['total_ms_median']:.1f} & "
            f"{r['outer_iterations']} & {status} & "
            f"{fmt_e(r['rel_residual'])} & {mem:.1f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- figures

def convergence_figure(records: List[Dict], title: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for r in records:
        h = r.get("outer_residual_history")
        if not h:
            continue
        h0 = h[0] if h[0] > 0 else 1.0
        ax.semilogy([v / h0 for v in h], label=r["method"], linewidth=1.4)
    ax.set_xlabel("outer solver step (restart-cycle boundaries included)")
    ax.set_ylabel(r"$\|r\|_2 / \|r_0\|_2$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def group_by_problem(records: List[Dict]) -> "OrderedDict[str, List[Dict]]":
    out: "OrderedDict[str, List[Dict]]" = OrderedDict()
    for r in records:
        key = r["problem"]
        if r.get("n_rhs", 1) != 1:
            key = f"{key}-nrhs{r['n_rhs']}"
        out.setdefault(key, []).append(r)
    return out


def robustness_matrix_md(records: List[Dict]) -> str:
    """dense_ic: rows = method, cols = kappa, cell = status/outer/fwd err."""
    kappas = sorted({r["problem_meta"]["kappa"] for r in records
                     if "problem_meta" in r and "kappa" in r.get("problem_meta", {})})
    methods = list(OrderedDict.fromkeys(r["method"] for r in records))
    hdr = "| Method | " + " | ".join(f"kappa={k:.0e}" for k in kappas) + " |\n"
    hdr += "|---|" + "---|" * len(kappas) + "\n"
    rows = []
    for m in methods:
        cells = []
        for k in kappas:
            rec = next((r for r in records if r["method"] == m
                        and r.get("problem_meta", {}).get("kappa") == k), None)
            if rec is None or rec.get("status") == "exception":
                cells.append("-")
            elif rec["converged"]:
                cells.append(f"ok ({rec['outer_iterations']} it, fwd {fmt_e(rec['forward_error'])})")
            else:
                cells.append(f"DNC ({rec['status']})")
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    return "**Robustness across condition numbers (n=2048 dense)**\n\n" + hdr + "\n".join(rows) + "\n\n"


def main() -> None:
    os.makedirs(os.path.join(DOCS, "tables"), exist_ok=True)
    os.makedirs(os.path.join(DOCS, "figures"), exist_ok=True)
    md: List[str] = ["# CAMPS matrix-class sweep - results\n"]

    env_done = False
    for exp in ("dense_wc", "dense_ic", "sparse_spd", "sparse_nsy", "placement"):
        try:
            data = load(exp)
        except FileNotFoundError:
            continue
        if not env_done:
            env = data.get("environment", {})
            md.append("## Environment\n\n```\n" + json.dumps(env, indent=1) + "\n```\n")
            env_done = True
        md.append(f"## {exp}\n")
        groups = group_by_problem(data["records"])
        if exp == "dense_ic":
            md.append(robustness_matrix_md(data["records"]))
        for prob, recs in groups.items():
            md.append(md_table(recs, prob))
            tex = tex_table(recs, f"{exp}: {prob}. Median over "
                            f"{recs[0].get('reps', '?')} timed runs after "
                            f"{recs[0].get('warmup', '?')} warmups; times in ms.",
                            f"tab:{exp}-{prob}")
            with open(os.path.join(DOCS, "tables", f"{exp}_{prob}.tex"), "w") as f:
                f.write(tex)
        # figures: one convergence plot per problem where histories exist
        for prob, recs in groups.items():
            if any(r.get("outer_residual_history") for r in recs):
                convergence_figure(
                    recs, prob,
                    os.path.join(DOCS, "figures", f"{exp}_{prob}.png"))

    with open(os.path.join(DOCS, "results_tables.md"), "w") as f:
        f.write("\n".join(md))
    print(f"wrote {os.path.join(DOCS, 'results_tables.md')}")


if __name__ == "__main__":
    main()
