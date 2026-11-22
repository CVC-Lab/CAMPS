"""Write REPRODUCIBILITY.md: a manifest of every archived result file, the
command that regenerates it, the seeds and protocol it used, the environment
it was measured in, and the part of the paper it feeds. Generated from the
JSON files themselves so that it cannot drift from them."""
from __future__ import annotations

import glob
import json
import os
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

# file -> (regenerating command, seed rule, paper location)
META = OrderedDict([
    ("dense_wc.json", ("python -m experiments.run_experiments --exp dense_wc",
                       "dense_conditioned(n, kappa=1e2, seed=100+n)",
                       "Section 5.3, dense well-conditioned table")),
    ("dense_ic.json", ("python -m experiments.run_experiments --exp dense_ic",
                       "dense_conditioned(2048, kappa, seed=200+log10(kappa))",
                       "Section 5.4, ill-conditioned sweep table (rows not involving an LU64 child; "
                       "LU64 rows at kappa <= 1e8)")),
    ("dense_ic_noboost.json", ("python -m experiments.rerun_noboost",
                               "same problems and seeds as dense_ic.json (kappa = 1e10, 1e12)",
                               "Section 5.4, LU64-dependent rows at kappa = 1e10 and 1e12, "
                               "thresholded damped fallback row")),
    ("dense_ic_side_checks.json", ("python -m experiments.side_checks",
                                   "same problems and seeds as dense_ic.json",
                                   "Section 5.4, thresholded damped fallback at kappa <= 1e8 (the file also "
                                   "records pivot-boost counts, 278 at kappa = 1e12, which the "
                                   "paper does not report)")),
    ("sparse_spd.json", ("python -m experiments.run_experiments --exp sparse_spd",
                         "sparse_problem('poisson', g, seed=300+g)",
                         "Section 5.5, Poisson table")),
    ("spd_energy_check.json", ("python -m experiments.energy_check",
                               "sparse_problem('poisson', g, seed=300+g); counterexample is exact",
                               "Section 4.3 counterexample; Appendix A.10 energy-contract margins and cost")),
    ("sparse_nsy.json", ("python -m experiments.run_experiments --exp sparse_nsy",
                         "sparse_problem('convdiff', g, w, seed=400+g)",
                         "Section 5.6, convection-diffusion table and wind sweep")),
    ("placement.json", ("python -m experiments.run_experiments --exp placement",
                        "dense_conditioned(2048, kappa, seed=500)",
                        "Section 5.7, edge-placement table")),
    ("f3r.json", ("python -m experiments.run_experiments --exp f3r",
                  "sparse_problem(kind, g, w, seed=600+g); tuning grids F3R_GRID, HYBRID_GRID, "
                  "HIER_GRID, IRTOP_GRID in run_experiments.py; selected configurations stored "
                  "under 'tuned_configs'",
                  "Section 5.8, F3R comparison table")),
    ("multirhs.json", ("python -m experiments.run_experiments --exp multirhs",
                       "dense_conditioned(4096, kappa=1e2, seed=700); right-hand-side block from "
                       "torch.Generator seed 701",
                       "Section 5.3, multi-RHS amortization table")),
    ("verify_recursion.json", ("python -m experiments.verify_recursion",
                               "dense_conditioned(512, kappa, seed=300+10*log10(kappa)+{0,1})",
                               "Section 5.9, recursive-bound verification table")),
])


def fmt_env(env):
    if not env:
        return "n/a"
    flags = env.get("backend_flags", {})
    return (f"{env.get('gpu')}; driver {env.get('driver')}; CUDA {env.get('cuda')}; "
            f"torch {env.get('torch')}; Python {env.get('python')}; {env.get('platform')}; "
            f"TF32 matmul {flags.get('cuda.matmul.allow_tf32')}, "
            f"fp16 reduced-precision reductions {flags.get('cuda.matmul.allow_fp16_reduced_precision_reduction')}, "
            f"float32 matmul precision '{flags.get('float32_matmul_precision')}'")


def protocol(recs):
    keys = ("warmup", "reps", "setup_reps", "tol")
    seen = sorted({tuple(r.get(k) for k in keys) for r in recs if all(k in r for k in keys)})
    return "; ".join(f"warmup={w}, timed reps={n}, setup reps={s}, tau={t:g}" for w, n, s, t in seen) or "n/a"


def main():
    out = []
    out.append("# Reproducibility manifest\n")
    out.append("Every number in the paper comes from a JSON file in `results/`. This file lists, "
               "for each of them, the command that regenerates it, the seeds it used, the "
               "measurement protocol, the environment recorded at run time, and where it is used "
               "in the paper. Timings will differ on other hardware; iteration counts, acceptance "
               "decisions, and residuals are deterministic for the recorded seeds and were "
               "reproduced bit for bit on reruns (see `experiments/regress_fallback.py` and the "
               "comparison in the verify_recursion log).\n")
    out.append("Install with `pip install -r requirements.txt` (CUDA build of torch required). "
               "`python -m experiments.run_experiments --exp all` regenerates the seven sweep "
               "files; the remaining files have their own commands below. "
               "`python -m experiments.analyze` rebuilds `docs/tables/` and `docs/figures/`.\n")
    for fn, (cmd, seeds, paper) in META.items():
        path = os.path.join(RES, fn)
        if not os.path.exists(path):
            out.append(f"## `{fn}`\n\nMISSING\n")
            continue
        d = json.load(open(path))
        recs = d.get("records", [])
        env = d.get("environment") or {}
        if not env and "gpu" in d:
            env = {"gpu": d.get("gpu"), "torch": d.get("torch"), "backend_flags": d.get("backend", {})}
        out.append(f"## `{fn}`\n")
        out.append(f"- **Regenerate:** `{cmd}`")
        out.append(f"- **Seeds:** {seeds}")
        out.append(f"- **Used in:** {paper}")
        if recs:
            probs = sorted({r.get("problem") for r in recs})
            methods = sorted({r.get("method") for r in recs})
            out.append(f"- **Records:** {len(recs)} ({len(probs)} problems x methods)")
            out.append(f"- **Protocol:** {protocol(recs)}")
            out.append("- **Problems:** " + ", ".join(f"`{p}`" for p in probs))
            out.append("- **Methods:** " + ", ".join(f"`{m}`" for m in methods))
        else:
            out.append("- **Contents:** " + ", ".join(f"`{k}`" for k in d.keys() if k not in ("environment", "backend", "torch", "gpu", "experiment")))
        if d.get("note"):
            out.append(f"- **Note:** {d['note']}")
        if d.get("tuned_configs"):
            out.append("- **Selected configurations (fair tuning, fastest converged):** `" + json.dumps(d["tuned_configs"]) + "`")
        out.append(f"- **Environment:** {fmt_env(env)}")
        out.append("")
    out.append("## Tuning grids (f3r.json)\n")
    out.append("From `experiments/run_experiments.py`: `F3R_GRID` = m2 in {10,20,30} x m3 in {5,10} x "
               "sweeps in {2,4}; `HYBRID_GRID` = {(m3=5, mid_maxiter=15), (5,10), (8,10)}; "
               "`IRTOP_GRID` = m2 in {10,15,20} x m3 in {5,8}, sweeps=2; `HIER_GRID` = four "
               "configurations listed in the script. Each candidate is run once (1 warmup, 2 "
               "timed) and the fastest converged configuration is kept, identically for every "
               "method.\n")
    out.append("## Fixed quantities\n")
    out.append("- Stopping rule everywhere: FP64-recomputed relative residual <= 1e-10, verified "
               "after the solve by an independent FP64 residual.")
    out.append("- Backend flags (`camps/diagnostics.py`): TF32 disabled for matmul and cuDNN, "
               "reduced-precision fp16/bf16 reductions disabled, float32 matmul precision "
               "'highest', so FP32 means IEEE binary32 arithmetic.")
    out.append("- Dense LU nodes: partial pivoting, no pivot modification (`lu_pivot_tol=0`); "
               "the zero-pivot boost (1e-8 threshold, 1e-6 replacement, relative to the largest "
               "pivot) is retained only for ILU(0).")
    out.append("- Adapter defaults: power-of-two right-hand-side scaling, residual contract with "
               "slack of a few parent ulps, stagnation window 3 for plain-IR instances, "
               "attempt budgets as given per instance in `experiments/instances.py`.")
    with open(os.path.join(ROOT, "REPRODUCIBILITY.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote REPRODUCIBILITY.md")


if __name__ == "__main__":
    main()
