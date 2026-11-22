"""Run the CAMPS matrix-class experiment sweep.

Usage:
    python -m experiments.run_experiments --exp dense_wc
    python -m experiments.run_experiments --exp all [--pilot]

--pilot shrinks sizes/repetitions for a fast smoke test of every code path.
"""
from __future__ import annotations

import argparse
import os
import time
import traceback
from typing import Any, Callable, Dict, List

import torch

from . import instances as I
from .harness import run_method, save_results
from .matrices import dense_conditioned, sparse_problem

DEV = "cuda"
TOL = 1e-10
from experiments.paths import results_dir
RESULTS = results_dir()


def _run(records: List[Dict], problem, build_fn: Callable, tol: float,
         warmup: int, reps: int, **kw) -> None:
    label = kw.pop("label", build_fn.__name__)
    t0 = time.perf_counter()
    try:
        rec = run_method(problem, build_fn, tol=tol, warmup=warmup, reps=reps, **kw)
        rec["harness_wall_s"] = time.perf_counter() - t0
        records.append(rec)
        print(f"  [{problem.name}] {rec['method']}: "
              f"{'ok' if rec['converged'] else 'DNC'} "
              f"status={rec['status']} outer={rec['outer_iterations']} "
              f"setup={rec['setup_ms']['median']:.1f}ms "
              f"solve={rec['solve_ms']['median']:.1f}ms "
              f"rr={rec['rel_residual']:.2e}", flush=True)
    except Exception:
        print(f"  [{problem.name}] {label}: EXCEPTION", flush=True)
        traceback.print_exc()
        records.append({
            "method": label, "problem": problem.name, "n": problem.n,
            "tol": tol, "status": "exception", "converged": False,
            "error": traceback.format_exc(limit=5),
        })


# --------------------------------------------------------------------------

def exp_dense_wc(pilot: bool) -> None:
    """Dense well-conditioned class: size sweep + full recursive hierarchy."""
    records: List[Dict] = []
    sizes = [1024] if pilot else [2048, 4096, 8192]
    warmup, reps = (1, 3) if pilot else (3, 15)
    for n in sizes:
        prob = dense_conditioned(n, kappa=1e2, device=DEV, seed=100 + n)
        common = dict(tol=TOL, warmup=warmup, reps=reps, restart=20, maxiter=300)
        _run(records, prob, I.build_lu64_direct, tol=TOL, warmup=warmup, reps=reps)
        _run(records, prob, I.build_fgmres64_lu64, **common)
        _run(records, prob, I.build_plain_ir_lu32, tol=TOL, warmup=warmup, reps=reps)
        _run(records, prob, I.build_camps_lu32, **common)
        if n <= 4096:  # inverse-based hierarchy children
            _run(records, prob, I.build_camps_hierarchy_dense, **common,
                 half_dtype=torch.float16)
            _run(records, prob, I.build_camps_hierarchy_dense, **common,
                 half_dtype=torch.bfloat16)
    save_results(os.path.join(RESULTS, "dense_wc.json"), "dense_wc", records)


def exp_dense_ic(pilot: bool) -> None:
    """Dense ill-conditioned class: condition-number sweep."""
    records: List[Dict] = []
    n = 1024 if pilot else 2048
    kappas = [1e4, 1e10] if pilot else [1e4, 1e6, 1e7, 1e8, 1e10, 1e12]
    warmup, reps = (1, 3) if pilot else (3, 15)
    for kappa in kappas:
        prob = dense_conditioned(n, kappa=kappa, device=DEV, seed=int(200 + torch.log10(torch.tensor(kappa)).item()))
        common = dict(tol=TOL, warmup=warmup, reps=reps, restart=30, maxiter=400)
        _run(records, prob, I.build_lu64_direct, tol=TOL, warmup=warmup, reps=reps)
        _run(records, prob, I.build_fgmres64_lu64, **common)
        _run(records, prob, I.build_plain_ir_lu32, tol=TOL, warmup=warmup, reps=reps)
        _run(records, prob, I.build_camps_lu32, **common)
        _run(records, prob, I.build_camps_lu32, **common, loose_contract=True,
             label="camps_lu32_loose")
        _run(records, prob, I.build_plain_ir_lu32_fallback, tol=TOL,
             warmup=warmup, reps=reps)
        _run(records, prob, I.build_camps_ladder, tol=TOL, warmup=warmup,
             reps=reps, restart=30, maxiter=400)
        # residual-minimizing damping (review: damping before attributing limits)
        _run(records, prob, I.build_plain_ir_lu32, tol=TOL, warmup=warmup, reps=reps,
             damping="residual_min", label="plain_ir_lu32_damped")
        _run(records, prob, I.build_camps_lu32, **common, damping="residual_min",
             label="camps_lu32_damped")
        _run(records, prob, I.build_camps_ladder, tol=TOL, warmup=warmup,
             reps=reps, restart=30, maxiter=400, damping="residual_min",
             label="ladder_damped")
    save_results(os.path.join(RESULTS, "dense_ic.json"), "dense_ic", records)


def exp_sparse_spd(pilot: bool) -> None:
    """Sparse SPD class: 3D Poisson size sweep."""
    records: List[Dict] = []
    grids = [16] if pilot else [32, 48, 64]
    warmup, reps = (1, 3) if pilot else (3, 10)
    for g in grids:
        prob = sparse_problem("poisson", g, DEV, seed=300 + g)
        common = dict(tol=TOL, warmup=warmup, reps=reps)
        _run(records, prob, I.build_cg64, **common, maxiter=3000, jacobi=False)
        _run(records, prob, I.build_cg64, **common, maxiter=3000, jacobi=True)
        _run(records, prob, I.build_fgmres64_jacobi, **common, restart=30, maxiter=1500)
        _run(records, prob, I.build_plain_ir_cg32, **common)
        _run(records, prob, I.build_camps_cg32, **common, restart=30, maxiter=1500)
        _run(records, prob, I.build_camps_sparse_hierarchy, **common,
             restart=30, maxiter=1500, half_dtype=torch.float16)
    save_results(os.path.join(RESULTS, "sparse_spd.json"), "sparse_spd", records)


def exp_sparse_nsy(pilot: bool) -> None:
    """Sparse nonsymmetric class: convection-diffusion wind + size sweep."""
    records: List[Dict] = []
    cases = [(16, 20.0)] if pilot else [(32, 2.0), (32, 20.0), (32, 100.0), (48, 20.0)]
    warmup, reps = (1, 3) if pilot else (3, 10)
    for g, w in cases:
        prob = sparse_problem("convdiff", g, DEV, w=w, seed=400 + g)
        common = dict(tol=TOL, warmup=warmup, reps=reps, restart=30, maxiter=1500)
        _run(records, prob, I.build_gmres64_jacobi, **common)
        _run(records, prob, I.build_fgmres64_jacobi, **common)
        _run(records, prob, I.build_camps_fgmres32, **common)
        _run(records, prob, I.build_camps_sparse_hierarchy, **common,
             half_dtype=torch.float16)
    save_results(os.path.join(RESULTS, "sparse_nsy.json"), "sparse_nsy", records)


def exp_placement(pilot: bool) -> None:
    """Edge-placement ablation: same child family, different edge subsets."""
    records: List[Dict] = []
    n = 1024 if pilot else 2048
    warmup, reps = (1, 3) if pilot else (3, 15)
    for kappa in ([1e4] if pilot else [1e2, 1e4, 1e6]):
        prob = dense_conditioned(n, kappa=kappa, device=DEV, seed=500)
        common = dict(tol=TOL, warmup=warmup, reps=reps, restart=20, maxiter=300)
        _run(records, prob, I.build_camps_inv32, **common)
        _run(records, prob, I.build_camps_inv16_direct, **common,
             half_dtype=torch.float16)
        _run(records, prob, I.build_camps_inv16_direct, **common,
             half_dtype=torch.bfloat16, label="inv_b16_direct")
        _run(records, prob, I.build_camps_nested_ir16, **common,
             half_dtype=torch.float16)
        _run(records, prob, I.build_camps_hierarchy_dense, **common,
             half_dtype=torch.float16)
        _run(records, prob, I.build_camps_hierarchy_dense, **common,
             half_dtype=torch.bfloat16, label="hierarchy_b16")
    save_results(os.path.join(RESULTS, "placement.json"), "placement", records)


def _tune(prob, build_fn, grid: List[Dict], label: str, tol: float,
          **fixed) -> Dict:
    """Fair tuning: try each config, keep the fastest converged one. Applied
    identically to F3R, the hybrid, and the full CAMPS hierarchy."""
    best, best_ms = None, float("inf")
    for cfg in grid:
        try:
            rec = run_method(prob, build_fn, tol=tol, warmup=1, reps=2,
                             setup_reps=1, restart=30, maxiter=800,
                             **fixed, **cfg)
        except Exception:
            continue
        ms = rec["solve_ms"]["median"]
        print(f"  tune {label} {cfg}: {'ok' if rec['converged'] else 'DNC'} "
              f"outer={rec['outer_iterations']} {ms:.0f}ms", flush=True)
        if rec["converged"] and ms < best_ms:
            best, best_ms = cfg, ms
    best = best if best is not None else grid[0]
    print(f"  tuned {label}: {best}", flush=True)
    return best


F3R_GRID = [dict(m2=m2, m3=m3, sweeps=s)
            for m2 in (10, 20, 30) for m3 in (5, 10) for s in (2, 4)]
HYBRID_GRID = [dict(m3=5, mid_maxiter=15), dict(m3=5, mid_maxiter=10),
               dict(m3=8, mid_maxiter=10)]
IRTOP_GRID = [dict(m2=m2, m3=m3, sweeps=2)
              for m2 in (10, 15, 20) for m3 in (5, 8)]
HIER_GRID = [
    dict(),  # defaults
    dict(J_outer=1, J_inner=1, mid_tol=1e-3, mid_restart=10, mid_maxiter=10,
         leaf_tol=1e-1, leaf_restart=5, leaf_maxiter=5),
    dict(J_outer=2, J_inner=1, mid_tol=1e-3, mid_restart=10, mid_maxiter=20,
         leaf_tol=1e-1, leaf_restart=8, leaf_maxiter=8),
    dict(J_outer=1, J_inner=1, mid_tol=1e-4, mid_restart=15, mid_maxiter=15,
         leaf_tol=5e-2, leaf_restart=6, leaf_maxiter=6),
]


def exp_f3r(pilot: bool) -> None:
    """Head-to-head vs the F3R nesting structure [9] on both sparse classes,
    plus the mixed adapter/cast tree that isolates the per-edge contract cost.
    All three hierarchies receive the same fair tuning procedure."""
    records: List[Dict] = []
    warmup, reps = (1, 3) if pilot else (3, 10)
    cases = ([("convdiff", 16, 20.0)] if pilot else
             [("poisson", 32, 0.0), ("poisson", 48, 0.0),
              ("convdiff", 32, 20.0), ("convdiff", 48, 20.0)])
    tuned: Dict[str, Dict[str, Dict]] = {}
    for kind, g, w in cases:
        prob = sparse_problem(kind, g, DEV, w=w, seed=600 + g)
        if kind not in tuned:
            tuned[kind] = {
                "f3r": _tune(prob, I.build_f3r, F3R_GRID, "F3R", TOL),
                "hybrid": _tune(prob, I.build_camps_hybrid_cast, HYBRID_GRID,
                                "hybrid", TOL),
                "hier": _tune(prob, I.build_camps_sparse_hierarchy, HIER_GRID,
                              "hierarchy", TOL, half_dtype=torch.float16),
                "irtop": _tune(prob, I.build_camps_ir_f3r_body, IRTOP_GRID,
                               "IR-topped", TOL),
            }
        common = dict(tol=TOL, warmup=warmup, reps=reps, restart=30, maxiter=1500)
        if kind == "poisson":
            _run(records, prob, I.build_cg64, tol=TOL, warmup=warmup, reps=reps,
                 maxiter=3000, jacobi=False)
        _run(records, prob, I.build_fgmres64_jacobi, **common)
        _run(records, prob, I.build_f3r, **common, **tuned[kind]["f3r"])
        _run(records, prob, I.build_camps_hybrid_cast, **common,
             **tuned[kind]["hybrid"])
        _run(records, prob, I.build_camps_sparse_hierarchy, **common,
             half_dtype=torch.float16, **tuned[kind]["hier"])
        _run(records, prob, I.build_camps_ir_f3r_body, tol=TOL, warmup=warmup,
             reps=reps, **tuned[kind]["irtop"])
    save_results(os.path.join(RESULTS, "f3r.json"), "f3r", records,
                 extra={"tuned_configs": tuned})


def exp_multirhs(pilot: bool) -> None:
    """Dense multi-RHS amortization: setup once, solve an (n, k) block."""
    records: List[Dict] = []
    n = 1024 if pilot else 4096
    ks = [1, 4] if pilot else [1, 4, 8, 16]
    warmup, reps = (1, 2) if pilot else (2, 5)
    prob = dense_conditioned(n, kappa=1e2, device=DEV, seed=700)
    g = torch.Generator(device="cpu").manual_seed(701)
    X_all = (torch.randn(n, max(ks), generator=g, dtype=torch.float64)
             / (n ** 0.5)).to(DEV)
    B_all = prob.dense(torch.float64) @ X_all
    for k in ks:
        X, B = X_all[:, :k], B_all[:, :k]
        common = dict(tol=TOL, warmup=warmup, reps=reps, rhs=B, x_star=X,
                      restart=20, maxiter=300)
        _run(records, prob, I.build_lu64_direct, tol=TOL, warmup=warmup,
             reps=reps, rhs=B, x_star=X)
        _run(records, prob, I.build_fgmres64_lu64, **common)
        _run(records, prob, I.build_plain_ir_lu32, tol=TOL, warmup=warmup,
             reps=reps, rhs=B, x_star=X)
        _run(records, prob, I.build_camps_lu32, **common)
        _run(records, prob, I.build_camps_hierarchy_dense, **common,
             half_dtype=torch.float16)
    save_results(os.path.join(RESULTS, "multirhs.json"), "multirhs", records)


EXPERIMENTS: Dict[str, Callable[[bool], None]] = {
    "dense_wc": exp_dense_wc,
    "dense_ic": exp_dense_ic,
    "sparse_spd": exp_sparse_spd,
    "sparse_nsy": exp_sparse_nsy,
    "placement": exp_placement,
    "f3r": exp_f3r,
    "multirhs": exp_multirhs,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    choices=list(EXPERIMENTS.keys()) + ["all"])
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(0)
    torch.sparse.check_sparse_tensor_invariants.disable()
    names = list(EXPERIMENTS) if args.exp == "all" else [args.exp]
    for name in names:
        print(f"=== {name} {'(pilot)' if args.pilot else ''} ===", flush=True)
        t0 = time.perf_counter()
        EXPERIMENTS[name](args.pilot)
        print(f"=== {name} done in {time.perf_counter() - t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
