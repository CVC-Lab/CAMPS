"""Measurement harness for the CAMPS matrix-class sweep.

Protocol (paper §5.2):

- uniform stopping rule: every method targets the same relative residual
  ||b - A x||_2 <= tol * ||b||_2, and the harness *independently recomputes*
  the residual in FP64 after every solve; convergence is claimed only on the
  recomputed value (paper §4.5).
- warmup: `warmup` untimed solves precede `reps` timed solves; timings are
  synchronized (torch.cuda.synchronize around each solve) and reported as
  median + IQR + mean + std, never bare means.
- setup (casts, factorizations, inverse construction) is timed separately,
  with derived-precision caches cleared between setup repetitions so each
  build pays its own conversion cost.
- memory: analytic per-precision byte accounting from each builder, plus
  measured CUDA allocator numbers (resident delta after build; peak delta
  during a solve).
- environment (GPU, driver, torch/CUDA versions, backend flags) is recorded
  in every result file; TF32 and reduced-precision reductions are disabled so
  'fp32' means IEEE fp32 multiplies (camps.diagnostics.configure_backend).
"""
from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional

import torch

from camps.diagnostics import configure_backend
from camps.results import AdapterResult, CorrectionResult

from .instances import BuiltMethod
from .matrices import Problem
from .nodes import reset_counts


def environment_record() -> Dict[str, Any]:
    rec = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "backend_flags": configure_backend(),
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        rec["driver"] = out.stdout.strip()
    except Exception:
        rec["driver"] = "unavailable"
    return rec


def _residual_vec_fp64(problem: Problem, x: torch.Tensor,
                       b: Optional[torch.Tensor] = None) -> torch.Tensor:
    """FP64 residual vector b - A64 x, recomputed independently of the solver."""
    b = problem.b64 if b is None else b
    if problem.kind == "dense":
        return b - problem.dense(torch.float64) @ x
    crow, col, vals, _ = problem.csr_parts(torch.float64)
    csr = torch.sparse_csr_tensor(crow, col, vals, size=(problem.n, problem.n))
    return b - csr @ x


def _abs_matvec_fp64(problem: Problem, v: torch.Tensor) -> torch.Tensor:
    """|A64| v for the backward-error denominators."""
    if problem.kind == "dense":
        return problem.dense(torch.float64).abs() @ v
    crow, col, vals, _ = problem.csr_parts(torch.float64)
    csr = torch.sparse_csr_tensor(crow, col, vals.abs(), size=(problem.n, problem.n))
    return csr @ v


def _residual_fp64(problem: Problem, x: torch.Tensor,
                   b: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-column FP64 residual norms (0-d for a single rhs)."""
    r = _residual_vec_fp64(problem, x, b)
    return torch.linalg.vector_norm(r, dim=0) if r.ndim == 2 else torch.linalg.vector_norm(r)


def backward_errors(problem: Problem, x: torch.Tensor,
                    b: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Normwise and componentwise backward errors (infinity norms), worst column.

    beta_norm = ||b - Ax||_inf / (||A||_inf ||x||_inf + ||b||_inf)
    beta_comp = max_i |b - Ax|_i / (|A||x| + |b|)_i   with the standard
    convention xi/0 = 0 if xi = 0 and infinity otherwise (Oettli-Prager;
    Arioli-Demmel-Duff 1989): a nonzero residual entry over a zero denominator
    cannot be explained by any componentwise perturbation of A and b."""
    b = problem.b64 if b is None else b
    r = _residual_vec_fp64(problem, x, b)
    ones = torch.ones(problem.n, dtype=torch.float64, device=x.device)
    a_inf = float(_abs_matvec_fp64(problem, ones).max())
    r_inf = r.abs().amax(dim=0) if r.ndim == 2 else r.abs().max()
    x_inf = x.abs().amax(dim=0) if x.ndim == 2 else x.abs().max()
    b_inf = b.abs().amax(dim=0) if b.ndim == 2 else b.abs().max()
    beta_norm = r_inf / (a_inf * x_inf + b_inf)
    denom = _abs_matvec_fp64(problem, x.abs()) + b.abs()
    zero_den = torch.where(r.abs() > 0, torch.full_like(r, float("inf")), torch.zeros_like(r))
    ratio = torch.where(denom > 0, r.abs() / torch.clamp(denom, min=torch.finfo(torch.float64).tiny),
                        zero_den)
    beta_comp = ratio.max()
    return {"normwise": float(beta_norm.max()), "componentwise": float(beta_comp),
            "A_inf_norm": a_inf}


def _walk_adapter_tree(res: CorrectionResult, agg: Dict[str, Any], depth: int = 0) -> None:
    """Aggregate corrections / acceptance / contract data over the result tree."""
    if isinstance(res, AdapterResult):
        key = f"depth{depth}"
        d = agg.setdefault(key, {"applications": 0, "corrections": 0,
                                 "accepted": 0, "rejected": 0,
                                 "contraction_ratios": []})
        d["applications"] += 1
        d["corrections"] += len(res.accepted)
        for a in res.accepted:
            if a is True or a == 1.0:
                d["accepted"] += 1
            else:
                d["rejected"] += 1
        for ev in res.contract_record.get("events", []):
            rb, ra = ev.get("r_before"), ev.get("r_after")
            if isinstance(rb, float) and isinstance(ra, float) and rb > 0:
                d["contraction_ratios"].append(ra / rb)
        for cres in res.child_results:
            _walk_adapter_tree(cres, agg, depth + 1)


def _tree_metrics(result: CorrectionResult, built: BuiltMethod) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}
    if built.wrapper is not None:
        for rec in built.wrapper.records:
            _walk_adapter_tree(rec, agg, 0)
    else:
        _walk_adapter_tree(result, agg, 0)
    # compress ratio lists into summary stats
    for d in agg.values():
        ratios = d.pop("contraction_ratios")
        if ratios:
            d["eta_hat_median"] = float(statistics.median(ratios))
            d["eta_hat_max"] = float(max(ratios))
            d["eta_hat_min"] = float(min(ratios))
    out: Dict[str, Any] = {"adapter_levels": agg}
    if built.zero_guard is not None:
        out["zero_guard_substitutions"] = built.zero_guard.substitutions
    return out


def _quantiles(xs: List[float]) -> Dict[str, float]:
    s = sorted(xs)
    med = statistics.median(s)
    if len(s) >= 4:
        q = statistics.quantiles(s, n=4)
        q1, q3 = q[0], q[2]
    else:
        q1, q3 = s[0], s[-1]
    return {
        "median": med, "q1": q1, "q3": q3, "iqr": q3 - q1,
        "mean": statistics.fmean(s), "std": statistics.pstdev(s) if len(s) > 1 else 0.0,
        "min": s[0], "max": s[-1], "n": len(s),
    }


def run_method(problem: Problem, build_fn: Callable[..., BuiltMethod],
               tol: float, warmup: int = 3, reps: int = 15, setup_reps: int = 3,
               rhs: Optional[torch.Tensor] = None,
               x_star: Optional[torch.Tensor] = None,
               **build_kwargs) -> Dict[str, Any]:
    """Build, warm up, and time one method on one problem.

    rhs/x_star override the problem's single manufactured pair, e.g. with an
    (n, k) block for multi-RHS amortization runs; convergence then requires
    every column to meet the tolerance."""
    dev = problem.device
    base_cache_keys = set(problem._cache.keys())

    # ---- setup timing: each rep re-pays casts and factorizations ----------
    setup_times: List[float] = []
    built: Optional[BuiltMethod] = None
    for i in range(setup_reps + 1):  # first is warmup (also compiles kernels)
        for k in list(problem._cache.keys()):
            if k not in base_cache_keys:
                del problem._cache[k]
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        built = build_fn(problem, tol=tol, **build_kwargs)
        torch.cuda.synchronize(dev)
        if i > 0:
            setup_times.append((time.perf_counter() - t0) * 1e3)

    assert built is not None
    b = problem.b64 if rhs is None else rhs
    xs = problem.x_star if x_star is None else x_star

    # ---- resident-memory delta of one fresh build -------------------------
    for k in list(problem._cache.keys()):
        if k not in base_cache_keys:
            del problem._cache[k]
    torch.cuda.empty_cache()
    torch.cuda.synchronize(dev)
    mem_before = torch.cuda.memory_allocated(dev)
    built = build_fn(problem, tol=tol, **build_kwargs)
    torch.cuda.synchronize(dev)
    mem_resident = torch.cuda.memory_allocated(dev) - mem_before

    def one_solve() -> CorrectionResult:
        reset_counts(built.registry)
        if built.wrapper is not None:
            built.wrapper.reset()
        if built.zero_guard is not None:
            built.zero_guard.reset()
        return built.solver.solve(b)

    # ---- warmup -----------------------------------------------------------
    for _ in range(warmup):
        result = one_solve()
    torch.cuda.synchronize(dev)

    # ---- solve-peak memory (one instrumented solve, not timed) ------------
    torch.cuda.reset_peak_memory_stats(dev)
    mem0 = torch.cuda.memory_allocated(dev)
    result = one_solve()
    torch.cuda.synchronize(dev)
    mem_solve_peak = torch.cuda.max_memory_allocated(dev) - mem0

    # ---- timed repetitions ------------------------------------------------
    solve_times: List[float] = []
    for _ in range(reps):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        result = one_solve()
        torch.cuda.synchronize(dev)
        solve_times.append((time.perf_counter() - t0) * 1e3)

    # ---- verification (independent FP64 recomputation, paper §4.5) --------
    x = result.correction
    bnorms = (torch.linalg.vector_norm(b, dim=0) if b.ndim == 2
              else torch.linalg.vector_norm(b))
    rnorms = _residual_fp64(problem, x, b)
    rel_all = rnorms / bnorms
    rel_res = float(rel_all.max())            # worst column governs
    rnorm = float(rnorms.max())
    xerr = x - xs
    if x.ndim == 2:
        fwd_err = float((torch.linalg.vector_norm(xerr, dim=0)
                         / torch.linalg.vector_norm(xs, dim=0)).max())
    else:
        fwd_err = float(torch.linalg.vector_norm(xerr)
                        / torch.linalg.vector_norm(xs))
    converged = bool(rel_res <= tol)
    berr = backward_errors(problem, x, b)

    record: Dict[str, Any] = {
        "method": built.name,
        "problem": problem.name,
        "n": problem.n,
        "problem_meta": {k: v for k, v in problem.meta.items()},
        "tol": tol,
        "n_rhs": int(b.shape[1]) if b.ndim == 2 else 1,
        "status": result.status.value,
        "converged": converged,
        "rel_residual": rel_res,
        "abs_residual": rnorm,
        "forward_error": fwd_err,
        "backward_error_normwise": berr["normwise"],
        "backward_error_componentwise": berr["componentwise"],
        "outer_iterations": result.iterations,
        "setup_ms": _quantiles(setup_times),
        "solve_ms": _quantiles(solve_times),
        "total_ms_median": _quantiles(setup_times)["median"] + _quantiles(solve_times)["median"],
        "matvecs": {k: dict(v) for k, v in built.registry.items()},
        "memory": {
            "analytic_bytes": {label: int(nbytes) for label, nbytes in built.analytic_bytes},
            "analytic_total_MiB": sum(nb for _, nb in built.analytic_bytes) / 2**20,
            "measured_resident_MiB": mem_resident / 2**20,
            "measured_solve_peak_MiB": mem_solve_peak / 2**20,
        },
        "warmup": warmup, "reps": reps, "setup_reps": setup_reps,
        "notes": built.notes,
    }
    record.update(_tree_metrics(result, built))

    # convergence history of the outermost solver (for figures)
    hist = result.diagnostics.get("residual_history")
    if hist and isinstance(hist, list):
        h0 = hist[0] if isinstance(hist[0], list) else hist
        record["outer_residual_history"] = [float(v) for v in h0[:600]]
    elif isinstance(result, AdapterResult) and result.residual_history:
        vals = []
        for h in result.residual_history[:600]:
            if isinstance(h, (int, float)):
                vals.append(float(h))
            else:  # 0-d tensor, or per-column (k,) tensor: track worst column
                vals.append(float(h.max()))
        record["outer_residual_history"] = vals
    return record


def save_results(path: str, experiment: str, records: List[Dict[str, Any]],
                 extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "experiment": experiment,
        "environment": environment_record(),
        "records": records,
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {path} ({len(records)} records)")
