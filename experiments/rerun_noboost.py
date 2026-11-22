"""Regenerate results/dense_ic_noboost.json: the LU64-dependent rows of the
ill-conditioned sweep at kappa = 1e10 and 1e12 with pivot boosting disabled
for the dense LU nodes (lu_pivot_tol = 0, the default since the
factorization-policy artifact of paper Section 5.4 was found), under the full
protocol of the sweep (3 warmups, 15 timed repetitions), plus the
thresholded damped fallback (rho = 0.9). Rows for kappa <= 1e8 are unchanged
from results/dense_ic.json because no pivot was ever boosted there.
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camps.diagnostics import configure_backend  # noqa: E402
from experiments import instances as I  # noqa: E402
from experiments.harness import run_method, save_results  # noqa: E402
from experiments.matrices import dense_conditioned  # noqa: E402

from experiments.paths import results_dir
RESULTS = results_dir()


def main():
    out_path = os.path.join(RESULTS, "dense_ic_noboost.json")
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
    configure_backend()
    records = []

    def run(prob, fn, **kw):
        rec = run_method(prob, fn, tol=1e-10, warmup=3, reps=15, **kw)
        records.append(rec)
        lv = rec.get("adapter_levels", {}).get("depth0", {})
        print(f"  [{prob.name}] {rec['method']}: {'ok' if rec['converged'] else 'DNC'} "
              f"{rec['status']} outer={rec['outer_iterations']} "
              f"setup={rec['setup_ms']['median']:.1f} solve={rec['solve_ms']['median']:.1f} "
              f"rr={rec['rel_residual']:.1e} fe={rec['forward_error']:.1e} "
              f"bn={rec['backward_error_normwise']:.1e} bc={rec['backward_error_componentwise']:.1e} "
              f"acc/rej={lv.get('accepted', '-')}/{lv.get('rejected', '-')}", flush=True)

    for kappa in (1e10, 1e12):
        prob = dense_conditioned(2048, kappa=kappa, device="cuda",
                                 seed=int(200 + math.log10(kappa)))
        run(prob, I.build_lu64_direct)
        run(prob, I.build_fgmres64_lu64, restart=30, maxiter=400)
        run(prob, I.build_plain_ir_lu32_fallback)
        run(prob, I.build_plain_ir_lu32_fallback, damping="residual_min", rho=0.9)
        run(prob, I.build_camps_ladder, restart=30, maxiter=400)
        run(prob, I.build_camps_ladder, restart=30, maxiter=400, damping="residual_min")

    save_results(out_path, "dense_ic_noboost", records,
                 extra={"note": "LU64 nodes with pivot_tol=0 (no pivot boosting); kappa 1e10 and 1e12"})


if __name__ == "__main__":
    main()
