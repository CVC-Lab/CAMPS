"""Regenerate results/dense_ic_side_checks.json (paper Section 5.4).

For every condition number of the ill-conditioned sweep (same problems and
seeds as results/dense_ic.json) this records

  boosted_pivots_LU32, boosted_pivots_LU64
      how many pivots the zero-pivot boost (threshold 1e-8, replacement 1e-6,
      relative to the largest pivot; the policy inherited from the ILU(0)
      instance of the preliminary study) modifies in the FP32 and FP64 LU
      factorizations -- 278 in FP64 at kappa = 1e12, none at any smaller
      kappa, none in FP32;
  lu64_noboost
      the FP64 direct solve with the boost disabled (lu_pivot_tol = 0): the
      backward error that exposed the artifact;
  fallback_damped_rho0.9
      the contract-fallback instance with residual-minimizing damping and the
      progress threshold rho = 0.9, run with the boosted FP64 fallback child
      (lu_pivot_tol = 1e-8) exactly as the side check was first made. For
      kappa <= 1e8 no pivot is boosted so this equals the unboosted run; the
      paper's table uses results/dense_ic_noboost.json for kappa >= 1e10.
      At kappa = 1e12 the boosted fallback child itself violates the
      contract; since the Algorithm-1 revision a fallback violation returns
      REJECTED after that attempt (4 attempts, 2 accepted) instead of
      retrying the same deterministic child until the budget is exhausted
      (60 attempts, MAX_STEPS, identical final iterate). This entry is not
      used in the paper.

Protocol: 3 warmups, 15 timed repetitions, tau = 1e-10.
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camps import DenseLUSolver, DenseOperator, configure_backend  # noqa: E402
from experiments import instances as I  # noqa: E402
from experiments.harness import run_method  # noqa: E402
from experiments.matrices import dense_conditioned  # noqa: E402

from experiments.paths import results_dir
RESULTS = results_dir()
BOOST_TOL = 1e-8   # the inherited policy; the dense nodes now default to 0


def main():
    applied = configure_backend()
    out = {}
    for kappa in (1e4, 1e6, 1e7, 1e8, 1e10, 1e12):
        prob = dense_conditioned(2048, kappa=kappa, device="cuda",
                                 seed=int(200 + math.log10(kappa)))
        entry = {}
        entry["boosted_pivots_LU32"] = DenseLUSolver(
            DenseOperator(prob.dense(torch.float32)), pivot_tol=BOOST_TOL).boosted_pivots
        entry["boosted_pivots_LU64"] = DenseLUSolver(
            DenseOperator(prob.A64), pivot_tol=BOOST_TOL).boosted_pivots

        rec = run_method(prob, I.build_lu64_direct, tol=1e-10, warmup=3, reps=15, lu_pivot_tol=0.0)
        entry["lu64_noboost"] = {
            "converged": rec["converged"], "rel_res": rec["rel_residual"],
            "fwd": rec["forward_error"], "bn": rec["backward_error_normwise"],
            "bc": rec["backward_error_componentwise"],
            "solve_ms": rec["solve_ms"]["median"], "setup_ms": rec["setup_ms"]["median"],
        }

        rec = run_method(prob, I.build_plain_ir_lu32_fallback, tol=1e-10, warmup=3, reps=15,
                         damping="residual_min", rho=0.9, lu_pivot_tol=BOOST_TOL)
        lv = rec.get("adapter_levels", {}).get("depth0", {})
        entry["fallback_damped_rho0.9"] = {
            "converged": rec["converged"], "status": rec["status"],
            "corrections": rec["outer_iterations"],
            "solve_ms": rec["solve_ms"]["median"], "setup_ms": rec["setup_ms"]["median"],
            "rel_res": rec["rel_residual"], "fwd": rec["forward_error"],
            "bn": rec["backward_error_normwise"], "bc": rec["backward_error_componentwise"],
            "accepted": lv.get("accepted"), "rejected": lv.get("rejected"),
            "eta_hat_median": lv.get("eta_hat_median"),
        }
        key = f"{kappa:.0e}"
        out[key] = entry
        print(f"kappa={key}: boosted LU32/LU64 = {entry['boosted_pivots_LU32']}/{entry['boosted_pivots_LU64']}; "
              f"LU64 noboost rr={entry['lu64_noboost']['rel_res']:.2e} bn={entry['lu64_noboost']['bn']:.2e}; "
              f"damped fallback rho=0.9: {entry['fallback_damped_rho0.9']['status']} "
              f"corr={entry['fallback_damped_rho0.9']['corrections']} "
              f"acc/rej={entry['fallback_damped_rho0.9']['accepted']}/{entry['fallback_damped_rho0.9']['rejected']}",
              flush=True)
    out["environment"] = {"backend_flags": applied, "torch": torch.__version__,
                          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                          "protocol": "warmup=3, reps=15, tau=1e-10"}
    path = os.path.join(RESULTS, "dense_ic_side_checks.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
