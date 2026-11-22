"""Regression check after the Algorithm-1 revisions (certified energy test,
success-before-stagnation, generalized fallback pair, attempt accounting):
rerun the fallback, thresholded-damped-fallback and ladder rows of the
ill-conditioned sweep at kappa = 1e8, 1e10, 1e12 and compare status, attempt
and accept/reject counts, and the attained relative residual with the stored
results (results/dense_ic.json for 1e8, results/dense_ic_noboost.json for the
rest). The arithmetic path of a same-precision fallback is unchanged, so the
residuals are expected to agree to the last bit."""
from __future__ import annotations

import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camps.diagnostics import configure_backend  # noqa: E402
from experiments import instances as I  # noqa: E402
from experiments.harness import run_method  # noqa: E402
from experiments.matrices import dense_conditioned  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stored(kappa, method):
    fn = "dense_ic.json" if kappa < 1e10 else "dense_ic_noboost.json"
    from experiments.paths import archive_dir
    d = json.load(open(os.path.join(archive_dir(), fn)))
    for r in d["records"]:
        if r["method"] == method and r.get("problem_meta", {}).get("kappa") == kappa:
            return r
    return None


def main():
    configure_backend()
    ok_all = True
    for kappa in (1e8, 1e10, 1e12):
        prob = dense_conditioned(2048, kappa=kappa, device="cuda",
                                 seed=int(200 + math.log10(kappa)))
        runs = [
            (I.build_plain_ir_lu32_fallback, {}),
            (I.build_plain_ir_lu32_fallback, {"damping": "residual_min", "rho": 0.9}),
            (I.build_camps_ladder, {"restart": 30, "maxiter": 400}),
        ]
        for fn, kw in runs:
            rec = run_method(prob, fn, tol=1e-10, warmup=1, reps=3, **kw)
            lv = rec.get("adapter_levels", {}).get("depth0", {})
            old = stored(kappa, rec["method"])
            if old is None:
                print(f"k={kappa:.0e} {rec['method']}: no stored row"); continue
            olv = old.get("adapter_levels", {}).get("depth0", {})
            same = (rec["status"] == old["status"] and rec["outer_iterations"] == old["outer_iterations"]
                    and lv.get("accepted") == olv.get("accepted") and lv.get("rejected") == olv.get("rejected")
                    and rec["rel_residual"] == old["rel_residual"])
            ok_all &= same
            print(f"k={kappa:.0e} {rec['method'][:52]:52s} new: {rec['status']:8s} it={rec['outer_iterations']:2d} "
                  f"acc/rej={lv.get('accepted')}/{lv.get('rejected')} rr={rec['rel_residual']:.6e} | "
                  f"stored: {old['status']:8s} it={old['outer_iterations']:2d} acc/rej={olv.get('accepted')}/{olv.get('rejected')} "
                  f"rr={old['rel_residual']:.6e} | {'IDENTICAL' if same else 'DIFFERS'}")
    print("ALL IDENTICAL" if ok_all else "SOME ROWS DIFFER")


if __name__ == "__main__":
    main()
