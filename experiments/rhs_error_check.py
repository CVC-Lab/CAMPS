"""Regenerate results/rhs_error_check.json (paper Appendix A.8 and A.9).

The dense experiments form the right-hand side from a manufactured vector,
b = fl(A x_m). The exact solution x_s = A^{-1} b of the stored system is
therefore not x_m. For the stored FP64 operands of the ill-conditioned sweep
(same problems and seeds as results/dense_ic.json) this script computes, with
error-free products and exact summation,

  eb_rel            ||b - A x_m||_2 / ||b||_2, the right-hand-side
                    construction error,
  eb_bound_gamma_n  its worst-case bound gamma_n || |A| |x_m| ||_2 / ||b||_2,
  xs_minus_xm_rel   ||x_s - x_m||_2 / ||x_m||_2, from x_s - x_m = A^{-1} e_b,

and, for the FP64 direct solve and the fallback instance, the forward error
against both reference vectors:

  fwd_vs_xm         ||x - x_m||_2 / ||x_m||_2 (the value the tables report),
  fwd_vs_xs         ||x - x_s||_2 / ||x_m||_2, from x - x_s = -A^{-1} r with
                    the residual r = b - A x computed exactly.

The products A^{-1} v are FP64 solves. Their relative accuracy is about
kappa * u64, which is enough for a norm. No timing is taken.

    python -m experiments.rhs_error_check
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import instances as I  # noqa: E402
from experiments.harness import save_results  # noqa: E402
from experiments.matrices import dense_conditioned  # noqa: E402
from experiments.paths import results_dir  # noqa: E402

DEV = "cuda"
SPLIT = 134217729.0  # 2**27 + 1
N = 2048
KAPPAS = (1e4, 1e8, 1e10, 1e12)
TOL = 1e-10


def two_product(a, b):
    """a*b = p + e exactly (Dekker), elementwise, without a fused multiply-add."""
    p = a * b
    ca = SPLIT * a
    ah = ca - (ca - a)
    al = a - ah
    cb = SPLIT * b
    bh = cb - (cb - b)
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def exact_residual(A, x, b):
    """b - A x with exact products and correctly rounded row sums."""
    out = np.empty(A.shape[0])
    for i in range(A.shape[0]):
        p, e = two_product(A[i], x)
        out[i] = math.fsum(np.concatenate([-p, -e, [b[i]]]).tolist())
    return out


def nrm(v):
    return float(np.linalg.norm(v))


def solve64(A_t, v):
    t = torch.as_tensor(v, dtype=torch.float64, device=A_t.device)
    return torch.linalg.solve(A_t, t).cpu().numpy()


def main():
    u = 2.0 ** -53
    records = []
    for kappa in KAPPAS:
        seed = int(200 + round(math.log10(kappa)))
        prob = dense_conditioned(N, kappa=kappa, device=DEV, seed=seed)
        A_t = prob.A64
        A, xm, b = A_t.cpu().numpy(), prob.x_star.cpu().numpy(), prob.b64.cpu().numpy()

        eb = exact_residual(A, xm, b)
        xs_minus_xm = solve64(A_t, eb)
        rec = {
            "problem": prob.name, "method": "exact right-hand-side check",
            "kappa": kappa, "seed": seed, "n": N,
            "eb_rel": nrm(eb) / nrm(b),
            "eb_over_u64": nrm(eb) / nrm(b) / u,
            "eb_bound_gamma_n": (N * u / (1 - N * u)) * nrm(np.abs(A) @ np.abs(xm)) / nrm(b),
            "xs_minus_xm_rel": nrm(xs_minus_xm) / nrm(xm),
            "norm_b": nrm(b), "norm_xm": nrm(xm),
            "solves": {},
        }
        methods = {"LU64 (direct)": lambda: I.build_lu64_direct(prob, lu_pivot_tol=0.0)}
        if kappa >= 1e10:
            methods["IR64<-32[LU32 | LU64 fallback]"] = lambda: I.build_plain_ir_lu32_fallback(
                prob, tol=TOL, lu_pivot_tol=0.0)
        for name, build in methods.items():
            x = build().solver.solve(prob.b64).correction.cpu().numpy()
            r = exact_residual(A, x, b)
            x_minus_xs = -solve64(A_t, r)
            rec["solves"][name] = {
                "rel_residual": nrm(r) / nrm(b),
                "fwd_vs_xm": nrm(x - xm) / nrm(xm),
                "fwd_vs_xs": nrm(x_minus_xs) / nrm(xm),
                "residual_only_bound": kappa * nrm(r) / nrm(xm),
            }
        records.append(rec)
        print(f"kappa {kappa:.0e}: ||e_b||/||b|| = {rec['eb_over_u64']:.2f} u64, "
              f"||x_s - x_m||/||x_m|| = {rec['xs_minus_xm_rel']:.2e}")
    save_results(os.path.join(results_dir(), "rhs_error_check.json"), "rhs_error_check", records,
                 extra={"note": "exact right-hand-side construction error and forward errors against "
                                "both reference vectors; no timings"})


if __name__ == "__main__":
    main()
