"""Numerical check of the recursive contraction rules (paper Section 4.7,
Theorem "recursive convergence", rules R1-R3, and the cascade corollary).

Tree under test (a pure adapter chain, so every rule is rigorous):

    IR_{64<-32}^{J0} [ IR_{32<-16}^{J1} [ B_16 ] ]

B_16 is a stored approximate inverse applied in FP16 (Base node, rule R1).
For each condition number, leaf quality and J1 the script reports

  theta16        leaf factor: ||I - A16 B16||_2 (exact arithmetic, FP64 SVD)
                 plus the measured FP16 mat-vec rounding contribution
  eta_mid_meas   worst pre-floor per-correction contraction at the 32<-16 edge
  eta_mid_bound  rigorous rule R3 value  eta_hat  at that edge
  eta_top_meas   worst pre-floor per-correction contraction at the 64<-32 edge
  eta_top_bound  rigorous bound  T_{64<-32}( theta(IR_{32<-16}^{J1}) )
  plateau ratio  eta_top_meas / (u32 * kappa2)   (cascade corollary: the top
                 edge sees the INTERMEDIATE precision, not the lowest one)
  beta_root      limiting normwise backward error at the root (Theorem
                 "a posteriori stability" / limiting accuracy: O(n u64),
                 independent of the subtree)

Everything that enters a bound (norms, mismatches) is computed in FP64 on the
device; nothing is fitted.
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camps import (AdapterConfig, ApproxInverseSolver, CampsAdapter,  # noqa: E402
                   DenseOperator, configure_backend)
from experiments.matrices import dense_conditioned  # noqa: E402

U = {16: 2.0 ** -11, 32: 2.0 ** -24, 64: 2.0 ** -53}
LAMBDA = {16: 2.0 ** -24, 32: 2.0 ** -149, 64: 2.0 ** -1074}   # smallest subnormal


def gamma(k: int, p: int) -> float:
    ku = k * U[p]
    return ku / (1.0 - ku) if ku < 1 else float("inf")


def norm2(M: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(M.to(torch.float64), 2))


def inv_norm2(M: torch.Tensor) -> float:
    s = torch.linalg.svdvals(M.to(torch.float64))
    return float(1.0 / s[-1])


def T(theta: float, eps: float, delta: float) -> float:
    """One-edge propagation with the size bound derived from the residual bound."""
    return eps + (1.0 + eps) * (theta + (1.0 + theta) * delta)


def rule_R3(eta: float, n: int, p: int, kap: float, kap_abs: float, J: int, tau: float = 0.0):
    """Rigorous adapter rule. Returns dict with eta_hat, psi, c, theta."""
    g_n1 = gamma(n + 1, p)
    g_1 = gamma(1, p)
    a = g_n1 * kap_abs
    denom = 1.0 - g_1 * kap
    if not (denom > 0 and math.isfinite(g_n1)):
        return {"eta_hat": float("inf"), "psi": float("inf"), "c": float("inf"),
                "theta": float("inf"), "a": a}
    eta_hat = (eta + (1.0 + eta) * a) / denom
    psi = ((1.0 + eta) * g_n1 * (1.0 + kap_abs) + g_1 * kap) / denom
    D = (1.0 - g_n1) * (1.0 - a) - (1.0 + g_n1) * (1.0 + a) * eta_hat
    if D <= 0:
        return {"eta_hat": eta_hat, "psi": psi, "c": float("inf"),
                "theta": float("inf"), "a": a}
    c = ((1.0 + g_n1) * (1.0 + a) + 2.0) / D
    tau_plus = tau * (1.0 + g_n1) / ((1.0 - g_n1) * (1.0 - a))
    theta = max(eta_hat ** J, tau_plus) + c * psi
    return {"eta_hat": eta_hat, "psi": psi, "c": c, "theta": theta, "a": a}


def prefloor_ratios(hist, rel_floor: float = None, floor_factor: float = 30.0):
    """Per-correction ratios h[t+1]/h[t] while h[t+1] is safely above the floor.

    The recurrence proved in the paper is ||r_{t+1}|| <= eta_hat ||r_t|| + f, so a
    pure ratio tests eta_hat only while f is negligible. If ``rel_floor`` is given
    the floor is rel_floor*h[0]; otherwise it is the smallest residual the run
    attained (valid when the run was long enough to reach its floor). A ratio is
    kept when h[t+1] exceeds ``floor_factor`` times that floor.
    """
    h = [float(v) for v in hist]
    floor = rel_floor * h[0] if rel_floor is not None else min(h)
    out = []
    for t in range(len(h) - 1):
        if h[t] > 0 and h[t + 1] > floor_factor * floor:
            out.append(h[t + 1] / h[t])
    return out


def run_case(n: int, kappa: float, damp: float, J1_list, device, seed: int,
             adversarial: bool = False):
    prob = dense_conditioned(n, kappa=kappa, device=device, seed=seed)
    A64 = prob.A64
    A32 = A64.to(torch.float32)
    A16 = A32.to(torch.float16)
    b = prob.b64
    if adversarial:
        # worst-case right-hand side: left singular vector of sigma_min, so that
        # ||A^{-1} b|| = ||A^{-1}|| ||b|| and every kappa-dependent term is attained
        Usv, _, _ = torch.linalg.svd(A64)
        b = Usv[:, -1].contiguous()

    # right-hand-side amplification ||A|| ||A^{-1} b|| / ||b||  (1 <= amp <= kappa)
    amp = float(torch.linalg.vector_norm(torch.linalg.solve(A64, b))
                * norm2(A64) / torch.linalg.vector_norm(b))

    # ---- quantities entering the rules (all in FP64) ----------------------
    k64 = norm2(A64) * inv_norm2(A64)
    k32 = norm2(A32) * inv_norm2(A32)
    kabs64 = norm2(A64.abs()) * inv_norm2(A64)
    kabs32 = norm2(A32.abs()) * inv_norm2(A32)
    delta_32_16 = norm2(A32.to(torch.float64) - A16.to(torch.float64)) * inv_norm2(A16)
    delta_64_32 = norm2(A64 - A32.to(torch.float64)) * inv_norm2(A32)
    eps_32_16 = U[16] + math.sqrt(n) * LAMBDA[16]
    eps_64_32 = U[32] + math.sqrt(n) * LAMBDA[32]

    # ---- leaf: stored approximate inverse applied in FP16 -----------------
    B16 = (torch.linalg.inv(A64) * damp).to(torch.float16)
    leaf = ApproxInverseSolver(B16, DenseOperator(A16), label="approxinv")
    A16d, B16d = A16.to(torch.float64), B16.to(torch.float64)
    theta_exact = norm2(torch.eye(n, dtype=torch.float64, device=device) - A16d @ B16d)
    # measured FP16 mat-vec rounding contribution over probe right-hand sides
    gen = torch.Generator(device="cpu").manual_seed(seed + 77)
    rho_mv, theta_probe = 0.0, 0.0
    for _ in range(64):
        g = torch.randn(n, generator=gen, dtype=torch.float64).to(device)
        g = g / g.abs().max()                      # top binade, as the adapter's scaling does
        g16 = g.to(torch.float16)
        d16 = leaf.solve(g16).correction
        g16d, d16d = g16.to(torch.float64), d16.to(torch.float64)
        exact = B16d @ g16d
        rho_mv = max(rho_mv, float(torch.linalg.vector_norm(A16d @ (d16d - exact))
                                   / torch.linalg.vector_norm(g16d)))
        theta_probe = max(theta_probe, float(torch.linalg.vector_norm(g16d - A16d @ d16d)
                                             / torch.linalg.vector_norm(g16d)))
    theta16 = theta_exact + rho_mv

    # ---- rigorous rules -----------------------------------------------------
    eta_mid = T(theta16, eps_32_16, delta_32_16)
    rows = []
    for J1 in J1_list:
        mid_rule = rule_R3(eta_mid, n, 32, k32, kabs32, J1)
        eta_top = T(mid_rule["theta"], eps_64_32, delta_64_32) if math.isfinite(mid_rule["theta"]) else float("inf")
        top_rule = rule_R3(eta_top, n, 64, k64, kabs64, 1) if math.isfinite(eta_top) else {"eta_hat": float("inf")}

        mid = CampsAdapter(DenseOperator(A32), leaf,
                           AdapterConfig(J=J1, tau=0.0, stag_h=10 ** 9, on_violation="reject"))
        root = CampsAdapter(DenseOperator(A64), mid,
                            AdapterConfig(J=90, tau=0.0, stag_h=10 ** 9, on_violation="reject"))
        res = root.solve(b)
        x = res.correction
        # root: run long enough (J=90) to reach its floor, so use the attained floor
        top_ratios = prefloor_ratios(res.residual_history)
        # middle edge: first application only (right-hand side b, amplification
        # amp); its floor is ~ u32 * amp relative to ||g||, measured below
        mid_ratios = prefloor_ratios(res.child_results[0].residual_history,
                                     rel_floor=U[32] * amp) if res.child_results else []
        r = b - A64 @ x
        beta = float(torch.linalg.vector_norm(r)
                     / (norm2(A64) * torch.linalg.vector_norm(x) + torch.linalg.vector_norm(b)))
        rows.append({
            "J1": J1,
            "eta_mid_meas": max(mid_ratios) if mid_ratios else None,
            "eta_mid_bound": mid_rule["eta_hat"],
            "theta_mid_bound": mid_rule["theta"],
            "eta_top_meas": max(top_ratios) if top_ratios else None,
            "eta_top_first": top_ratios[0] if top_ratios else None,
            "eta_top_bound": top_rule["eta_hat"],
            "plateau_over_u32kappa": (max(top_ratios) / (U[32] * k32)) if top_ratios else None,
            "eta_top_over_u32amp": (top_ratios[0] / (U[32] * amp)) if top_ratios else None,
            "beta_root": beta,
            "root_status": res.status.value,
            "root_corrections": res.iterations,
        })
    return {
        "n": n, "kappa": kappa, "damp": damp, "adversarial": adversarial,
        "rhs_amplification": float(torch.linalg.vector_norm(torch.linalg.solve(A64, b))
                                   * norm2(A64) / torch.linalg.vector_norm(b)),
        "kappa2_64": k64, "kappa_abs_64": kabs64, "kappa2_32": k32, "kappa_abs_32": kabs32,
        "delta_32_16": delta_32_16, "delta_64_32": delta_64_32,
        "theta16_exact": theta_exact, "theta16_mv_rounding": rho_mv,
        "theta16_probe_max": theta_probe, "theta16": theta16,
        "T_mid": eta_mid, "rows": rows,
    }


def main():
    applied = configure_backend()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 512
    cases = []
    for kappa in (1e1, 1e2, 1e3):
        for damp, adversarial in ((1.0, False), (0.5, False), (1.0, True)):
            J1_list = [1, 2, 3, 4, 6, 8] if damp == 1.0 else [1, 2, 4, 8, 16, 24, 32]
            seed = int(300 + math.log10(kappa) * 10 + (0 if damp == 1.0 else 1))
            c = run_case(n, kappa, damp, J1_list, device, seed, adversarial=adversarial)
            cases.append(c)
            print(f"\nkappa={kappa:.0e} damp={damp} adversarial={adversarial} "
                  f"(||A|| ||A^-1 b||/||b|| = {c['rhs_amplification']:.1f}): theta16={c['theta16']:.3e} "
                  f"(exact {c['theta16_exact']:.3e} + mv {c['theta16_mv_rounding']:.1e}; probe max {c['theta16_probe_max']:.3e}) "
                  f"delta32<-16={c['delta_32_16']:.2e} delta64<-32={c['delta_64_32']:.2e} "
                  f"kappa'32={c['kappa_abs_32']:.0f} T_mid={c['T_mid']:.3e}")
            print("  J1 | eta_mid meas / bound | eta_top meas (first) / bound | eta_top/(u32 amp) | beta_root | status")
            for r in c["rows"]:
                f = lambda v: "   --   " if v is None else f"{v:8.2e}"
                print(f"  {r['J1']:2d} | {f(r['eta_mid_meas'])} / {f(r['eta_mid_bound'])} | "
                      f"{f(r['eta_top_meas'])} ({f(r['eta_top_first'])}) / {f(r['eta_top_bound'])} | "
                      f"{f(r['eta_top_over_u32amp'])} | {r['beta_root']:.1e} | {r['root_status']} ({r['root_corrections']})")
    from experiments.paths import results_dir
    out = os.path.join(results_dir(), "verify_recursion.json")
    with open(out, "w") as fh:
        json.dump({"experiment": "verify_recursion", "backend": applied,
                   "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "cases": cases}, fh, indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
