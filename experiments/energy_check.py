"""Certified energy contract (paper Section 4.3): counterexample and Poisson rerun.

Part 1 reproduces the reviewer's counterexample for the proposed-step energy
test and shows that the certified test on the committed displacement rejects
it:  A = 1, accepted y = 2^53, g = 2^53 + 2 (so r = 2), proposed d = 1/4.
    proposed-step expression  d'Ad/2 - r'd = -0.46875  (would accept)
    fl(y + d) = y, so h = 0, dphi_hat = 0, bound E = 0  (certified test rejects)

Part 2 reruns the sparse SPD plain-IR instance IR64<-32[CG32(Jacobi32)] on
Poisson 32^3 and 48^3 under the residual contract and under the certified
energy contract (same protocol as results/sparse_spd.json: tau = 1e-10,
3 warmups, 10 timed repetitions), and records per-correction dphi_hat and the
bound E so the certification margin is visible.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camps import AdapterConfig, CampsAdapter, DenseOperator, IdentitySolver, configure_backend  # noqa: E402
from experiments import instances as I  # noqa: E402
from experiments.harness import run_method  # noqa: E402
from experiments.matrices import sparse_problem  # noqa: E402

TOL = 1e-10


def counterexample(device):
    A64 = DenseOperator(torch.tensor([[1.0]], dtype=torch.float64, device=device),
                        symmetric=True, positive_definite=True)
    A32 = DenseOperator(torch.tensor([[1.0]], dtype=torch.float32, device=device))
    adapter = CampsAdapter(A64, IdentitySolver(A32), AdapterConfig(acceptance="energy"))
    y = torch.tensor([2.0 ** 53], dtype=torch.float64, device=device)
    g = torch.tensor([2.0 ** 53 + 2.0], dtype=torch.float64, device=device)
    d = torch.tensor([0.25], dtype=torch.float64, device=device)
    r_hat = g - A64.apply(y)                                  # = 2 exactly
    y_new = y + d                                             # fl(2^53 + 1/4) = 2^53
    proposed = float(0.5 * torch.dot(d, A64.apply(d)) - torch.dot(r_hat, d))
    dphi, bound, h = adapter.energy_certificate(g, y, r_hat, y_new)
    out = {
        "y": float(y), "g": float(g), "r": float(r_hat), "d": float(d),
        "y_new_equals_y": bool(torch.equal(y_new, y)),
        "proposed_step_expression": proposed,
        "proposed_step_accepts": proposed < 0,
        "committed_h": float(h), "dphi_hat": float(dphi), "bound": float(bound),
        "certified_test_accepts": bool(float(dphi) + float(bound) < 0),
    }
    print("counterexample:", json.dumps(out, indent=1))
    return out


def poisson(device):
    results = {}
    margins = {}
    for g in (32, 48):
        prob = sparse_problem("poisson", g, device, seed=300 + g)
        for acc in ("residual", "energy"):
            rec = run_method(prob, I.build_plain_ir_cg32, tol=TOL, warmup=3, reps=10,
                             acceptance=acc)
            lv = rec.get("adapter_levels", {}).get("depth0", {})
            key = f"g{g}_{acc}"
            results[key] = {
                "converged": rec["converged"], "status": rec["status"],
                "corrections": rec["outer_iterations"],
                "accepted": lv.get("accepted"), "rejected": lv.get("rejected"),
                "solve_ms": rec["solve_ms"]["median"],
                "rel_res": rec["rel_residual"],
                "eta_hat_median": lv.get("eta_hat_median"),
                "matvecs": rec.get("matvecs"),
            }
            print(f"{key}: {results[key]}")
            if acc == "energy":
                built = I.build_plain_ir_cg32(prob, tol=TOL, acceptance="energy")
                res = built.solver.solve(prob.b64)
                ev = [e for e in res.contract_record["events"] if "delta_phi" in e]
                margins[f"g{g}"] = [{"t": e["t"], "dphi_hat": e["delta_phi"],
                                     "bound": e["delta_phi_bound"],
                                     "ratio_bound_over_dphi": (e["delta_phi_bound"] / abs(e["delta_phi"])
                                                               if e["delta_phi"] else None),
                                     "accepted": e["satisfied"]} for e in ev]
                for m in margins[f"g{g}"]:
                    print(f"   t={m['t']}: dphi_hat={m['dphi_hat']:.3e} bound={m['bound']:.3e} "
                          f"bound/|dphi|={m['ratio_bound_over_dphi']:.2e} accepted={m['accepted']}")
    return results, margins


def main():
    applied = configure_backend()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ce = counterexample(device)
    res, margins = poisson(device)
    from experiments.paths import results_dir
    out = os.path.join(results_dir(), "spd_energy_check.json")
    payload = dict(res)
    payload["counterexample"] = ce
    payload["energy_margins"] = margins
    payload["backend"] = applied
    payload["torch"] = torch.__version__
    payload["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
