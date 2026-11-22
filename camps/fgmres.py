"""Flexible GMRES with a variable right preconditioner (paper §2, §4.5).

FGMRES stores each realized preconditioned basis vector z_j = P_j(v_j)
separately and permits P_j to vary between Arnoldi steps; the solution is
updated in span{z_1,...,z_m} while the true residual of the original system is
minimized (paper §2). Because every z_j is stored, the preconditioner need not
be fixed or linear — which is exactly why a CAMPS adapter (variable and
nonlinear: local stopping, scaling, failure handling, spec §10) is admissible
as its right preconditioner (Theorem 1, final step).

Reliable residuals (paper §4.5): the Hessenberg least-squares estimate may
drift from the true residual in finite precision, so this implementation
recomputes r = b - A x in the node's precision at every restart, uses that for
convergence decisions at restart boundaries, and reports BOTH the stopping
residual estimate and an independently recomputed post-solve residual.
"""
from __future__ import annotations

from typing import Optional

import torch

from .base_solvers import SolverNode
from .operators import LinearOperator
from .precision import colnorms, spec_of
from .results import CorrectionResult, Status


class FGMRESSolver(SolverNode):
    # An FGMRES node with a variable preconditioner is itself variable from an
    # enclosing solver's viewpoint (its applications depend on the child's
    # stopping paths).
    def __init__(
        self,
        op: LinearOperator,
        tol: float = 1e-8,
        atol: float = 0.0,
        restart: int = 30,
        maxiter: int = 1000,
        preconditioner: Optional[SolverNode] = None,
    ):
        if preconditioner is not None and preconditioner.dtype != op.dtype:
            raise TypeError(
                f"FGMRES preconditioner must present format {op.dtype} to its "
                f"parent (paper §3 type rule), got {preconditioner.dtype}"
            )
        self.op = op
        self.tol = tol
        self.atol = atol
        self.restart = restart
        self.maxiter = maxiter
        self.pre = preconditioner
        self.dtype = op.dtype
        self.device = op.device
        self.is_fixed = preconditioner.is_fixed if preconditioner is not None else True
        self.name = f"fgmres[{spec_of(self.dtype).name}]"

    def solve(self, rhs: torch.Tensor) -> CorrectionResult:
        self._check_rhs(rhs)
        single = rhs.ndim == 1
        B = rhs.unsqueeze(1) if single else rhs
        X = torch.zeros_like(B)
        histories, statuses, pre_logs, hess_finals = [], [], [], []
        iters = 0
        failed = False
        for j in range(B.shape[1]):
            x, hist, it, st, plog, hess_final = self._solve_single(B[:, j])
            X[:, j] = x
            histories.append(hist)
            statuses.append(st)
            pre_logs.append(plog)
            hess_finals.append(hess_final)
            iters = max(iters, it)
            failed = failed or st == "preconditioner_failure"
        d = X.squeeze(1) if single else X
        init = colnorms(rhs)
        # independently recomputed post-solve residual (paper §4.5)
        final = colnorms(rhs - self.op.apply(d))
        if failed or not torch.isfinite(d).all():
            status = Status.FAILURE
        elif all(s == "converged" for s in statuses):
            status = Status.SUCCESS
        else:
            status = Status.MAX_STEPS
        return CorrectionResult(
            correction=d, initial_residual=init, final_residual=final,
            iterations=iters, status=status,
            diagnostics={
                "residual_history": histories,
                "per_column_status": statuses,
                "stopping_residual_estimates": hess_finals,   # Hessenberg values
                "recomputed_final_residual": final,            # true values
                "preconditioner_log": pre_logs,
            },
        )

    def _solve_single(self, b: torch.Tensor):
        acc = spec_of(self.dtype).accumulation_dtype
        n = b.shape[0]
        x = torch.zeros_like(b)  # zero start (spec §2)
        bnorm = float(torch.linalg.vector_norm(b.to(acc)))
        target = self.tol * bnorm + self.atol
        hist = [bnorm]
        pre_log = []
        hess_final = bnorm
        if bnorm == 0.0 or bnorm <= target:
            return x, hist, 0, "converged", pre_log, hess_final
        total = 0
        while total < self.maxiter:
            # authoritative residual at each restart (paper §4.5)
            r = b - self.op.apply(x)
            beta = float(torch.linalg.vector_norm(r.to(acc)))
            hist.append(beta)
            if beta <= target:
                return x, hist, total, "converged", pre_log, hess_final
            m = min(self.restart, self.maxiter - total)
            V = torch.zeros(n, m + 1, dtype=self.dtype, device=b.device)
            Z = torch.zeros(n, m, dtype=self.dtype, device=b.device)  # stored z_j
            H = torch.zeros(m + 1, m, dtype=acc, device=b.device)
            cs = torch.zeros(m, dtype=acc, device=b.device)
            sn = torch.zeros(m, dtype=acc, device=b.device)
            g = torch.zeros(m + 1, dtype=acc, device=b.device)
            g[0] = beta
            V[:, 0] = (r / beta).to(self.dtype)
            k = 0
            for j_it in range(m):
                total += 1
                k = j_it + 1
                # variable preconditioner application P_j(v_j) — stored, never
                # re-applied (paper §2)
                if self.pre is None:
                    z = V[:, j_it]
                else:
                    pres = self.pre.solve(V[:, j_it].clone())
                    pre_log.append({
                        "iteration": total,
                        "status": pres.status.value,
                        "final_residual": _to_float_list(pres.final_residual),
                    })
                    z = pres.correction
                    if pres.status is Status.FAILURE or not torch.isfinite(z).all():
                        # Choice I-10: a dead preconditioner application leaves
                        # flexible Arnoldi nothing principled to orthogonalize;
                        # abort with diagnostics.
                        return x, hist, total, "preconditioner_failure", pre_log, hess_final
                Z[:, j_it] = z
                w = self.op.apply(z)
                for i in range(j_it + 1):
                    hij = torch.dot(V[:, i].to(acc), w.to(acc))
                    H[i, j_it] = hij
                    w = w - hij.to(self.dtype) * V[:, i]
                hnext = torch.linalg.vector_norm(w.to(acc))
                H[j_it + 1, j_it] = hnext
                if float(hnext) > 0:
                    V[:, j_it + 1] = (w / hnext.to(self.dtype))
                for i in range(j_it):
                    t = cs[i] * H[i, j_it] + sn[i] * H[i + 1, j_it]
                    H[i + 1, j_it] = -sn[i] * H[i, j_it] + cs[i] * H[i + 1, j_it]
                    H[i, j_it] = t
                denom = torch.sqrt(H[j_it, j_it] ** 2 + H[j_it + 1, j_it] ** 2)
                if float(denom) == 0.0:
                    cs[j_it], sn[j_it] = 1.0, 0.0
                else:
                    cs[j_it] = H[j_it, j_it] / denom
                    sn[j_it] = H[j_it + 1, j_it] / denom
                H[j_it, j_it] = cs[j_it] * H[j_it, j_it] + sn[j_it] * H[j_it + 1, j_it]
                H[j_it + 1, j_it] = 0.0
                g[j_it + 1] = -sn[j_it] * g[j_it]
                g[j_it] = cs[j_it] * g[j_it]
                est = abs(float(g[j_it + 1]))
                hess_final = est
                hist.append(est)  # inexpensive estimate; may drift (paper §4.5)
                if est <= target or float(hnext) == 0.0:
                    break
            y = torch.linalg.solve_triangular(
                H[:k, :k], g[:k].unsqueeze(1), upper=True
            ).squeeze(1)
            x = x + Z[:, :k] @ y.to(self.dtype)  # solution in span{z_1..z_k}
        r = b - self.op.apply(x)
        rn = float(torch.linalg.vector_norm(r.to(acc)))
        hist.append(rn)
        st = "converged" if rn <= target else "max_steps"
        return x, hist, total, st, pre_log, hess_final


def _to_float_list(t: torch.Tensor):
    return float(t) if t.ndim == 0 else [float(v) for v in t]
