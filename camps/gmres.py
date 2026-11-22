"""Right-preconditioned restarted GMRES with a FIXED preconditioner.

This is the baseline the paper says must not be forgotten (§6, "Outer IR versus
outer FGMRES"): with a fixed number of deterministic adapter corrections the
effective preconditioner is close to a fixed linear map, so ordinary GMRES is a
relevant baseline.

Standard right-preconditioned GMRES reconstructs x = x0 + M(V_m y_m) with ONE
preconditioner application at the end of each cycle. That reconstruction is
correct only when M is a fixed linear map; handing this solver a variable node
(e.g. a CAMPS adapter) is refused at construction (spec §10; a test
demonstrates the failure mode this refusal prevents).
"""
from __future__ import annotations

from typing import Optional

import torch

from .base_solvers import SolverNode
from .operators import LinearOperator
from .precision import colnorms, spec_of
from .results import CorrectionResult, Status


class GMRESSolver(SolverNode):
    is_fixed = True

    def __init__(
        self,
        op: LinearOperator,
        tol: float = 1e-8,
        atol: float = 0.0,
        restart: int = 30,
        maxiter: int = 1000,
        preconditioner: Optional[SolverNode] = None,
        allow_variable_preconditioner: bool = False,
    ):
        if preconditioner is not None:
            if preconditioner.dtype != op.dtype:
                raise TypeError("preconditioner format must match operator format")
            if not preconditioner.is_fixed and not allow_variable_preconditioner:
                raise TypeError(
                    "GMRES assumes a fixed linear preconditioner; use FGMRES for "
                    "a variable node such as a CAMPS adapter (spec §10). Pass "
                    "allow_variable_preconditioner=True only in tests that "
                    "deliberately demonstrate the failure."
                )
        self.op = op
        self.tol = tol
        self.atol = atol
        self.restart = restart
        self.maxiter = maxiter
        self.pre = preconditioner
        self.dtype = op.dtype
        self.device = op.device
        self.name = f"gmres[{spec_of(self.dtype).name}]"

    def _apply_pre(self, v: torch.Tensor) -> torch.Tensor:
        return v if self.pre is None else self.pre.solve(v).correction

    def solve(self, rhs: torch.Tensor) -> CorrectionResult:
        self._check_rhs(rhs)
        single = rhs.ndim == 1
        B = rhs.unsqueeze(1) if single else rhs
        X = torch.zeros_like(B)
        histories, statuses = [], []
        iters = 0
        for j in range(B.shape[1]):
            x, hist, it, st = self._solve_single(B[:, j])
            X[:, j] = x
            histories.append(hist)
            statuses.append(st)
            iters = max(iters, it)
        d = X.squeeze(1) if single else X
        init = colnorms(rhs)
        final = colnorms(rhs - self.op.apply(d))
        if not torch.isfinite(d).all():
            status = Status.FAILURE
        elif all(s == "converged" for s in statuses):
            status = Status.SUCCESS
        else:
            status = Status.MAX_STEPS
        return CorrectionResult(
            correction=d, initial_residual=init, final_residual=final,
            iterations=iters, status=status,
            diagnostics={"residual_history": histories, "per_column_status": statuses},
        )

    def _solve_single(self, b: torch.Tensor):
        acc = spec_of(self.dtype).accumulation_dtype
        n = b.shape[0]
        x = torch.zeros_like(b)  # zero start (spec §2)
        bnorm = float(torch.linalg.vector_norm(b.to(acc)))
        target = self.tol * bnorm + self.atol
        hist = [bnorm]
        if bnorm == 0.0 or bnorm <= target:
            return x, hist, 0, "converged"
        total = 0
        while total < self.maxiter:
            r = b - self.op.apply(x)  # true residual at each restart (paper §4.5)
            beta = float(torch.linalg.vector_norm(r.to(acc)))
            hist.append(beta)
            if beta <= target:
                return x, hist, total, "converged"
            m = min(self.restart, self.maxiter - total)
            V = torch.zeros(n, m + 1, dtype=self.dtype, device=b.device)
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
                w = self.op.apply(self._apply_pre(V[:, j_it]))
                # modified Gram-Schmidt, reductions in acc (choice I-12)
                for i in range(j_it + 1):
                    hij = torch.dot(V[:, i].to(acc), w.to(acc))
                    H[i, j_it] = hij
                    w = w - hij.to(self.dtype) * V[:, i]
                hnext = torch.linalg.vector_norm(w.to(acc))
                H[j_it + 1, j_it] = hnext
                if float(hnext) > 0:
                    V[:, j_it + 1] = (w / hnext.to(self.dtype))
                # apply stored Givens rotations
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
                hist.append(est)  # Hessenberg estimate; may drift (paper §4.5)
                if est <= target or float(hnext) == 0.0:
                    break
            y = torch.linalg.solve_triangular(
                H[:k, :k], g[:k].unsqueeze(1), upper=True
            ).squeeze(1)
            # x = x + M (V_k y): ONE application of the (assumed fixed, linear) M
            x = x + self._apply_pre((V[:, :k] @ y.to(self.dtype)))
        r = b - self.op.apply(x)
        rn = float(torch.linalg.vector_norm(r.to(acc)))
        hist.append(rn)
        return x, hist, total, "converged" if rn <= target else "max_steps"
