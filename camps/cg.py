"""Zero-start (preconditioned) conjugate gradients as a CAMPS node.

CG is NOT part of the paper's grammar (spec choice I-11). It is provided as a
Base_p-style correction solver for SPD systems (the K-FAC use case). Caveats,
recorded rather than hidden:

- CG minimizes the A-norm error; its 2-norm residual is not monotone, so a CG
  child does not satisfy contract (1) step-by-step by construction. It is
  admitted as a child only under the adapter's runtime acceptance test.
- A preconditioner used inside CG must be a fixed SPD map; `is_fixed` is
  checked and a variable preconditioner is refused (use FGMRES instead,
  spec §10).
"""
from __future__ import annotations

from typing import Optional

import torch

from .base_solvers import SolverNode
from .operators import LinearOperator
from .precision import colnorms, spec_of
from .results import CorrectionResult, Status


class CGSolver(SolverNode):
    is_fixed = True  # deterministic given (op, preconditioner, rhs)

    def __init__(
        self,
        op: LinearOperator,
        tol: float = 1e-8,
        atol: float = 0.0,
        maxiter: int = 1000,
        preconditioner: Optional[SolverNode] = None,
    ):
        if op.symmetric is False or op.positive_definite is False:
            raise TypeError("CGSolver requires an operator declared SPD (or unknown)")
        if preconditioner is not None and not preconditioner.is_fixed:
            raise TypeError(
                "CG assumes a fixed preconditioner; a variable node (e.g. a "
                "CAMPS adapter) requires FGMRES (spec §10)"
            )
        if preconditioner is not None and preconditioner.dtype != op.dtype:
            raise TypeError("preconditioner format must match operator format")
        self.op = op
        self.tol = tol
        self.atol = atol
        self.maxiter = maxiter
        self.pre = preconditioner
        self.dtype = op.dtype
        self.device = op.device
        self.name = f"cg[{spec_of(self.dtype).name}]"

    def solve(self, rhs: torch.Tensor) -> CorrectionResult:
        self._check_rhs(rhs)
        single = rhs.ndim == 1
        B = rhs.unsqueeze(1) if single else rhs
        acc = spec_of(self.dtype).accumulation_dtype

        X = torch.zeros_like(B)
        histories = []
        iters_used = 0
        statuses = []
        for j in range(B.shape[1]):
            x, hist, it, st = self._solve_single(B[:, j], acc)
            X[:, j] = x
            histories.append(hist)
            iters_used = max(iters_used, it)
            statuses.append(st)

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
            correction=d,
            initial_residual=init,
            final_residual=final,
            iterations=iters_used,
            status=status,
            diagnostics={
                "residual_history": histories,
                "per_column_status": statuses,
                "accumulation_dtype": str(acc),
            },
        )

    def _solve_single(self, b: torch.Tensor, acc: torch.dtype):
        def dot(u, v):
            # Choice I-12: reductions accumulate in `acc`, recorded in
            # diagnostics; storage stays in the node format.
            return torch.dot(u.to(acc), v.to(acc))

        x = torch.zeros_like(b)  # zero start (spec §2)
        r = b.clone()
        z = self.pre.solve(r).correction if self.pre is not None else r
        p = z.clone()
        rz = dot(r, z)
        bnorm = float(torch.linalg.vector_norm(b.to(acc)))
        target = self.tol * bnorm + self.atol
        hist = [float(torch.linalg.vector_norm(r.to(acc)))]
        if hist[0] <= target:
            return x, hist, 0, "converged"
        for it in range(1, self.maxiter + 1):
            Ap = self.op.apply(p)
            pAp = dot(p, Ap)
            if not torch.isfinite(pAp) or pAp <= 0:
                # loss of positive definiteness in this format — stop honestly
                return x, hist, it, f"breakdown(pAp={float(pAp):.3e})"
            alpha = (rz / pAp).to(b.dtype)
            x = x + alpha * p
            r = r - alpha * Ap
            rn = float(torch.linalg.vector_norm(r.to(acc)))
            hist.append(rn)
            if rn <= target:
                return x, hist, it, "converged"
            z = self.pre.solve(r).correction if self.pre is not None else r
            rz_new = dot(r, z)
            beta = (rz_new / rz).to(b.dtype)
            rz = rz_new
            p = z + beta * p
        return x, hist, self.maxiter, "max_steps"
