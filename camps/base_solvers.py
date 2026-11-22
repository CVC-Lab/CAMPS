"""Base solver nodes Base_p (spec §3; paper §3).

Every node here implements the uniform CAMPS interface (spec §2):

    solve(rhs) -> CorrectionResult, correction approximates A_p^{-1} rhs from a
    zero start, input and output in the node's own format.

`is_fixed` records whether the node realizes the same linear map at every call
(spec §10) — an enclosing GMRES may assume a fixed node; a variable node
requires FGMRES.

Zero-pivot handling follows the paper's evaluated instance (§5.1): detected
zero/tiny pivots are boosted (tolerance 1e-8, replacement 1e-6, relative to the
factor's scale) and the event recorded, rather than silently promoting
precision (spec choice I-13).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from .operators import DenseOperator, LinearOperator, as_operator
from .precision import colnorms, spec_of
from .results import CorrectionResult, Status

PIVOT_TOL = 1e-8
PIVOT_REPLACEMENT = 1e-6


class SolverNode(ABC):
    """S_p: rhs in F_p^n -> approximate solution in F_p^n (paper §3)."""

    dtype: torch.dtype
    device: torch.device
    is_fixed: bool = True
    name: str = "node"

    @abstractmethod
    def solve(self, rhs: torch.Tensor) -> CorrectionResult: ...

    def _check_rhs(self, rhs: torch.Tensor) -> None:
        if rhs.dtype != self.dtype:
            raise TypeError(
                f"{self.name}: expects rhs dtype {self.dtype}, got {rhs.dtype}; "
                "CAMPS forbids implicit casts (paper §3)"
            )


def _result(op: LinearOperator, rhs: torch.Tensor, d: torch.Tensor,
            iterations: int, status: Status, diagnostics: Dict[str, Any]) -> CorrectionResult:
    """Assemble a CorrectionResult, measuring ||r - A d|| in the node format."""
    init = colnorms(rhs)
    final = colnorms(rhs - op.apply(d))
    return CorrectionResult(
        correction=d,
        initial_residual=init,
        final_residual=final,
        iterations=iterations,
        status=status,
        diagnostics=diagnostics,
    )


class DenseLUSolver(SolverNode):
    """Direct LU solve in the node's format (fp32/fp64 only; torch.linalg
    does not support half-precision factorization — that backend limitation is
    itself a documented precision fact, spec §13)."""

    def __init__(self, operator, pivot_tol: float = PIVOT_TOL,
                 pivot_replacement: float = PIVOT_REPLACEMENT):
        op = as_operator(operator)
        if not isinstance(op, DenseOperator):
            raise TypeError("DenseLUSolver requires an explicitly stored matrix")
        if op.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"torch.linalg.lu_factor does not support {op.dtype}; use "
                "ApproxInverseSolver or an iterative node for half formats"
            )
        self.op = op
        self.dtype = op.dtype
        self.device = op.device
        self.name = f"lu[{spec_of(self.dtype).name}]"
        LU, piv = torch.linalg.lu_factor(op.matrix)
        # Zero-pivot boost (paper §5.1; choice I-13): U diagonal entries below
        # pivot_tol * max|diag U| are replaced by pivot_replacement * max|diag U|.
        diagU = LU.diagonal()
        scale = diagU.abs().max()
        self.boosted_pivots = 0
        if scale > 0:
            tiny = diagU.abs() < pivot_tol * scale
            self.boosted_pivots = int(tiny.sum())
            if self.boosted_pivots:
                diagU[tiny] = torch.sign(diagU[tiny] + (diagU[tiny] == 0)) * pivot_replacement * scale
        self._LU, self._piv = LU, piv

    def solve(self, rhs: torch.Tensor) -> CorrectionResult:
        self._check_rhs(rhs)
        single = rhs.ndim == 1
        R = rhs.unsqueeze(1) if single else rhs
        d = torch.linalg.lu_solve(self._LU, self._piv, R)
        d = d.squeeze(1) if single else d
        status = Status.FAILURE if not torch.isfinite(d).all() else Status.SUCCESS
        return _result(self.op, rhs, d, 1, status,
                       {"boosted_pivots": self.boosted_pivots})


class CholeskySolver(SolverNode):
    """Direct SPD solve via Cholesky in the node's format (fp32/fp64).

    On factorization failure, adds a documented diagonal jitter and retries
    (recorded, not silent — choice I-13 analogue for SPD)."""

    def __init__(self, operator, jitter: float = PIVOT_REPLACEMENT, max_tries: int = 3):
        op = as_operator(operator, symmetric=True, positive_definite=True)
        if op.dtype not in (torch.float32, torch.float64):
            raise TypeError("CholeskySolver requires fp32/fp64 (backend limitation)")
        self.op = op
        self.dtype = op.dtype
        self.device = op.device
        self.name = f"chol[{spec_of(self.dtype).name}]"
        A = op.matrix
        scale = A.diagonal().abs().max()
        self.jitter_applied = 0.0
        for attempt in range(max_tries + 1):
            L, info = torch.linalg.cholesky_ex(A)
            if int(info) == 0:
                break
            bump = jitter * float(scale) * (10.0 ** attempt)
            A = A + bump * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
            self.jitter_applied += bump
        else:
            raise RuntimeError("Cholesky failed after jitter attempts")
        if int(info) != 0:
            raise RuntimeError("Cholesky failed after jitter attempts")
        self._L = L

    def solve(self, rhs: torch.Tensor) -> CorrectionResult:
        self._check_rhs(rhs)
        single = rhs.ndim == 1
        R = rhs.unsqueeze(1) if single else rhs
        d = torch.cholesky_solve(R, self._L)
        d = d.squeeze(1) if single else d
        status = Status.FAILURE if not torch.isfinite(d).all() else Status.SUCCESS
        return _result(self.op, rhs, d, 1, status,
                       {"jitter_applied": self.jitter_applied})


class ApproxInverseSolver(SolverNode):
    """Fixed approximate inverse B_p stored and applied in the node's format.

    d = B @ r via matmul — the canonical honest low-precision child: storage in
    p, matmul arithmetic per backend (CUDA tensor cores accumulate FP32; see
    diagnostics.promotion_report / spec §13). ||I - A B|| < 1 is sufficient for
    contraction in exact arithmetic (paper §4.3) but is NOT verified here; the
    adapter's runtime acceptance test guards it.
    """

    def __init__(self, B: torch.Tensor, op: LinearOperator, label: str = "approxinv"):
        if B.dtype != op.dtype:
            raise TypeError("approximate inverse and operator must share a format")
        self.B = B
        self.op = op
        self.dtype = B.dtype
        self.device = B.device
        self.name = f"{label}[{spec_of(self.dtype).name}]"

    @classmethod
    def from_matrix(cls, A_parent: torch.Tensor, target_dtype: torch.dtype,
                    label: str = "approxinv") -> "ApproxInverseSolver":
        """Compute A^{-1} in fp64, then cast inverse and operator to target.

        The inverse is *computed* wide and *stored/applied* narrow; claiming a
        narrow solve refers to the application arithmetic only (spec §13).
        """
        B64 = torch.linalg.inv(A_parent.to(torch.float64))
        B = B64.to(target_dtype)
        op = DenseOperator(A_parent.to(target_dtype))
        return cls(B, op, label=label)

    def solve(self, rhs: torch.Tensor) -> CorrectionResult:
        self._check_rhs(rhs)
        d = self.B @ rhs
        status = Status.FAILURE if not torch.isfinite(d).all() else Status.SUCCESS
        return _result(self.op, rhs, d, 1, status, {"kind": "fixed approximate inverse"})


class JacobiSolver(SolverNode):
    """Diagonal (Jacobi) preconditioner: d = r / diag(A). Elementwise division
    executes natively in half formats (spec §13)."""

    def __init__(self, operator):
        op = as_operator(operator)
        if not isinstance(op, DenseOperator):
            raise TypeError("JacobiSolver needs a stored matrix (for its diagonal)")
        self.op = op
        self.dtype = op.dtype
        self.device = op.device
        self.name = f"jacobi[{spec_of(self.dtype).name}]"
        diag = op.matrix.diagonal().clone()
        zero = diag == 0
        self.zero_diagonal_entries = int(zero.sum())
        if self.zero_diagonal_entries:
            # Documented boost, mirroring choice I-13; a zero diagonal after
            # conversion means the child system was structurally damaged
            # (paper §4.1) — surfaced, not hidden.
            scale = diag.abs().max()
            diag[zero] = PIVOT_REPLACEMENT * (scale if scale > 0 else 1.0)
        self._diag = diag

    def solve(self, rhs: torch.Tensor) -> CorrectionResult:
        self._check_rhs(rhs)
        d = rhs / (self._diag if rhs.ndim == 1 else self._diag.unsqueeze(1))
        status = Status.FAILURE if not torch.isfinite(d).all() else Status.SUCCESS
        return _result(self.op, rhs, d, 1, status,
                       {"zero_diagonal_entries": self.zero_diagonal_entries})


class IdentitySolver(SolverNode):
    """d = r. A testing/baseline node: satisfies the residual contract only
    when ||I - A|| behaves — no guarantee is implied."""

    def __init__(self, op: LinearOperator):
        self.op = op
        self.dtype = op.dtype
        self.device = op.device
        self.name = f"identity[{spec_of(self.dtype).name}]"

    def solve(self, rhs: torch.Tensor) -> CorrectionResult:
        self._check_rhs(rhs)
        return _result(self.op, rhs, rhs.clone(), 0, Status.SUCCESS, {})
