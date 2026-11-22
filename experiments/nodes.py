"""Experiment-side operators and solver nodes.

The camps/ core is used verbatim (as in the sibling CAMPS application repos);
everything the matrix-class sweep additionally needs lives here:

- SparseCSROperator     torch sparse-CSR operator in a declared format
                        (CUDA CSR matvec is supported for fp64/fp32/fp16/bf16
                        on this backend; verified by the capability probe).
- CountingOperator      transparent wrapper that counts matvec/matmat calls
                        per precision — the data-movement metric.
- DiagJacobiSolver      Jacobi node from an explicit diagonal vector, so it
                        works for sparse and matrix-free operators in any
                        format (camps.JacobiSolver requires a stored dense
                        matrix).
- CountingPreconditioner
                        wrapper around the outermost preconditioner node that
                        retains every returned result object, so the harness
                        can walk the full AdapterResult tree for per-level
                        iteration counts and contract events.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from camps.base_solvers import SolverNode
from camps.operators import DenseOperator, LinearOperator
from camps.precision import colnorms, spec_of
from camps.results import CorrectionResult, Status

PIVOT_REPLACEMENT = 1e-6


class SparseCSROperator(LinearOperator):
    """Sparse CSR matrix stored and applied in a single declared format."""

    def __init__(self, crow: torch.Tensor, col: torch.Tensor, vals: torch.Tensor,
                 n: int, symmetric: Optional[bool] = None,
                 positive_definite: Optional[bool] = None,
                 diagonal: Optional[torch.Tensor] = None):
        spec_of(vals.dtype)
        self._csr = torch.sparse_csr_tensor(crow, col, vals, size=(n, n))
        self._abs_csr = None  # |A| in the same format, built on first use
        self.nnz = vals.numel()
        self.shape = (n, n)
        self.dtype = vals.dtype
        self.device = vals.device
        self.symmetric = symmetric
        self.positive_definite = positive_definite
        self._diag = diagonal  # optional explicit diagonal, same dtype
        self.max_nnz_per_row = int((crow[1:] - crow[:-1]).max().item())

    supports_abs = True

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self._csr @ x

    def matmat(self, X: torch.Tensor) -> torch.Tensor:
        return self._csr @ X

    def apply_abs(self, x: torch.Tensor) -> torch.Tensor:
        """|A||x| with the same CSR structure (values replaced by |values|)."""
        self._check_input(x)
        if self._abs_csr is None:
            self._abs_csr = torch.sparse_csr_tensor(
                self._csr.crow_indices(), self._csr.col_indices(),
                self._csr.values().abs(), size=self.shape)
        return self._abs_csr @ x.abs()

    @property
    def diagonal(self) -> torch.Tensor:
        if self._diag is None:
            raise ValueError("operator was built without an explicit diagonal")
        return self._diag

    def value_bytes(self) -> int:
        return self.nnz * self._csr.values().element_size()

    def index_bytes(self) -> int:
        return (self._csr.crow_indices().numel() * self._csr.crow_indices().element_size()
                + self._csr.col_indices().numel() * self._csr.col_indices().element_size())


class CountingDenseOperator(DenseOperator):
    """DenseOperator that counts matvec/matmat calls per precision.

    Subclasses DenseOperator (rather than wrapping) so nodes that require an
    explicitly stored matrix (DenseLUSolver, JacobiSolver) accept it."""

    def __init__(self, matrix: torch.Tensor, registry: Dict[str, Dict[str, int]],
                 symmetric: Optional[bool] = None,
                 positive_definite: Optional[bool] = None):
        super().__init__(matrix, symmetric=symmetric, positive_definite=positive_definite)
        key = spec_of(matrix.dtype).name
        self._stats = registry.setdefault(key, {"matvec": 0, "matmat_cols": 0})

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        self._stats["matvec"] += 1
        return super().matvec(x)

    def matmat(self, X: torch.Tensor) -> torch.Tensor:
        self._stats["matmat_cols"] += X.shape[1]
        return super().matmat(X)

    def apply_abs(self, x: torch.Tensor) -> torch.Tensor:
        # executed work of the certified energy test, counted separately
        self._stats["absmatvec"] = self._stats.get("absmatvec", 0) + (1 if x.ndim == 1 else x.shape[1])
        return super().apply_abs(x)


class CountingOperator(LinearOperator):
    """Transparent matvec/matmat counter around any LinearOperator."""

    def __init__(self, inner: LinearOperator, registry: Dict[str, Dict[str, int]],
                 label: Optional[str] = None):
        self.inner = inner
        self.shape = inner.shape
        self.dtype = inner.dtype
        self.device = inner.device
        self.symmetric = inner.symmetric
        self.positive_definite = inner.positive_definite
        self.supports_abs = getattr(inner, "supports_abs", False)
        self.max_nnz_per_row = getattr(inner, "max_nnz_per_row", None)
        key = label or spec_of(inner.dtype).name
        self._stats = registry.setdefault(key, {"matvec": 0, "matmat_cols": 0})

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        self._stats["matvec"] += 1
        return self.inner.matvec(x)

    def matmat(self, X: torch.Tensor) -> torch.Tensor:
        self._stats["matmat_cols"] += X.shape[1]
        return self.inner.matmat(X)

    def apply_abs(self, x: torch.Tensor) -> torch.Tensor:
        self._stats["absmatvec"] = self._stats.get("absmatvec", 0) + (1 if x.ndim == 1 else x.shape[1])
        return self.inner.apply_abs(x)

    # convenience passthroughs used by some builders
    @property
    def matrix(self):
        return self.inner.matrix

    @property
    def diagonal(self):
        return self.inner.diagonal


def reset_counts(registry: Dict[str, Dict[str, int]]) -> None:
    for stats in registry.values():
        for k in stats:
            stats[k] = 0


class DiagJacobiSolver(SolverNode):
    """d = r / diag(A), diagonal supplied explicitly (sparse/matrix-free safe).

    Mirrors camps.base_solvers.JacobiSolver, including the documented boost of
    zero diagonal entries (a zero diagonal after conversion means the child
    system was structurally damaged — surfaced, not hidden)."""

    is_fixed = True

    def __init__(self, op: LinearOperator, diagonal: torch.Tensor):
        if diagonal.dtype != op.dtype:
            raise TypeError("diagonal dtype must match operator dtype")
        self.op = op
        self.dtype = op.dtype
        self.device = op.device
        self.name = f"jacobi[{spec_of(self.dtype).name}]"
        diag = diagonal.clone()
        zero = diag == 0
        self.zero_diagonal_entries = int(zero.sum())
        if self.zero_diagonal_entries:
            scale = diag.abs().max()
            diag[zero] = PIVOT_REPLACEMENT * (scale if scale > 0 else 1.0)
        self._diag = diag

    def solve(self, rhs: torch.Tensor) -> CorrectionResult:
        self._check_rhs(rhs)
        d = rhs / (self._diag if rhs.ndim == 1 else self._diag.unsqueeze(1))
        status = Status.FAILURE if not torch.isfinite(d).all() else Status.SUCCESS
        init = colnorms(rhs)
        final = colnorms(rhs - self.op.apply(d))
        return CorrectionResult(
            correction=d, initial_residual=init, final_residual=final,
            iterations=1, status=status,
            diagnostics={"zero_diagonal_entries": self.zero_diagonal_entries},
        )


class RichardsonSolver(SolverNode):
    """Fixed-count Jacobi-preconditioned Richardson sweeps, zero start.

    d_{t+1} = d_t + D^{-1}(rhs - A d_t), exactly `sweeps` iterations, no
    stopping test. This is the innermost smoother of the F3R structure of
    Suzuki & Iwashita [9]: a terminal stationary sweep whose convergence is
    never checked locally."""

    is_fixed = True

    def __init__(self, op: LinearOperator, diagonal: torch.Tensor, sweeps: int = 2):
        if diagonal.dtype != op.dtype:
            raise TypeError("diagonal dtype must match operator dtype")
        self.op = op
        self.sweeps = sweeps
        self.dtype = op.dtype
        self.device = op.device
        self.name = f"richardson{sweeps}[{spec_of(self.dtype).name}]"
        diag = diagonal.clone()
        zero = diag == 0
        if int(zero.sum()):
            scale = diag.abs().max()
            diag[zero] = PIVOT_REPLACEMENT * (scale if scale > 0 else 1.0)
        self._diag = diag

    def solve(self, rhs: torch.Tensor) -> CorrectionResult:
        self._check_rhs(rhs)
        dinv = self._diag if rhs.ndim == 1 else self._diag.unsqueeze(1)
        d = rhs / dinv
        for _ in range(self.sweeps - 1):
            d = d + (rhs - self.op.apply(d)) / dinv
        status = Status.FAILURE if not torch.isfinite(d).all() else Status.SUCCESS
        init = colnorms(rhs)
        final = colnorms(rhs - self.op.apply(d))
        return CorrectionResult(
            correction=d, initial_residual=init, final_residual=final,
            iterations=self.sweeps, status=status, diagnostics={})


class CastBridge(SolverNode):
    """Plain-cast precision boundary: convert down, run the child, convert up.

    This is the F3R-style transition [9]: NO parent-level residual formation,
    NO scaling, NO acceptance test — the child result is trusted as cast. It
    exists here as the ablation partner of the CAMPS adapter: comparing
    CastBridge hierarchies against IR-adapter hierarchies isolates what the
    adapter's parent-residual contract adds (and costs)."""

    is_fixed = False

    def __init__(self, parent_dtype: torch.dtype, child: SolverNode):
        p, q = spec_of(parent_dtype), spec_of(child.dtype)
        p.assert_valid_edge(q)
        self.child = child
        self.dtype = parent_dtype
        self.device = child.device
        self.name = f"cast[{p.name}<-{q.name}]({child.name})"

    def solve(self, rhs: torch.Tensor) -> CorrectionResult:
        self._check_rhs(rhs)
        res = self.child.solve(rhs.to(self.child.dtype))
        d = res.correction.to(self.dtype)
        status = Status.FAILURE if not torch.isfinite(d).all() else res.status
        return CorrectionResult(
            correction=d, initial_residual=res.initial_residual,
            final_residual=res.final_residual, iterations=res.iterations,
            status=status, diagnostics={"bridge": self.name, "inner": res.diagnostics})


class ZeroGuardPreconditioner(SolverNode):
    """Return the identity direction when the wrapped node yields exactly zero.

    An FGMRES right preconditioner may be any variable map — each realized
    vector z_j is stored and the true residual is minimized over span{z_j}
    (paper §2) — but a zero z_j gives flexible Arnoldi nothing to orthogonalize.
    A CAMPS adapter can return the zero vector when its *first* correction is
    rejected by the acceptance test (contract violation rolled back, spec §6),
    which happens on severely ill-conditioned systems where the low-precision
    child stops contracting. This wrapper substitutes z_j = v_j in exactly that
    case, so the enclosing FGMRES degrades toward unpreconditioned FGMRES
    instead of dying; every substitution is counted and reported.
    """

    is_fixed = False

    def __init__(self, inner: SolverNode):
        self.inner = inner
        self.dtype = inner.dtype
        self.device = inner.device
        self.name = f"zeroguard({inner.name})"
        self.substitutions = 0

    def reset(self) -> None:
        self.substitutions = 0

    def solve(self, rhs: torch.Tensor) -> CorrectionResult:
        res = self.inner.solve(rhs)
        d = res.correction
        if res.status is not Status.FAILURE and bool((d == 0).all()):
            self.substitutions += 1
            return CorrectionResult(
                correction=rhs.clone(), initial_residual=res.initial_residual,
                final_residual=res.initial_residual, iterations=res.iterations,
                status=res.status,
                diagnostics={"zero_guard": True, "inner": res.diagnostics},
            )
        return res


class DirectionFallbackGuard(SolverNode):
    """Three-tier degradation policy for extreme conditioning.

    Wraps an adapter (typically one with an LU64 contract-fallback child). If
    the adapter certifies nothing — returns the zero vector because every
    correction, including the parent-precision fallback's, violated the
    acceptance test — this guard hands the enclosing FGMRES the RAW
    parent-precision solver direction instead, uncertified, which is exactly
    the standing of a classical preconditioner application (FGMRES minimizes
    the true residual over realized directions regardless of their local
    quality, paper §2). After `sticky_after` consecutive substitutions the
    guard commits to direction mode for the rest of the solve. Substitutions
    are counted and reported; the adapter's contract record still documents
    every rejection, so degradation is visible, not silent.

    Measured standing (paper § dense-ic): with pivot boosting disabled for
    the dense LU nodes, the certified fallback child always sufficed on the
    tested sweep and this third tier was never exercised; it remains a
    safety net rather than a measured contributor."""

    is_fixed = False

    def __init__(self, inner: SolverNode, direction_solver: SolverNode,
                 sticky_after: int = 1):
        self.inner = inner
        self.direction = direction_solver
        self.sticky_after = sticky_after
        self.dtype = inner.dtype
        self.device = inner.device
        self.name = f"dirguard({inner.name})"
        self.substitutions = 0
        self._consecutive = 0
        self.sticky = False

    def reset(self) -> None:
        self.substitutions = 0
        self._consecutive = 0
        self.sticky = False

    def _direction(self, rhs: torch.Tensor) -> CorrectionResult:
        self.substitutions += 1
        d = self.direction.solve(rhs.clone())
        return CorrectionResult(
            correction=d.correction, initial_residual=d.initial_residual,
            final_residual=d.final_residual, iterations=1,
            status=Status.SUCCESS, diagnostics={"direction_fallback": True})

    def solve(self, rhs: torch.Tensor) -> CorrectionResult:
        if self.sticky:
            return self._direction(rhs)
        res = self.inner.solve(rhs)
        if res.status is not Status.FAILURE and bool((res.correction == 0).all()):
            self._consecutive += 1
            if self._consecutive >= self.sticky_after:
                self.sticky = True
            return self._direction(rhs)
        self._consecutive = 0
        return res


class CountingPreconditioner(SolverNode):
    """Retains every result object returned by the wrapped node.

    Used as the FGMRES-facing wrapper around the outermost CAMPS adapter so the
    harness can walk each application's full AdapterResult tree (child_results
    recursion) after the solve. Forwarding is exact; is_fixed mirrors the
    wrapped node so flexible/fixed admission rules stay intact."""

    def __init__(self, inner: SolverNode):
        self.inner = inner
        self.dtype = inner.dtype
        self.device = inner.device
        self.is_fixed = inner.is_fixed
        self.name = inner.name
        self.records: List[CorrectionResult] = []

    def reset(self) -> None:
        self.records = []

    def solve(self, rhs: torch.Tensor) -> CorrectionResult:
        res = self.inner.solve(rhs)
        self.records.append(res)
        return res
