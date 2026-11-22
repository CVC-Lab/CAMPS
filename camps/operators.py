"""Precision-aware linear operators (spec §2; paper §3).

An operator declares its shape, input/output dtype, device, and optional
symmetry / positive-definiteness metadata, and applies itself to vectors
(matvec) and blocks (matmat). Both explicitly stored matrices and matrix-free
callables are supported. Dtype checks are strict: an operator never silently
casts its input (paper §3, no implicit casts).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import torch

from .precision import spec_of


class LinearOperator(ABC):
    """A_p: F_p^n -> F_p^n (square unless stated otherwise)."""

    shape: Tuple[int, int]
    dtype: torch.dtype
    device: torch.device
    symmetric: Optional[bool] = None
    positive_definite: Optional[bool] = None
    # |A||x| support (certified energy acceptance, paper §4.3): operators that
    # can form the entrywise-absolute product set supports_abs = True.
    supports_abs: bool = False
    # largest number of nonzeros in a row, if known (sharpens the rounding
    # constants of the certified tests; None means "use n").
    max_nnz_per_row: Optional[int] = None

    @abstractmethod
    def matvec(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def matmat(self, X: torch.Tensor) -> torch.Tensor: ...

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatch on ndim: (n,) -> matvec, (n, k) -> matmat."""
        self._check_input(x)
        return self.matvec(x) if x.ndim == 1 else self.matmat(x)

    def apply_abs(self, x: torch.Tensor) -> torch.Tensor:
        """|A| |x| in the operator's format, for the evaluation-error bound of
        the certified energy test (paper §4.3). Executed work: counted and
        timed like any other product. Not available for matrix-free operators."""
        raise NotImplementedError(f"{type(self).__name__} cannot form |A||x|")

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dtype != self.dtype:
            raise TypeError(
                f"operator expects dtype {self.dtype}, got {x.dtype}; CAMPS "
                "forbids implicit casts (paper §3) — convert explicitly via "
                "precision.convert"
            )
        if x.shape[0] != self.shape[1]:
            raise ValueError(f"operator shape {self.shape} incompatible with input {tuple(x.shape)}")

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply(x)


class DenseOperator(LinearOperator):
    """Explicitly stored dense matrix in a single format."""

    def __init__(
        self,
        matrix: torch.Tensor,
        symmetric: Optional[bool] = None,
        positive_definite: Optional[bool] = None,
    ):
        if matrix.ndim != 2:
            raise ValueError("DenseOperator requires a 2-D matrix")
        spec_of(matrix.dtype)  # raises for unsupported formats
        self.matrix = matrix
        self.shape = tuple(matrix.shape)
        self.dtype = matrix.dtype
        self.device = matrix.device
        self.symmetric = symmetric
        self.positive_definite = positive_definite

    supports_abs = True

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.matrix @ x

    def matmat(self, X: torch.Tensor) -> torch.Tensor:
        return self.matrix @ X

    def apply_abs(self, x: torch.Tensor) -> torch.Tensor:
        """|A||x|. The absolute matrix is cached on first use (doubles the
        operator's storage while the energy contract is in use)."""
        self._check_input(x)
        if getattr(self, "_abs_matrix", None) is None:
            self._abs_matrix = self.matrix.abs()
        return self._abs_matrix @ x.abs()

    def to_precision(self, target) -> "DenseOperator":
        """Q_q(A): explicit stored-matrix conversion (paper §3, A_q = Q_q(A)).

        Conversion damage (overflow / structural underflow / zero diagonal) is
        the caller's responsibility to audit via precision.conversion_report —
        rhs scaling cannot repair a damaged A_q (paper §4.1).
        """
        spec = spec_of(target)
        return DenseOperator(
            self.matrix.to(spec.torch_dtype),
            symmetric=self.symmetric,
            positive_definite=self.positive_definite,
        )


class MatrixFreeOperator(LinearOperator):
    """Operator given by callables; must be honest about its dtype/device."""

    def __init__(
        self,
        matvec: Callable[[torch.Tensor], torch.Tensor],
        shape: Tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        matmat: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        symmetric: Optional[bool] = None,
        positive_definite: Optional[bool] = None,
    ):
        spec_of(dtype)
        self._matvec = matvec
        self._matmat = matmat
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.symmetric = symmetric
        self.positive_definite = positive_definite

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self._matvec(x)

    def matmat(self, X: torch.Tensor) -> torch.Tensor:
        if self._matmat is not None:
            return self._matmat(X)
        cols = [self._matvec(X[:, j]) for j in range(X.shape[1])]
        return torch.stack(cols, dim=1)


def as_operator(obj, **kw) -> LinearOperator:
    if isinstance(obj, LinearOperator):
        return obj
    if isinstance(obj, torch.Tensor):
        return DenseOperator(obj, **kw)
    raise TypeError(f"cannot interpret {type(obj)} as LinearOperator")
