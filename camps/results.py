"""Structured results for CAMPS correction solves (spec §4).

Every correction solver returns a CorrectionResult; every adapter application
returns an AdapterResult, which additionally retains the authoritative
parent-residual history of every correction (spec §15, prompt requirement:
"Store the residual history from every adapter application").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch


class Status(Enum):
    """Terminal statuses of a CAMPS solve (paper Alg. 1; spec §9)."""

    SUCCESS = "success"        # parent stopping criterion met (Alg. 1 step 4)
    MAX_STEPS = "max_steps"    # J corrections exhausted (Alg. 1 step 10)
    STAGNATED = "stagnated"    # h stagnant checks (Alg. 1 step 9)
    FAILURE = "failure"        # nonfinite child output / numerically zero
                               # converted residual (Alg. 1 step 7)
    REJECTED = "rejected"      # contract-violating correction rolled back
                               # (implementation extension, spec §6 / choice I-6)


def _to_scalar_or_list(t: torch.Tensor):
    if t.ndim == 0:
        return float(t)
    return [float(v) for v in t]


@dataclass
class CorrectionResult:
    """Result of one zero-start correction solve (spec §4).

    Residual fields are per-column tensors for multi-RHS inputs (choice I-1:
    the contract is enforced per column).
    """

    correction: torch.Tensor
    initial_residual: torch.Tensor       # ||r||_2 of the rhs handed to the solver
    final_residual: torch.Tensor         # ||r - A d||_2 measured by the solver
    iterations: int
    status: Status
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def relative_residual(self) -> torch.Tensor:
        init = self.initial_residual
        out = torch.where(
            init > 0,
            self.final_residual / torch.clamp(init, min=torch.finfo(init.dtype).tiny),
            torch.zeros_like(self.final_residual),
        )
        return out

    @property
    def converged(self) -> bool:
        return self.status is Status.SUCCESS

    def summary(self) -> str:
        return (
            f"{self.status.value}: iters={self.iterations} "
            f"init={_to_scalar_or_list(self.initial_residual)} "
            f"final={_to_scalar_or_list(self.final_residual)}"
        )


@dataclass
class AdapterResult(CorrectionResult):
    """Result of one adapter application (spec §§4, 6).

    residual_history[t] is the authoritative parent-precision residual norm
    (per column) *before* correction t; the last entry is the final recomputed
    residual. scales[t] is the per-column power-of-two scale s_t. accepted[t]
    records the outcome of the acceptance test (choice I-6). child_results
    holds each child CorrectionResult verbatim.
    """

    residual_history: List[torch.Tensor] = field(default_factory=list)
    child_results: List[CorrectionResult] = field(default_factory=list)
    scales: List[torch.Tensor] = field(default_factory=list)
    accepted: List[bool] = field(default_factory=list)
    contract_record: Dict[str, Any] = field(default_factory=dict)
