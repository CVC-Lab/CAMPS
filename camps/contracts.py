"""Residual contracts (paper §4.2 eq. (1), §4.3 eq. (2), Lemma 1; spec §§7-8).

The runtime checks here verify the contract on *measured* parent residuals for
the specific inputs seen. Passing them is existence evidence for those inputs
only — it is not a proof that the theorem's hypotheses hold for arbitrary
inputs (spec §11, scientific restrictions).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch

from .precision import spec_of

Mode = Literal["norm_nonincrease", "contraction"]


@dataclass
class ResidualContract:
    """Configured per-edge residual requirement.

    norm_nonincrease  — clean contract (1): ||r+|| <= ||r||, checked with a
                        small finite-precision slack (choice I-7), because the
                        measured parent residual is itself rounded.
    contraction       — contract (2): ||r+|| <= eta*||r|| + omega with
                        0 <= eta < 1, omega >= 0, checked verbatim.
    """

    mode: Mode = "norm_nonincrease"
    eta: Optional[float] = None
    omega: float = 0.0
    slack_ulps: float = 8.0          # multiplies parent unit roundoff (I-7)
    atol_slack: float = 0.0          # absolute slack for near-floor residuals

    def __post_init__(self):
        if self.mode == "contraction":
            if self.eta is None or not (0.0 <= self.eta < 1.0):
                raise ValueError("contraction mode requires 0 <= eta < 1 (paper eq. (2))")
            if self.omega < 0.0:
                raise ValueError("omega must be >= 0")

    def check(
        self,
        r_before: torch.Tensor,
        r_after: torch.Tensor,
        parent_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Per-column boolean: does the step satisfy the configured contract?

        r_before / r_after are per-column parent-residual norms (0-d or (k,)).
        """
        u_p = spec_of(parent_dtype).unit_roundoff
        if self.mode == "norm_nonincrease":
            bound = r_before * (1.0 + self.slack_ulps * u_p) + self.atol_slack
        else:
            bound = self.eta * r_before + self.omega
        return r_after <= bound

    def describe(self) -> Dict:
        return {
            "mode": self.mode,
            "eta": self.eta,
            "omega": self.omega,
            "slack_ulps": self.slack_ulps,
            "atol_slack": self.atol_slack,
        }


def lemma1_bound(eta: float, omega: float, J: int, r0: float) -> float:
    """Per-edge geometric residual envelope (paper Lemma 1).

    ||r_J|| <= eta^J ||r_0|| + (1 - eta^J)/(1 - eta) * omega.

    Valid under the hypothesis that *every* one of the J corrections satisfies
    contract (2) with the same (eta, omega); the lemma is conditional and per
    edge (spec §8, §11).
    """
    if not (0.0 <= eta < 1.0):
        raise ValueError("Lemma 1 requires 0 <= eta < 1")
    geom = (1.0 - eta**J) / (1.0 - eta) if eta > 0 else float(min(J, 1) and 1.0)
    if eta == 0.0:
        geom = 1.0 if J >= 1 else 0.0
    return (eta**J) * r0 + geom * omega


def check_zero_start(solver) -> bool:
    """True iff the solver exposes the CAMPS zero-start interface (spec §2).

    A CAMPS correction solver takes only a right-hand side; an initial-guess
    parameter on the solving interface marks it incompatible (paper §4.2: a
    child started from a nonzero guess reports the residual of a shifted
    subproblem). Used by the adapter at construction time.
    """
    import inspect

    sig = inspect.signature(solver.solve)
    forbidden = {"x0", "initial_guess", "guess", "warm_start"}
    return not (forbidden & set(sig.parameters))
