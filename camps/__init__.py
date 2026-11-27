"""CAMPS: Composable Adapters for Mixed-Precision Solvers.

Reference implementation of the framework in Kevin Li and Chandrajit Bajaj,
"CAMPS: Composable Adapters for Mixed-Precision Solvers" (arXiv:2611.33703, 2026).
Comments of the form "spec §N" refer to an internal design specification that
is not part of this repository. experiments/tacc/README.md maps the paper's
algorithms and sections to the code.
"""
from .adapter import AdapterConfig, CampsAdapter
from .base_solvers import (
    ApproxInverseSolver,
    CholeskySolver,
    DenseLUSolver,
    IdentitySolver,
    JacobiSolver,
    SolverNode,
)
from .cg import CGSolver
from .contracts import ResidualContract, check_zero_start, lemma1_bound
from .diagnostics import configure_backend, probe_matmul_accumulation, promotion_report
from .fgmres import FGMRESSolver
from .gmres import GMRESSolver
from .operators import DenseOperator, LinearOperator, MatrixFreeOperator, as_operator
from .precision import BF16, FP16, FP32, FP64, PrecisionSpec, colnorms, conversion_report, convert, spec_of
from .results import AdapterResult, CorrectionResult, Status
from .scaling import apply_scale, choose_pow2_scale, unapply_scale

__all__ = [
    "AdapterConfig", "CampsAdapter", "SolverNode",
    "ApproxInverseSolver", "CholeskySolver", "DenseLUSolver", "IdentitySolver",
    "JacobiSolver", "CGSolver", "GMRESSolver", "FGMRESSolver",
    "ResidualContract", "check_zero_start", "lemma1_bound",
    "configure_backend", "probe_matmul_accumulation", "promotion_report",
    "DenseOperator", "LinearOperator", "MatrixFreeOperator", "as_operator",
    "PrecisionSpec", "FP64", "FP32", "FP16", "BF16",
    "spec_of", "convert", "conversion_report", "colnorms",
    "AdapterResult", "CorrectionResult", "Status",
    "apply_scale", "choose_pow2_scale", "unapply_scale",
]
