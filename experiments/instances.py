"""CAMPS instances and baseline solvers per matrix class.

Every builder takes a Problem and returns a BuiltMethod whose `solver` accepts
the FP64 right-hand side. Builders perform ALL their own precision casts and
factorizations inside build() so the harness can time setup honestly.

Instance naming follows the paper's grammar, e.g.
    FGMRES64[IR64<-32[LU32]]          one-edge dense instance
    FGMRES64[IR64<-32[FGMRES32[IR32<-16[INV16]]]]   two-edge recursive instance

Baselines are the like-for-like FP64 methods the paper demands (§6):
uniform-precision direct solve, FP64-preconditioned FGMRES *and* GMRES
(fixed-preconditioner control), and plain outer IR (adapter alone, the
classical mixed-precision IR of Buttari et al.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from camps.adapter import AdapterConfig, CampsAdapter
from camps.base_solvers import ApproxInverseSolver, DenseLUSolver, SolverNode
from camps.cg import CGSolver
from camps.fgmres import FGMRESSolver
from camps.gmres import GMRESSolver
from camps.operators import DenseOperator
from camps.contracts import ResidualContract

from .matrices import Problem
from .nodes import (CastBridge, CountingDenseOperator, CountingOperator,
                    CountingPreconditioner, DiagJacobiSolver,
                    DirectionFallbackGuard, RichardsonSolver,
                    SparseCSROperator, ZeroGuardPreconditioner)


@dataclass
class BuiltMethod:
    name: str
    solver: SolverNode
    registry: Dict[str, Dict[str, int]]          # per-precision matvec counts
    wrapper: Optional[CountingPreconditioner] = None   # outer adapter records
    zero_guard: Optional[ZeroGuardPreconditioner] = None
    analytic_bytes: List[Tuple[str, int]] = field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------

def _dense_op(problem: Problem, dtype: torch.dtype, reg: Dict) -> CountingDenseOperator:
    A = problem.dense(dtype)
    sym = problem.meta.get("symmetric")
    return CountingDenseOperator(A, reg, symmetric=sym, positive_definite=sym)


def _sparse_op(problem: Problem, dtype: torch.dtype, reg: Dict) -> CountingOperator:
    crow, col, vals, diag = problem.csr_parts(dtype)
    spd = problem.meta.get("spd")
    return CountingOperator(
        SparseCSROperator(crow, col, vals, problem.n,
                          symmetric=spd, positive_definite=spd, diagonal=diag), reg)


def _op(problem: Problem, dtype: torch.dtype, reg: Dict) -> CountingOperator:
    return (_dense_op if problem.kind == "dense" else _sparse_op)(problem, dtype, reg)


def _dense_bytes(problem: Problem, dtype: torch.dtype) -> int:
    return problem.n * problem.n * torch.tensor([], dtype=dtype).element_size()


def _sparse_value_bytes(problem: Problem, dtype: torch.dtype) -> int:
    return problem.meta["nnz"] * torch.tensor([], dtype=dtype).element_size()


def _basis_bytes(n: int, restart: int, dtype: torch.dtype, flexible: bool) -> int:
    esz = torch.tensor([], dtype=dtype).element_size()
    vecs = (restart + 1) + (restart if flexible else 0)
    return vecs * n * esz


# --------------------------------------------------------------------------
# dense baselines
# --------------------------------------------------------------------------

def build_lu64_direct(problem: Problem, lu_pivot_tol: float = 0.0, **_) -> BuiltMethod:
    reg: Dict = {}
    op = _dense_op(problem, torch.float64, reg)
    solver = DenseLUSolver(op, pivot_tol=lu_pivot_tol)
    return BuiltMethod(
        name="LU64 (direct)", solver=solver, registry=reg,
        analytic_bytes=[("A64", _dense_bytes(problem, torch.float64)),
                        ("LU64", _dense_bytes(problem, torch.float64))],
    )


def build_fgmres64_lu64(problem: Problem, tol: float, restart: int,
                        maxiter: int, lu_pivot_tol: float = 0.0, **_) -> BuiltMethod:
    reg: Dict = {}
    op = _dense_op(problem, torch.float64, reg)
    pre = DenseLUSolver(op, pivot_tol=lu_pivot_tol)
    solver = FGMRESSolver(op, tol=tol, restart=restart, maxiter=maxiter,
                          preconditioner=pre)
    return BuiltMethod(
        name="FGMRES64(LU64)", solver=solver, registry=reg,
        analytic_bytes=[("A64", _dense_bytes(problem, torch.float64)),
                        ("LU64", _dense_bytes(problem, torch.float64)),
                        ("basis64", _basis_bytes(problem.n, restart, torch.float64, True))],
    )


def build_plain_ir_lu32(problem: Problem, tol: float, ir_max: int = 60,
                        damping: str = "none", **_) -> BuiltMethod:
    """Classical mixed-precision IR: the adapter alone, no outer Krylov."""
    reg: Dict = {}
    op64 = _dense_op(problem, torch.float64, reg)
    op32 = _dense_op(problem, torch.float32, reg)
    child = DenseLUSolver(op32)
    cfg = AdapterConfig(J=ir_max, tau=tol, scaling="pow2", stag_h=3,
                        damping=damping)
    solver = CampsAdapter(op64, child, cfg)
    return BuiltMethod(
        name="IR64<-32[LU32] (plain IR)" + (" damped" if damping != "none" else ""), solver=solver, registry=reg,
        analytic_bytes=[("A64", _dense_bytes(problem, torch.float64)),
                        ("A32", _dense_bytes(problem, torch.float32)),
                        ("LU32", _dense_bytes(problem, torch.float32))],
    )


def build_plain_ir_lu32_fallback(problem: Problem, tol: float, ir_max: int = 60,
                                 lu_pivot_tol: float = 0.0, damping: str = "none",
                                 rho: Optional[float] = None, **_) -> BuiltMethod:
    """IR64<-32[LU32] with a parent-precision fallback (spec §6,
    on_violation='fallback'): when the FP32 child stops satisfying the
    contract, the adapter rolls the correction back and switches to an LU64
    child for the remaining steps. The robust instance for unknown
    conditioning — it pays the FP64 factorization in setup but only *uses*
    it when the contract detects a non-contractive child."""
    reg: Dict = {}
    op64 = _dense_op(problem, torch.float64, reg)
    op32 = _dense_op(problem, torch.float32, reg)
    child = DenseLUSolver(op32)
    fallback = DenseLUSolver(op64, pivot_tol=lu_pivot_tol)
    contract = (ResidualContract(mode="contraction", eta=rho, omega=0.0)
                if rho is not None else ResidualContract())
    cfg = AdapterConfig(J=ir_max, tau=tol, scaling="pow2", stag_h=3,
                        on_violation="fallback", fallback_solver=fallback,
                        damping=damping, contract=contract)
    solver = CampsAdapter(op64, child, cfg)
    return BuiltMethod(
        name=("IR64<-32[LU32 | LU64 fallback]" + (" damped" if damping != "none" else "")
              + (f" rho={rho:g}" if rho is not None else "")), solver=solver, registry=reg,
        analytic_bytes=[("A64", _dense_bytes(problem, torch.float64)),
                        ("A32", _dense_bytes(problem, torch.float32)),
                        ("LU32", _dense_bytes(problem, torch.float32)),
                        ("LU64", _dense_bytes(problem, torch.float64))],
        notes="fallback factor is built eagerly in setup; a lazy build would "
              "defer that cost to first contract violation",
    )


def build_camps_ladder(problem: Problem, tol: float, restart: int = 30,
                       maxiter: int = 400, J: int = 2, damping: str = "none",
                       lu_pivot_tol: float = 0.0, **_) -> BuiltMethod:
    """The degradation-ladder instance:
    FGMRES64[ IR64<-32[LU32 | LU64 contract-fallback] | LU64 direction ].

    One configuration for the whole conditioning range: certified FP32
    corrections while the contract holds; certified FP64 corrections after
    the first measured violation; raw (uncertified) FP64 preconditioner
    directions when even the parent-precision child cannot satisfy the
    zero-start residual-reducing contract — which provably happens on
    unit-norm Krylov vectors once n*u64*||A||*||A^{-1}v|| exceeds 1."""
    reg: Dict = {}
    op64 = _dense_op(problem, torch.float64, reg)
    op32 = _dense_op(problem, torch.float32, reg)
    lu64 = DenseLUSolver(op64, pivot_tol=lu_pivot_tol)
    cfg = AdapterConfig(J=J, tau=1e-8, scaling="pow2",
                        on_violation="fallback", fallback_solver=lu64,
                        damping=damping)
    adapter = CampsAdapter(op64, DenseLUSolver(op32), cfg)
    wrapper = CountingPreconditioner(adapter)
    guard = DirectionFallbackGuard(wrapper, lu64, sticky_after=1)
    solver = FGMRESSolver(op64, tol=tol, restart=restart, maxiter=maxiter,
                          preconditioner=guard)
    bm = BuiltMethod(
        name="FGMRES64[IR64<-32[LU32|LU64 fb]|dir(LU64)] (ladder)"
             + (" damped" if damping != "none" else ""),
        solver=solver, registry=reg, wrapper=wrapper,
        analytic_bytes=[("A64", _dense_bytes(problem, torch.float64)),
                        ("A32", _dense_bytes(problem, torch.float32)),
                        ("LU32", _dense_bytes(problem, torch.float32)),
                        ("LU64", _dense_bytes(problem, torch.float64)),
                        ("basis64", _basis_bytes(problem.n, restart, torch.float64, True))],
        notes="three-tier degradation: certified fp32 -> certified fp64 -> "
              "uncertified fp64 direction; substitutions counted",
    )
    bm.zero_guard = guard  # harness reset/reporting hook (counts substitutions)
    return bm


# --------------------------------------------------------------------------
# dense CAMPS instances
# --------------------------------------------------------------------------

def build_camps_lu32(problem: Problem, tol: float, restart: int, maxiter: int,
                     J: int = 3, adapter_tau: float = 1e-8,
                     loose_contract: bool = False, damping: str = "none",
                     **_) -> BuiltMethod:
    """FGMRES64[IR64<-32[LU32]]: the paper's instance shape on a dense operator.

    loose_contract=True configures the acceptance test for the PRECONDITIONER
    role: a correction is accepted unless it grows the residual by more than
    100x. Rationale: inside FGMRES a returned correction is a *direction*, not
    a commitment — the outer method minimizes the true residual over the span
    of realized directions, so a low-precision solve whose standalone residual
    grows (kappa * u32 > 1) can still be the GMRES-IR-quality preconditioner
    direction that classical analysis relies on. The strict variant keeps the
    clean contract (1); comparing both isolates what the acceptance extension
    costs in the ill-conditioned regime."""
    reg: Dict = {}
    op64 = _dense_op(problem, torch.float64, reg)
    op32 = _dense_op(problem, torch.float32, reg)
    child = DenseLUSolver(op32)
    if loose_contract:
        # bound = r_before * (1 + slack_ulps * u64); u64 = 2^-53
        contract = ResidualContract(slack_ulps=99.0 / 2.0**-53)
        cfg = AdapterConfig(J=J, tau=adapter_tau, scaling="pow2", contract=contract,
                            damping=damping)
    else:
        cfg = AdapterConfig(J=J, tau=adapter_tau, scaling="pow2",
                            damping=damping)
    adapter = CampsAdapter(op64, child, cfg)
    wrapper = CountingPreconditioner(adapter)
    guard = ZeroGuardPreconditioner(wrapper)
    solver = FGMRESSolver(op64, tol=tol, restart=restart, maxiter=maxiter,
                          preconditioner=guard)
    label = ("FGMRES64[IR64<-32[LU32]]" + (" (loose)" if loose_contract else "")
             + (" damped" if damping != "none" else ""))
    return BuiltMethod(
        name=label, solver=solver, registry=reg,
        wrapper=wrapper, zero_guard=guard,
        analytic_bytes=[("A64", _dense_bytes(problem, torch.float64)),
                        ("A32", _dense_bytes(problem, torch.float32)),
                        ("LU32", _dense_bytes(problem, torch.float32)),
                        ("basis64", _basis_bytes(problem.n, restart, torch.float64, True))],
    )


def _inv_child(problem: Problem, dtype: torch.dtype, reg: Dict) -> ApproxInverseSolver:
    """Approximate inverse computed wide, stored/applied narrow (tensor-core
    matmul child for half formats; the honest fp16 'factorization substitute'
    given that cuSOLVER has no half LU — capability probe)."""
    A64 = problem.dense(torch.float64)
    B64 = torch.linalg.inv(A64)
    B = B64.to(dtype)
    op = _dense_op(problem, dtype, reg)
    return ApproxInverseSolver(B, op, label="inv")


def build_camps_hierarchy_dense(problem: Problem, tol: float, restart: int,
                                maxiter: int, half_dtype: torch.dtype = torch.float16,
                                J_outer: int = 2, J_inner: int = 2,
                                inner_tol: float = 1e-4, inner_restart: int = 10,
                                inner_maxiter: int = 20, **_) -> BuiltMethod:
    """FGMRES64[IR64<-32[FGMRES32[IR32<-16[INV16]]]] — the full recursive
    hierarchy: refinement inserted at BOTH precision edges (Theorem 1)."""
    reg: Dict = {}
    hname = "16" if half_dtype == torch.float16 else "b16"
    op64 = _dense_op(problem, torch.float64, reg)
    op32 = _dense_op(problem, torch.float32, reg)
    child16 = _inv_child(problem, half_dtype, reg)
    inner_cfg = AdapterConfig(J=J_inner, tau=inner_tol, scaling="pow2")
    ir32_16 = CampsAdapter(op32, child16, inner_cfg)
    guard32 = ZeroGuardPreconditioner(ir32_16)
    fg32 = FGMRESSolver(op32, tol=inner_tol, restart=inner_restart,
                        maxiter=inner_maxiter, preconditioner=guard32)
    outer_cfg = AdapterConfig(J=J_outer, tau=1e-8, scaling="pow2")
    ir64_32 = CampsAdapter(op64, fg32, outer_cfg)
    wrapper = CountingPreconditioner(ir64_32)
    guard = ZeroGuardPreconditioner(wrapper)
    solver = FGMRESSolver(op64, tol=tol, restart=restart, maxiter=maxiter,
                          preconditioner=guard)
    return BuiltMethod(
        name=f"FGMRES64[IR64<-32[FGMRES32[IR32<-{hname}[INV{hname}]]]]",
        solver=solver, registry=reg, wrapper=wrapper, zero_guard=guard,
        analytic_bytes=[("A64", _dense_bytes(problem, torch.float64)),
                        ("A32", _dense_bytes(problem, torch.float32)),
                        (f"A{hname}", _dense_bytes(problem, half_dtype)),
                        (f"B{hname}", _dense_bytes(problem, half_dtype)),
                        ("basis64", _basis_bytes(problem.n, restart, torch.float64, True)),
                        ("basis32", _basis_bytes(problem.n, inner_restart, torch.float32, True))],
        notes="inverse computed in FP64 during setup, stored+applied in half "
              "(tensor-core matmul, FP32 accumulate; no half LU exists in cuSOLVER)",
    )


# ---- placement ablation: same child family, different edge sets ----------

def build_camps_inv32(problem: Problem, tol: float, restart: int, maxiter: int,
                      J: int = 3, **_) -> BuiltMethod:
    """A: FGMRES64[IR64<-32[INV32]] — one edge, FP32 child."""
    reg: Dict = {}
    op64 = _dense_op(problem, torch.float64, reg)
    child = _inv_child(problem, torch.float32, reg)
    adapter = CampsAdapter(op64, child, AdapterConfig(J=J, tau=1e-8, scaling="pow2"))
    wrapper = CountingPreconditioner(adapter)
    guard = ZeroGuardPreconditioner(wrapper)
    solver = FGMRESSolver(op64, tol=tol, restart=restart, maxiter=maxiter,
                          preconditioner=guard)
    return BuiltMethod(
        name="FGMRES64[IR64<-32[INV32]]", solver=solver, registry=reg,
        wrapper=wrapper, zero_guard=guard,
        analytic_bytes=[("A64", _dense_bytes(problem, torch.float64)),
                        ("A32", _dense_bytes(problem, torch.float32)),
                        ("B32", _dense_bytes(problem, torch.float32)),
                        ("basis64", _basis_bytes(problem.n, restart, torch.float64, True))],
    )


def build_camps_inv16_direct(problem: Problem, tol: float, restart: int,
                             maxiter: int, J: int = 3,
                             half_dtype: torch.dtype = torch.float16, **_) -> BuiltMethod:
    """B: FGMRES64[IR64<-16[INV16]] — one edge straight across two format levels."""
    reg: Dict = {}
    hname = "16" if half_dtype == torch.float16 else "b16"
    op64 = _dense_op(problem, torch.float64, reg)
    child = _inv_child(problem, half_dtype, reg)
    adapter = CampsAdapter(op64, child, AdapterConfig(J=J, tau=1e-8, scaling="pow2"))
    wrapper = CountingPreconditioner(adapter)
    guard = ZeroGuardPreconditioner(wrapper)
    solver = FGMRESSolver(op64, tol=tol, restart=restart, maxiter=maxiter,
                          preconditioner=guard)
    return BuiltMethod(
        name=f"FGMRES64[IR64<-{hname}[INV{hname}]]", solver=solver, registry=reg,
        wrapper=wrapper, zero_guard=guard,
        analytic_bytes=[("A64", _dense_bytes(problem, torch.float64)),
                        (f"A{hname}", _dense_bytes(problem, half_dtype)),
                        (f"B{hname}", _dense_bytes(problem, half_dtype)),
                        ("basis64", _basis_bytes(problem.n, restart, torch.float64, True))],
    )


def build_camps_nested_ir16(problem: Problem, tol: float, restart: int,
                            maxiter: int, J_outer: int = 2, J_inner: int = 2,
                            half_dtype: torch.dtype = torch.float16, **_) -> BuiltMethod:
    """C: FGMRES64[IR64<-32[IR32<-16[INV16]]] — two nested adapters, no inner
    Krylov: the minimal multi-edge composition of Theorem 1."""
    reg: Dict = {}
    hname = "16" if half_dtype == torch.float16 else "b16"
    op64 = _dense_op(problem, torch.float64, reg)
    op32 = _dense_op(problem, torch.float32, reg)
    child16 = _inv_child(problem, half_dtype, reg)
    ir32_16 = CampsAdapter(op32, child16,
                           AdapterConfig(J=J_inner, tau=1e-6, scaling="pow2"))
    ir64_32 = CampsAdapter(op64, ir32_16,
                           AdapterConfig(J=J_outer, tau=1e-8, scaling="pow2"))
    wrapper = CountingPreconditioner(ir64_32)
    guard = ZeroGuardPreconditioner(wrapper)
    solver = FGMRESSolver(op64, tol=tol, restart=restart, maxiter=maxiter,
                          preconditioner=guard)
    return BuiltMethod(
        name=f"FGMRES64[IR64<-32[IR32<-{hname}[INV{hname}]]]",
        solver=solver, registry=reg, wrapper=wrapper, zero_guard=guard,
        analytic_bytes=[("A64", _dense_bytes(problem, torch.float64)),
                        ("A32", _dense_bytes(problem, torch.float32)),
                        (f"A{hname}", _dense_bytes(problem, half_dtype)),
                        (f"B{hname}", _dense_bytes(problem, half_dtype)),
                        ("basis64", _basis_bytes(problem.n, restart, torch.float64, True))],
    )


# --------------------------------------------------------------------------
# sparse baselines
# --------------------------------------------------------------------------

def build_cg64(problem: Problem, tol: float, maxiter: int,
               jacobi: bool = True, **_) -> BuiltMethod:
    reg: Dict = {}
    op = _sparse_op(problem, torch.float64, reg)
    pre = DiagJacobiSolver(op, op.diagonal) if jacobi else None
    solver = CGSolver(op, tol=tol, maxiter=maxiter, preconditioner=pre)
    label = "CG64(Jacobi64)" if jacobi else "CG64"
    return BuiltMethod(
        name=label, solver=solver, registry=reg,
        analytic_bytes=[("A64 vals", _sparse_value_bytes(problem, torch.float64)),
                        ("work64", 4 * problem.n * 8)],
    )


def build_fgmres64_jacobi(problem: Problem, tol: float, restart: int,
                          maxiter: int, **_) -> BuiltMethod:
    reg: Dict = {}
    op = _sparse_op(problem, torch.float64, reg)
    pre = DiagJacobiSolver(op, op.diagonal)
    solver = FGMRESSolver(op, tol=tol, restart=restart, maxiter=maxiter,
                          preconditioner=pre)
    return BuiltMethod(
        name="FGMRES64(Jacobi64)", solver=solver, registry=reg,
        analytic_bytes=[("A64 vals", _sparse_value_bytes(problem, torch.float64)),
                        ("basis64", _basis_bytes(problem.n, restart, torch.float64, True))],
    )


def build_gmres64_jacobi(problem: Problem, tol: float, restart: int,
                         maxiter: int, **_) -> BuiltMethod:
    reg: Dict = {}
    op = _sparse_op(problem, torch.float64, reg)
    pre = DiagJacobiSolver(op, op.diagonal)
    solver = GMRESSolver(op, tol=tol, restart=restart, maxiter=maxiter,
                         preconditioner=pre)
    return BuiltMethod(
        name="GMRES64(Jacobi64)", solver=solver, registry=reg,
        analytic_bytes=[("A64 vals", _sparse_value_bytes(problem, torch.float64)),
                        ("basis64", _basis_bytes(problem.n, restart, torch.float64, False))],
    )


# --------------------------------------------------------------------------
# sparse CAMPS instances
# --------------------------------------------------------------------------

def build_camps_cg32(problem: Problem, tol: float, restart: int, maxiter: int,
                     J: int = 2, inner_tol: float = 1e-4,
                     inner_maxiter: int = 200, **_) -> BuiltMethod:
    """FGMRES64[IR64<-32[CG32(Jacobi32)]] for SPD sparse systems."""
    reg: Dict = {}
    op64 = _sparse_op(problem, torch.float64, reg)
    op32 = _sparse_op(problem, torch.float32, reg)
    pre32 = DiagJacobiSolver(op32, op32.diagonal)
    cg32 = CGSolver(op32, tol=inner_tol, maxiter=inner_maxiter, preconditioner=pre32)
    adapter = CampsAdapter(op64, cg32, AdapterConfig(J=J, tau=1e-8, scaling="pow2"))
    wrapper = CountingPreconditioner(adapter)
    guard = ZeroGuardPreconditioner(wrapper)
    solver = FGMRESSolver(op64, tol=tol, restart=restart, maxiter=maxiter,
                          preconditioner=guard)
    return BuiltMethod(
        name="FGMRES64[IR64<-32[CG32(Jacobi32)]]", solver=solver, registry=reg,
        wrapper=wrapper, zero_guard=guard,
        analytic_bytes=[("A64 vals", _sparse_value_bytes(problem, torch.float64)),
                        ("A32 vals", _sparse_value_bytes(problem, torch.float32)),
                        ("basis64", _basis_bytes(problem.n, restart, torch.float64, True))],
    )


def build_plain_ir_cg32(problem: Problem, tol: float, ir_max: int = 60,
                        inner_tol: float = 1e-4, inner_maxiter: int = 200,
                        acceptance: str = "residual", **_) -> BuiltMethod:
    """Outer-IR ablation partner of build_camps_cg32 (no outer Krylov)."""
    reg: Dict = {}
    op64 = _sparse_op(problem, torch.float64, reg)
    op32 = _sparse_op(problem, torch.float32, reg)
    pre32 = DiagJacobiSolver(op32, op32.diagonal)
    cg32 = CGSolver(op32, tol=inner_tol, maxiter=inner_maxiter, preconditioner=pre32)
    cfg = AdapterConfig(J=ir_max, tau=tol, scaling="pow2", stag_h=3,
                        acceptance=acceptance)
    solver = CampsAdapter(op64, cg32, cfg)
    return BuiltMethod(
        name="IR64<-32[CG32(Jacobi32)] (plain IR)" + (" energy" if acceptance == "energy" else ""), solver=solver, registry=reg,
        analytic_bytes=[("A64 vals", _sparse_value_bytes(problem, torch.float64)),
                        ("A32 vals", _sparse_value_bytes(problem, torch.float32))],
    )


def build_camps_fgmres32(problem: Problem, tol: float, restart: int, maxiter: int,
                         J: int = 2, inner_tol: float = 1e-4,
                         inner_restart: int = 30, inner_maxiter: int = 200, **_) -> BuiltMethod:
    """FGMRES64[IR64<-32[FGMRES32(Jacobi32)]] for nonsymmetric sparse systems."""
    reg: Dict = {}
    op64 = _sparse_op(problem, torch.float64, reg)
    op32 = _sparse_op(problem, torch.float32, reg)
    pre32 = DiagJacobiSolver(op32, op32.diagonal)
    fg32 = FGMRESSolver(op32, tol=inner_tol, restart=inner_restart,
                        maxiter=inner_maxiter, preconditioner=pre32)
    adapter = CampsAdapter(op64, fg32, AdapterConfig(J=J, tau=1e-8, scaling="pow2"))
    wrapper = CountingPreconditioner(adapter)
    guard = ZeroGuardPreconditioner(wrapper)
    solver = FGMRESSolver(op64, tol=tol, restart=restart, maxiter=maxiter,
                          preconditioner=guard)
    return BuiltMethod(
        name="FGMRES64[IR64<-32[FGMRES32(Jacobi32)]]", solver=solver, registry=reg,
        wrapper=wrapper, zero_guard=guard,
        analytic_bytes=[("A64 vals", _sparse_value_bytes(problem, torch.float64)),
                        ("A32 vals", _sparse_value_bytes(problem, torch.float32)),
                        ("basis64", _basis_bytes(problem.n, restart, torch.float64, True)),
                        ("basis32", _basis_bytes(problem.n, inner_restart, torch.float32, True))],
    )


def build_f3r(problem: Problem, tol: float, restart: int, maxiter: int,
              m2: int = 20, m3: int = 10, sweeps: int = 2, **_) -> BuiltMethod:
    """Same-harness structural reimplementation of F3R (Suzuki & Iwashita [9]):

        FGMRES64[cast[FGMRES32[cast[FGMRES16[Richardson16(Jacobi)]]]]]

    Three FGMRES levels over FP64/FP32/FP16 plus an innermost fixed-count
    Richardson sweep. Inner levels run FIXED iteration budgets (m2, m3) with
    no local stopping test; convergence is checked only in the outermost
    FGMRES, and precision boundaries are plain casts (no residual formation,
    no scaling, no acceptance) — the published F3R design, in contrast to
    CAMPS's independently stopping adapters. Same kernels, operators, Jacobi
    data, and stopping rule as every other method in this study; this is a
    reimplementation of the solver structure, not the authors' code."""
    reg: Dict = {}
    op64 = _sparse_op(problem, torch.float64, reg)
    op32 = _sparse_op(problem, torch.float32, reg)
    op16 = _sparse_op(problem, torch.float16, reg)
    rich16 = RichardsonSolver(op16, op16.diagonal, sweeps=sweeps)
    fg16 = FGMRESSolver(op16, tol=0.0, atol=0.0, restart=m3, maxiter=m3,
                        preconditioner=rich16)
    fg32 = FGMRESSolver(op32, tol=0.0, atol=0.0, restart=m2, maxiter=m2,
                        preconditioner=CastBridge(torch.float32, fg16))
    bridge = CountingPreconditioner(CastBridge(torch.float64, fg32))
    solver = FGMRESSolver(op64, tol=tol, restart=restart, maxiter=maxiter,
                          preconditioner=bridge)
    return BuiltMethod(
        name=f"F3R: FGMRES64[FGMRES32^{m2}[FGMRES16^{m3}[Rich16x{sweeps}]]]",
        solver=solver, registry=reg, wrapper=bridge,
        analytic_bytes=[("A64 vals", _sparse_value_bytes(problem, torch.float64)),
                        ("A32 vals", _sparse_value_bytes(problem, torch.float32)),
                        ("A16 vals", _sparse_value_bytes(problem, torch.float16)),
                        ("basis64", _basis_bytes(problem.n, restart, torch.float64, True)),
                        ("basis32", _basis_bytes(problem.n, m2, torch.float32, True)),
                        ("basis16", _basis_bytes(problem.n, m3, torch.float16, True))],
        notes="structural F3R reimplementation in this harness; fixed inner "
              "budgets, outermost-only convergence checks, plain-cast boundaries",
    )


def build_camps_hybrid_cast(problem: Problem, tol: float, restart: int,
                            maxiter: int, m3: int = 5, mid_maxiter: int = 15,
                            mid_tol: float = 1e-4, J: int = 2, **_) -> BuiltMethod:
    """FGMRES64[IR64<-32[FGMRES32[cast32<-16[FGMRES16(Jacobi16)]]]] — a MIXED
    tree: CAMPS adapter at the outer precision edge, F3R-style plain cast at
    the inner one. Theorem 1 permits refinement at any compatible SUBSET of
    edges; this instance takes E = {outer edge only} and quantifies the
    per-edge cost of the contract against build_f3r (no instrumented edges)
    and build_camps_sparse_hierarchy (both edges instrumented)."""
    reg: Dict = {}
    op64 = _sparse_op(problem, torch.float64, reg)
    op32 = _sparse_op(problem, torch.float32, reg)
    op16 = _sparse_op(problem, torch.float16, reg)
    pre16 = DiagJacobiSolver(op16, op16.diagonal)
    fg16 = FGMRESSolver(op16, tol=0.0, atol=0.0, restart=m3, maxiter=m3,
                        preconditioner=pre16)
    fg32 = FGMRESSolver(op32, tol=mid_tol, restart=mid_maxiter,
                        maxiter=mid_maxiter,
                        preconditioner=CastBridge(torch.float32, fg16))
    ir64_32 = CampsAdapter(op64, fg32, AdapterConfig(J=J, tau=1e-8, scaling="pow2"))
    wrapper = CountingPreconditioner(ir64_32)
    guard = ZeroGuardPreconditioner(wrapper)
    solver = FGMRESSolver(op64, tol=tol, restart=restart, maxiter=maxiter,
                          preconditioner=guard)
    return BuiltMethod(
        name=f"hybrid: FGMRES64[IR64<-32[FGMRES32[cast32<-16[FGMRES16^{m3}]]]]",
        solver=solver, registry=reg, wrapper=wrapper, zero_guard=guard,
        analytic_bytes=[("A64 vals", _sparse_value_bytes(problem, torch.float64)),
                        ("A32 vals", _sparse_value_bytes(problem, torch.float32)),
                        ("A16 vals", _sparse_value_bytes(problem, torch.float16)),
                        ("basis64", _basis_bytes(problem.n, restart, torch.float64, True)),
                        ("basis32", _basis_bytes(problem.n, mid_maxiter, torch.float32, True)),
                        ("basis16", _basis_bytes(problem.n, m3, torch.float16, True))],
        notes="adapter at outer edge only; inner boundary is an F3R-style cast",
    )


def build_camps_ir_f3r_body(problem: Problem, tol: float, m2: int = 10,
                            m3: int = 5, sweeps: int = 2, ir_max: int = 12,
                            **_) -> BuiltMethod:
    """IR64<-32[FGMRES32^m2[cast32<-16[FGMRES16^m3[Rich16 x sweeps]]]] —
    the F3R body with its outermost FGMRES64 replaced by a plain CAMPS
    adapter. The adapter's per-correction work (parent residual, scale,
    cast, acceptance) is CHEAPER than the FP64 Arnoldi + basis storage +
    orthogonalization it replaces, and it needs no FP64 Krylov basis at
    all, while adding contract supervision (measured contraction,
    stagnation detection, attributed rejection) at the top edge."""
    reg: Dict = {}
    op64 = _sparse_op(problem, torch.float64, reg)
    op32 = _sparse_op(problem, torch.float32, reg)
    op16 = _sparse_op(problem, torch.float16, reg)
    rich16 = RichardsonSolver(op16, op16.diagonal, sweeps=sweeps)
    fg16 = FGMRESSolver(op16, tol=0.0, atol=0.0, restart=m3, maxiter=m3,
                        preconditioner=rich16)
    fg32 = FGMRESSolver(op32, tol=0.0, atol=0.0, restart=m2, maxiter=m2,
                        preconditioner=CastBridge(torch.float32, fg16))
    solver = CampsAdapter(op64, fg32,
                          AdapterConfig(J=ir_max, tau=tol, scaling="pow2", stag_h=3))
    return BuiltMethod(
        name=f"IR64<-32[FGMRES32^{m2}[cast32<-16[FGMRES16^{m3}[Rich16x{sweeps}]]]]",
        solver=solver, registry=reg,
        analytic_bytes=[("A64 vals", _sparse_value_bytes(problem, torch.float64)),
                        ("A32 vals", _sparse_value_bytes(problem, torch.float32)),
                        ("A16 vals", _sparse_value_bytes(problem, torch.float16)),
                        ("work64", 4 * problem.n * 8),
                        ("basis32", _basis_bytes(problem.n, m2, torch.float32, True)),
                        ("basis16", _basis_bytes(problem.n, m3, torch.float16, True))],
        notes="plain-IR-topped F3R body: no FP64 Krylov basis; contract "
              "supervision at the top precision edge only",
    )


def build_camps_sparse_hierarchy(problem: Problem, tol: float, restart: int,
                                 maxiter: int, half_dtype: torch.dtype = torch.float16,
                                 J_outer: int = 2, J_inner: int = 2,
                                 mid_tol: float = 1e-4, mid_restart: int = 20,
                                 mid_maxiter: int = 60,
                                 leaf_tol: float = 5e-2, leaf_restart: int = 10,
                                 leaf_maxiter: int = 20, **_) -> BuiltMethod:
    """FGMRES64[IR64<-32[FGMRES32[IR32<-16[FGMRES16(Jacobi16)]]]] — the full
    representative sparse hierarchy: three storage precisions, refinement at
    both edges, half-precision CSR matvec at the leaf."""
    reg: Dict = {}
    hname = "16" if half_dtype == torch.float16 else "b16"
    op64 = _sparse_op(problem, torch.float64, reg)
    op32 = _sparse_op(problem, torch.float32, reg)
    op16 = _sparse_op(problem, half_dtype, reg)
    pre16 = DiagJacobiSolver(op16, op16.diagonal)
    fg16 = FGMRESSolver(op16, tol=leaf_tol, restart=leaf_restart,
                        maxiter=leaf_maxiter, preconditioner=pre16)
    ir32_16 = CampsAdapter(op32, fg16,
                           AdapterConfig(J=J_inner, tau=mid_tol, scaling="pow2"))
    guard32 = ZeroGuardPreconditioner(ir32_16)
    fg32 = FGMRESSolver(op32, tol=mid_tol, restart=mid_restart,
                        maxiter=mid_maxiter, preconditioner=guard32)
    ir64_32 = CampsAdapter(op64, fg32, AdapterConfig(J=J_outer, tau=1e-8, scaling="pow2"))
    wrapper = CountingPreconditioner(ir64_32)
    guard = ZeroGuardPreconditioner(wrapper)
    solver = FGMRESSolver(op64, tol=tol, restart=restart, maxiter=maxiter,
                          preconditioner=guard)
    return BuiltMethod(
        name=f"FGMRES64[IR64<-32[FGMRES32[IR32<-{hname}[FGMRES{hname}(Jacobi{hname})]]]]",
        solver=solver, registry=reg, wrapper=wrapper, zero_guard=guard,
        analytic_bytes=[("A64 vals", _sparse_value_bytes(problem, torch.float64)),
                        ("A32 vals", _sparse_value_bytes(problem, torch.float32)),
                        (f"A{hname} vals", _sparse_value_bytes(problem, half_dtype)),
                        ("basis64", _basis_bytes(problem.n, restart, torch.float64, True)),
                        ("basis32", _basis_bytes(problem.n, mid_restart, torch.float32, True)),
                        (f"basis{hname}", _basis_bytes(problem.n, leaf_restart, half_dtype, True))],
        notes="half-precision leaf runs CSR matvec and Jacobi natively in "
              "storage format; norms/Krylov scalars accumulate FP32 (choice I-12)",
    )
