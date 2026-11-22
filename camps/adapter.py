"""The CAMPS precision adapter IR_{p<-q}^{J,tau}[S_q] (paper §4.1, Algorithm 1).

The adapter accepts and returns vectors in the parent format p, stores its
state and forms residuals and updates in p, converts a safely scaled residual
to q, invokes the zero-start child S_q, and promotes the returned correction
back to p (paper §1, §4.1). Exactly one explicit precision boundary is declared
per adapter edge (paper Fig. 1).

An adapter is itself a SolverNode with dtype p, so it can stand wherever a
parent expects an approximate inverse — including as the child of another
adapter (paper §4.4, Theorem 1). It is a VARIABLE node (is_fixed = False):
local stopping, scaling, and failure handling make successive applications
differ, which is why an enclosing Krylov method must be flexible (spec §10).

Beyond Algorithm 1, the adapter re-verifies each accumulated correction against
the configured residual contract and can reject / fail / fall back on
violation. That acceptance test is an implementation extension (spec §6,
choice I-6), not a paper requirement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch

from .base_solvers import SolverNode
from .contracts import ResidualContract, check_zero_start
from .operators import LinearOperator
from .precision import colnorms, conversion_report, convert, spec_of
from .results import AdapterResult, Status
from .scaling import apply_scale, choose_pow2_scale, unapply_scale


@dataclass
class AdapterConfig:
    """Configuration of one adapter edge (paper Alg. 1 parameters + spec §12).

    J                 attempt budget (Alg. 1): every attempt that reaches the
                      acceptance test counts, accepted or rejected, including
                      the attempt on which a fallback switch happens.
    tau, atol         parent stopping ||r|| <= tau*||g|| + atol (Alg. 1 step 4;
                      atol default 0, choice I-8).
    scaling           'pow2' (Alg. 1 step 5) or 'none' (the paper's simplified
                      fixed-count realization, §4.1/§5.1 — direct cast).
    target_binade     scale aims max|s r| into (2**(t-1), 2**t] (choice I-3).
    contract          runtime acceptance requirement (choice I-6).
    on_violation      'reject' | 'fail' | 'fallback' (spec §6).
    fallback_solver   SolverNode used when on_violation = 'fallback'. Its
                      precision q_f may be the parent's (same-precision
                      recovery, scale 1 and identity cast) or any lower
                      precision with u_p <= u_{q_f}; on a violation the
                      adapter switches its active pair (q, S_q) -> (q_f, S_f)
                      and every later attempt rebuilds the child right-hand
                      side from the RETAINED accepted residual at the new
                      precision (Alg. 1 step 8).
    stag_h, stag_rtol stagnation: h consecutive checks with relative decrease
                      worse than stag_rtol (Alg. 1 step 9; predicate is choice
                      I-5).
    residual_dtype    optional higher-precision residual format (Alg. 1 step 3
                      permits "a separately documented higher residual
                      precision", choice I-2). Requires residual_op.
    residual_op       operator stored in residual_dtype for residual formation.
    child_system_scale
                      exact power-of-two scalar equilibration of the child
                      SYSTEM: the child was built to solve (A/σ) ĉ = r, and
                      the adapter recovers c = ĉ/σ during promotion, in
                      PARENT precision (so a child whose true solution
                      overflows the child range remains solvable). This is
                      the scalar realization of the equilibration the paper
                      names as separate from rhs scaling ("After any required
                      equilibration, A_p = Q_p(A)", §3; §4.1). Default 1.0 =
                      no equilibration. Must be an exact power of two.
    column_acceptance for multi-RHS blocks: accept/reject the correction PER
                      COLUMN instead of all-or-nothing. Passing columns are
                      accumulated, violating columns are rolled back
                      individually and re-attempted on later corrections;
                      per-column reject counts and the final per-column
                      convergence mask are reported in diagnostics
                      (column_rejects, column_converged) rather than hidden.
                      Default False preserves the original all-or-nothing
                      semantics (extension of choice I-6 for block solves
                      such as inverse construction M X = I).
    damping           'none' applies the promoted correction with unit step.
                      'residual_min' scales it by the residual-minimizing
                      step alpha* = <r, A c> / <A c, A c> at the cost of one
                      extra parent matvec. Branches: if A c = 0 or
                      <r, A c> <= 0 the step is alpha = 0 (a null step, which
                      the stagnation detector reports); otherwise alpha =
                      min(alpha*, damping_max). For 0 <= alpha <= alpha* the
                      exact residual satisfies ||r - alpha A c||^2 <=
                      ||r||^2 - alpha <r, A c>, so the clipped step never
                      increases the residual; the rounded iterate must still
                      pass the acceptance test.
    acceptance        'residual' checks the configured ResidualContract on the
                      recomputed parent residual (default). 'energy' checks,
                      for an SPD parent operator, certified strict decrease of
                      the quadratic energy phi(y) = y'A y / 2 - g'y for the
                      COMMITTED displacement h = fl(y + d) - y:
                      dphi_hat + E < 0, where dphi_hat = fl(h'A h / 2 - r'h)
                      and E bounds the evaluation errors in h, A h, the dot
                      products, and the residual (paper §4.3). Testing the
                      proposed d instead of h certifies steps that the rounded
                      update does not take (A = 1, y = 2^53, r = 2, d = 1/4:
                      d'Ad/2 - r'd = -0.46875 but fl(y + d) = y). Costs one
                      product with A and two with |A| per attempt; requires
                      an operator with supports_abs.
    """

    J: int = 3
    tau: float = 1e-10
    atol: float = 0.0
    scaling: Literal["pow2", "none"] = "pow2"
    target_binade: int = 0
    contract: ResidualContract = field(default_factory=ResidualContract)
    on_violation: Literal["reject", "fail", "fallback"] = "reject"
    fallback_solver: Optional[SolverNode] = None
    stag_h: int = 2
    stag_rtol: float = 0.99
    residual_dtype: Optional[torch.dtype] = None
    residual_op: Optional[LinearOperator] = None
    child_system_scale: float = 1.0
    column_acceptance: bool = False
    damping: Literal["none", "residual_min"] = "none"
    damping_max: float = 2.0
    acceptance: Literal["residual", "energy"] = "residual"


class CampsAdapter(SolverNode):
    is_fixed = False  # variable by construction (spec §10)

    def __init__(self, parent_op: LinearOperator, child: SolverNode,
                 config: Optional[AdapterConfig] = None):
        cfg = config or AdapterConfig()
        p, q = spec_of(parent_op.dtype), spec_of(child.dtype)
        # Type rule (paper §3): child accepts q, adapter returns p, u_p < u_q;
        # violations are construction-time errors, never silent casts.
        p.assert_valid_edge(q)
        if not check_zero_start(child):
            raise TypeError(
                f"child solver {child.name} exposes an initial-guess parameter; "
                "CAMPS requires zero-initialized correction solves (paper §4.2)"
            )
        self.fallback_spec = None
        if cfg.on_violation == "fallback":
            fb = cfg.fallback_solver
            if fb is None:
                raise TypeError("fallback mode requires a fallback_solver")
            fspec = spec_of(fb.dtype)
            if fspec.unit_roundoff < p.unit_roundoff:
                raise TypeError(
                    f"fallback solver precision {fspec.name} exceeds the parent's "
                    f"{p.name}; the fallback pair must satisfy u_p <= u_f (same-"
                    "precision recovery is allowed, paper Alg. 1 step 8)")
            if not check_zero_start(fb):
                raise TypeError("fallback solver must be zero-initialized (paper §4.2)")
            self.fallback_spec = fspec
        if cfg.acceptance == "energy":
            if parent_op.symmetric is False or parent_op.positive_definite is False:
                raise TypeError("energy acceptance requires an SPD parent operator")
            if not getattr(parent_op, "supports_abs", False):
                raise TypeError(
                    "certified energy acceptance needs an operator that provides "
                    "apply_abs (|A||x|) for its evaluation-error bound (paper §4.3)")
        if cfg.child_system_scale != 1.0:
            import math
            m, _ = math.frexp(cfg.child_system_scale)
            if m != 0.5 or cfg.child_system_scale <= 0:
                raise ValueError("child_system_scale must be an exact positive "
                                 "power of two (equilibration adds no rounding)")
        if cfg.residual_dtype is not None:
            rspec = spec_of(cfg.residual_dtype)
            if rspec.unit_roundoff > p.unit_roundoff:
                raise TypeError("residual_dtype must be at least parent precision (choice I-2)")
            if cfg.residual_op is None or cfg.residual_op.dtype != cfg.residual_dtype:
                raise TypeError("residual_dtype requires a matching residual_op")
        self.op = parent_op
        self.child = child
        self.cfg = cfg
        self.parent_spec, self.child_spec = p, q
        self.dtype = parent_op.dtype
        self.device = parent_op.device
        self.name = f"IR[{p.name}<-{q.name}]({child.name})"
        # rounding constants of the parent format for the certified energy
        # test (paper §4.3, §4.7 model M1-M3): gamma_k = k u / (1 - k u), with
        # the row count s in place of n for the products when it is known
        n = parent_op.shape[0]
        s_row = getattr(parent_op, "max_nnz_per_row", None) or n
        u = p.unit_roundoff

        def _gam(k: int) -> float:
            ku = k * u
            return ku / (1.0 - ku) if ku < 1.0 else float("inf")

        self._u_p = u
        self._gam_s = _gam(s_row)        # product A h (s terms per entry)
        self._gam_s1 = _gam(s_row + 1)   # residual evaluation (M1)
        self._gam_n = _gam(n)            # dot products over n entries
        self._gam_self = _gam(n + 3)     # rounding of the bound itself

    # -- certified energy test ----------------------------------------------
    def _dot(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        acc = self.parent_spec.accumulation_dtype
        if a.ndim == 1:
            return torch.dot(a.to(acc), b.to(acc))
        return (a.to(acc) * b.to(acc)).sum(0)

    def energy_certificate(self, g: torch.Tensor, y: torch.Tensor,
                           r_hat: torch.Tensor, y_new: torch.Tensor):
        """Certified energy change of the COMMITTED displacement h = y_new - y.

        Returns (dphi_hat, bound, h): dphi_hat = fl(h'A h/2 - r_hat'h) and a
        bound E >= |dphi_hat - (h'A h/2 - r'h)| with r the exact residual at y,
        covering the rounding of h, of A h, of both dot products, and of the
        evaluated residual r_hat (paper §4.3). Acceptance is dphi_hat + E < 0.
        If y_new == y (the rounded update did not move), h = 0 and the test
        fails: no decrease is certified for a step that was not taken."""
        h = y_new - y                       # fl_p; exact where Sterbenz applies
        w_h = self.op.apply(h)              # A_p h
        a_h = self.op.apply_abs(h)          # |A_p||h|
        a_y = self.op.apply_abs(y)          # |A_p||y|
        dphi = 0.5 * self._dot(h, w_h) - self._dot(r_hat, h)
        t1 = self._dot(h.abs(), a_h)
        t2 = self._dot(g.abs() + a_y, h.abs())
        t3 = self._dot(r_hat.abs(), h.abs())
        u = self._u_p
        bound = ((2.0 * u + self._gam_s + self._gam_n) * t1
                 + self._gam_s1 * (1.0 + u) * t2
                 + (u + self._gam_n) * t3
                 + u * dphi.abs()) * (1.0 + self._gam_self)
        return dphi, bound, h

    # -- residual formation -------------------------------------------------
    def _authoritative_residual(self, g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """r_p = g_p - A_p y_p in parent precision (Alg. 1 step 3), or in the
        separately documented residual precision (choice I-2)."""
        if self.cfg.residual_dtype is None:
            return g - self.op.apply(y)
        gr = convert(g, self.cfg.residual_dtype)
        yr = convert(y, self.cfg.residual_dtype)
        r = gr - self.cfg.residual_op.apply(yr)
        return convert(r, self.dtype)

    # -- Algorithm 1 --------------------------------------------------------
    def solve(self, g: torch.Tensor) -> AdapterResult:
        self._check_rhs(g)
        cfg = self.cfg
        single = g.ndim == 1

        y = torch.zeros_like(g)                      # step 1: y_p = 0
        gnorm = colnorms(g)
        target = cfg.tau * gnorm + cfg.atol

        history, child_results, scales, accepted = [], [], [], []
        contract_events = []
        column_rejects: dict = {}
        stagnant = 0
        status = Status.MAX_STEPS
        fallback_used = False
        fallback_switch_t = None
        attempts = 0

        r = self._authoritative_residual(g, y)       # step 1
        rnorm = colnorms(r)
        history.append(rnorm)
        if bool((rnorm <= target).all()):            # step 1: already converged
            status = Status.SUCCESS

        for t in range(cfg.J if status is not Status.SUCCESS else 0):   # step 2: attempt budget
            attempts += 1
            # active pair (q, S_q): the configured child, or the fallback pair
            # (q_f, S_f) after a violation (Alg. 1 step 8)
            if fallback_used:
                active_child, active_spec = cfg.fallback_solver, self.fallback_spec
            else:
                active_child, active_spec = self.child, self.child_spec
            same_precision = active_spec.torch_dtype == self.dtype

            # step 3: scale and cast the RETAINED accepted residual for the
            # active child; a same-precision child gets scale 1 and no cast
            if same_precision:
                s = torch.ones((() if single else (g.shape[1],)), dtype=torch.float64,
                               device=g.device)
                scale_info = {"mode": "identity (same-precision child)",
                              "fallback": fallback_used}
                r_q = r.clone()
                conv_rep = None
            else:
                if cfg.scaling == "pow2":
                    s, scale_info = choose_pow2_scale(r, active_spec, cfg.target_binade)
                else:
                    s = torch.ones((() if single else (g.shape[1],)), dtype=torch.float64,
                                   device=g.device)
                    scale_info = {"mode": "none (simplified fixed-count realization, §5.1)"}
                r_scaled = apply_scale(r, s)
                r_q = convert(r_scaled, active_spec)         # Q_q(s r_p)
                conv_rep = conversion_report(r_scaled, active_spec)

                # converted residual numerically zero (choice I-4: no nonzero
                # entry survives) or nonfinite
                if conv_rep["all_zero"] or conv_rep["any_nonfinite"]:
                    status = Status.FAILURE
                    scales.append(s)
                    contract_events.append({
                        "t": t, "event": "conversion_destroyed_residual",
                        "conversion": conv_rep, "fallback": fallback_used,
                    })
                    break

            cres = active_child.solve(r_q)               # step 4: zero-start child
            child_results.append(cres)
            c_q = cres.correction

            # step 4: nonfinite child output
            if cres.status is Status.FAILURE or not torch.isfinite(c_q).all():
                status = Status.FAILURE
                scales.append(s)
                contract_events.append({
                    "t": t, "event": "fallback_failure" if fallback_used else "child_failure",
                    "child_status": cres.status.value})
                break

            # step 5: promote (exact), unscale (exact), accumulate in parent
            # precision. The child_system_scale unscale happens HERE, in
            # parent precision, so an equilibrated child whose true solution
            # exceeds the child range still promotes finitely.
            c_p = unapply_scale(convert(c_q, self.parent_spec), s)
            if cfg.child_system_scale != 1.0 and not fallback_used:
                c_p = c_p / cfg.child_system_scale

            # -- optional residual-minimizing damping (step 5) --------------
            acc_dt = self.parent_spec.accumulation_dtype
            alpha = None
            w = None
            if cfg.damping == "residual_min":
                w = self.op.apply(c_p)                       # A_p c_p (one parent matvec)
                if single:
                    num = torch.dot(r.to(acc_dt), w.to(acc_dt))
                    den = torch.dot(w.to(acc_dt), w.to(acc_dt))
                else:
                    num = (r.to(acc_dt) * w.to(acc_dt)).sum(0)
                    den = (w.to(acc_dt) * w.to(acc_dt)).sum(0)
                tiny = torch.finfo(acc_dt).tiny
                alpha = torch.where(den > 0, num / torch.clamp(den, min=tiny),
                                    torch.zeros_like(num))
                alpha = torch.clamp(alpha, 0.0, cfg.damping_max)
                a_p = alpha.to(self.dtype)
                c_p = c_p * (a_p if single else a_p.unsqueeze(0))
                w = w * (a_p if single else a_p.unsqueeze(0))

            y_new = y + c_p                                  # step 6: trial iterate, fl_p
            scales.append(s)

            # -- step 7: acceptance test on the COMMITTED iterate -------------
            r_new = self._authoritative_residual(g, y_new)   # recomputed residual
            rnorm_new = colnorms(r_new)
            dphi = None
            dphi_bound = None
            if cfg.acceptance == "energy":
                # certified SPD energy decrease of the committed displacement
                # h = y_new - y (not of the proposed c_p): dphi_hat + E < 0
                dphi, dphi_bound, _ = self.energy_certificate(g, y, r, y_new)
                ok = (dphi + dphi_bound) < 0
            else:
                ok = self.cfg.contract.check(rnorm, rnorm_new, self.dtype)
            ok_all = bool(ok.all())
            contract_events.append({
                "t": t,
                "r_before": _fl(rnorm), "r_after": _fl(rnorm_new),
                "satisfied": _fl(ok), "fallback": fallback_used,
                "active": active_spec.name,
                "alpha": None if alpha is None else _fl(alpha),
                "delta_phi": None if dphi is None else _fl(dphi),
                "delta_phi_bound": None if dphi_bound is None else _fl(dphi_bound),
            })

            if not ok_all and cfg.column_acceptance and not single:
                # per-column acceptance (spec: block-solve extension of I-6):
                # accumulate passing columns, roll back violating ones
                # individually, report — never hide — the partial failures
                mask = ok  # (k,) bool
                y = torch.where(mask.unsqueeze(0), y_new, y)
                r = torch.where(mask.unsqueeze(0), r_new, r)
                prev_rnorm = rnorm
                rnorm = torch.where(mask, rnorm_new, rnorm)
                history.append(rnorm)
                accepted.append(float(mask.to(torch.float64).mean()))
                for j in torch.nonzero(~mask).flatten().tolist():
                    column_rejects[j] = column_rejects.get(j, 0) + 1
                # success is checked immediately after every accepted update,
                # before stagnation (Alg. 1 step 7)
                if bool((rnorm <= target).all()):
                    status = Status.SUCCESS
                    break
                if bool((rnorm > cfg.stag_rtol * prev_rnorm).all()):
                    stagnant += 1
                    if stagnant >= cfg.stag_h:
                        status = Status.STAGNATED
                        break
                else:
                    stagnant = 0
                continue

            if not ok_all:
                accepted.append(False)
                if cfg.on_violation == "reject":
                    # roll back and stop: retrying the same deterministic
                    # child would loop (spec §6)
                    status = Status.REJECTED
                    break
                if cfg.on_violation == "fail":
                    status = Status.FAILURE
                    break
                # fallback: discard the trial (the accepted state is kept) and
                # switch the active pair to (q_f, S_f) for the remaining attempts
                if not fallback_used:
                    fallback_used = True
                    fallback_switch_t = t
                    continue
                # the fallback child itself violated the contract: stop
                status = Status.REJECTED
                break

            accepted.append(True)
            y = y_new
            r, prev_rnorm, rnorm = r_new, rnorm, rnorm_new
            history.append(rnorm)

            # step 7: success is checked immediately after every accepted
            # update, before the stagnation test
            if bool((rnorm <= target).all()):
                status = Status.SUCCESS
                break

            # step 8: stagnation over m_stag checks (choice I-5)
            if bool((rnorm > cfg.stag_rtol * prev_rnorm).all()):
                stagnant += 1
                if stagnant >= cfg.stag_h:
                    status = Status.STAGNATED
                    break
            else:
                stagnant = 0
        else:
            # attempt budget exhausted; the last accepted update was already
            # tested for success, so this is MAX_STEPS unless the loop never ran
            if bool((rnorm <= target).all()):
                status = Status.SUCCESS

        final = colnorms(self._authoritative_residual(g, y))
        accepted_count = sum(1 for a in accepted if a is True or a == 1.0)
        rejected_count = sum(1 for a in accepted if a is False or a == 0.0)
        return AdapterResult(
            correction=y,
            initial_residual=gnorm,
            final_residual=final,
            iterations=len(accepted),
            status=status,
            diagnostics={
                "edge": self.name,
                "scaling": cfg.scaling,
                "fallback_used": fallback_used,
                "fallback_switch_attempt": fallback_switch_t,
                "fallback_precision": None if self.fallback_spec is None else self.fallback_spec.name,
                "attempts": attempts,
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "last_conversion_report": conv_rep if 'conv_rep' in locals() else None,
                "target": _fl(target),
                "column_rejects": column_rejects,
                "column_converged": _fl(final <= target) if not single else None,
                "damping": cfg.damping,
                "acceptance": cfg.acceptance,
            },
            residual_history=history,
            child_results=child_results,
            scales=scales,
            accepted=accepted,
            contract_record={
                "configured": cfg.contract.describe(),
                "events": contract_events,
            },
        )


def _fl(t: torch.Tensor):
    if t.ndim == 0:
        return float(t)
    return [float(v) for v in t]
