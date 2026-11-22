"""Power-of-two residual scaling (paper Alg. 1 step 5, §4.1; spec choice I-3).

The paper requires an *exactly representable power-of-two* scale s such that
Q_q(s * r_p) is finite and representable; power-of-two scaling adds no rounding
in binary arithmetic as long as the scaled values stay finite and do not
underflow, and it is especially important before FP16 conversion (paper §4.1).

The paper does not specify how to pick s. Choice I-3: aim the largest residual
entry at the binade (2**(target_binade - 1), 2**target_binade], default
target_binade = 0, i.e. max|s r| in (0.5, 1]. This sits far from BOTH ends of
every supported format's range, and — unlike aiming near the top of the child
range — leaves the child solve room to amplify by up to ~cond(A) without
overflowing, and leaves nested adapters a sane input magnitude. (An earlier
draft of this implementation aimed at 2**(E_max_q - 3) and overflowed fp32
children on cond>=100 systems; recorded in docs/results.md as a negative
result.) Scales are chosen per column for multi-RHS input (choice I-1).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from .precision import PrecisionSpec, spec_of


def choose_pow2_scale(
    r: torch.Tensor,
    child: PrecisionSpec,
    target_binade: int = 0,
) -> Tuple[torch.Tensor, Dict]:
    """Choose per-column power-of-two scales for residual conversion to `child`.

    Returns (s, info): s is a float64 tensor of shape () for a vector input or
    (k,) for an (n, k) block, each entry an exact power of two; info records
    zero columns and the chosen exponents.

    s = 2**k with k = floor(log2(2**target_binade / max|r_col|)),
    so that max|s r| lands in (2**(target_binade-1), 2**target_binade].

    A zero column gets s = 1 and is flagged; the caller decides whether that is
    SUCCESS (stopping test already fired) or FAILURE (Alg. 1 step 7).
    """
    child = spec_of(child)
    if target_binade >= child.max_exponent:
        raise ValueError(
            f"target_binade {target_binade} not representable in {child.name}"
        )
    single = r.ndim == 1
    R = r.unsqueeze(1) if single else r
    amax = R.abs().amax(dim=0).to(torch.float64)  # per-column max magnitude
    zero_cols = amax == 0

    # k = floor(log2(2^target / amax)); guard zeros to keep log finite.
    safe_amax = torch.where(zero_cols, torch.ones_like(amax), amax)
    k = torch.floor(target_binade - torch.log2(safe_amax))
    # Exact powers of two: exponentiation of 2.0 by integer k is exact in fp64
    # for the exponent ranges of all supported formats.
    s = torch.pow(torch.tensor(2.0, dtype=torch.float64), k)
    s = torch.where(zero_cols, torch.ones_like(s), s)

    info = {
        "exponents": k.tolist(),
        "zero_columns": zero_cols.tolist(),
        "target_binade": target_binade,
    }
    if single:
        s = s.squeeze(0)
    return s, info


def apply_scale(r: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Multiply per-column scales into r without changing dtype.

    s is float64 (exact powers of two); the product is formed in r's dtype —
    multiplication by an in-range power of two is exact in binary FP.
    """
    s_cast = s.to(r.dtype)
    if r.ndim == 1:
        return r * s_cast
    return r * s_cast.unsqueeze(0)


def unapply_scale(c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Divide per-column scales out of c (Alg. 1 step 8: s^{-1} Q_p(c_q))."""
    s_cast = s.to(c.dtype)
    if c.ndim == 1:
        return c / s_cast
    return c / s_cast.unsqueeze(0)
