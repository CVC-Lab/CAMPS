"""Precision semantics for CAMPS (spec §13; paper §3).

Each supported floating-point format is described by an explicit `PrecisionSpec`.
The solver grammar orders formats by unit roundoff (`u_p < u_q` on every adapter
edge, paper §3). Dynamic-range properties are recorded separately because FP16 and
BF16 are *not* ordered by a single scalar: BF16 has less significand precision but
substantially greater dynamic range (spec §13).

All format conversions in CAMPS go through :func:`convert` /
:func:`conversion_report` so that every cast is explicit and auditable
(paper §3: "forbidding an implicit or undocumented cast").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class PrecisionSpec:
    """Static description of a floating-point format.

    unit_roundoff        u = 2**-t for a t-bit significand (hidden bit included),
                         round-to-nearest.
    significand_bits     t, including the hidden bit.
    min_exponent         E_min of the normal range (unbiased).
    max_exponent         E_max of the normal range (unbiased).
    min_normal           smallest positive normal number.
    min_subnormal        smallest positive subnormal number.
    max_finite           largest finite number.
    accumulation_dtype   dtype this implementation uses for reductions (norms,
                         dot products) over tensors stored in this format
                         (spec choice I-12).
    backend_notes        known backend promotion behavior; storage precision is
                         not arithmetic precision (spec §13).
    """

    name: str
    torch_dtype: torch.dtype
    unit_roundoff: float
    significand_bits: int
    min_exponent: int
    max_exponent: int
    min_normal: float
    min_subnormal: float
    max_finite: float
    accumulation_dtype: torch.dtype
    backend_notes: str = ""

    def assert_valid_edge(self, child: "PrecisionSpec") -> None:
        """Enforce the grammar side-condition u_p < u_q (paper §3).

        `self` is the parent format p, `child` the child format q. Note that by
        unit roundoff FP16 (u=2^-11) is *higher* precision than BF16 (u=2^-8);
        their dynamic ranges order the opposite way and are checked nowhere
        here — range hazards are handled by scaling and conversion reports.
        """
        if not self.unit_roundoff < child.unit_roundoff:
            raise TypeError(
                f"invalid adapter edge {self.name}<-{child.name}: requires "
                f"u_parent < u_child but u({self.name})={self.unit_roundoff:.3e} "
                f">= u({child.name})={child.unit_roundoff:.3e} (paper §3)"
            )


FP64 = PrecisionSpec(
    name="fp64",
    torch_dtype=torch.float64,
    unit_roundoff=2.0**-53,
    significand_bits=53,
    min_exponent=-1022,
    max_exponent=1023,
    min_normal=2.0**-1022,
    min_subnormal=2.0**-1074,
    max_finite=(2.0 - 2.0**-52) * 2.0**1023,
    accumulation_dtype=torch.float64,
    backend_notes="native on CPU and CUDA; CUDA FP64 throughput is reduced on "
    "consumer GPUs but arithmetic is IEEE double",
)

FP32 = PrecisionSpec(
    name="fp32",
    torch_dtype=torch.float32,
    unit_roundoff=2.0**-24,
    significand_bits=24,
    min_exponent=-126,
    max_exponent=127,
    min_normal=2.0**-126,
    min_subnormal=2.0**-149,
    max_finite=(2.0 - 2.0**-23) * 2.0**127,
    accumulation_dtype=torch.float32,
    backend_notes="native everywhere; on CUDA, matmul may use TF32 "
    "(10-bit significand multiplies) unless "
    "torch.backends.cuda.matmul.allow_tf32 is False — CAMPS sets it False "
    "in diagnostics.configure_backend() so 'fp32' means IEEE fp32 multiplies",
)

FP16 = PrecisionSpec(
    name="fp16",
    torch_dtype=torch.float16,
    unit_roundoff=2.0**-11,
    significand_bits=11,
    min_exponent=-14,
    max_exponent=15,
    min_normal=2.0**-14,
    min_subnormal=2.0**-24,
    max_finite=65504.0,
    accumulation_dtype=torch.float32,
    backend_notes="CUDA tensor-core matmul accumulates in FP32; CPU half matmul "
    "is emulated via FP32. Elementwise ops execute in FP16 storage/arithmetic. "
    "Narrow exponent range (max 65504) makes pre-conversion scaling essential "
    "(paper §4.1).",
)

BF16 = PrecisionSpec(
    name="bf16",
    torch_dtype=torch.bfloat16,
    unit_roundoff=2.0**-8,
    significand_bits=8,
    min_exponent=-126,
    max_exponent=127,
    min_normal=2.0**-126,
    min_subnormal=2.0**-133,
    max_finite=(2.0 - 2.0**-7) * 2.0**127,
    accumulation_dtype=torch.float32,
    backend_notes="CUDA tensor-core matmul accumulates in FP32; CPU bf16 matmul "
    "accumulates in FP32. FP32-like exponent range: overflow on conversion is "
    "rare, but only 8 significand bits (coarser rounding than FP16).",
)

_SPECS: Dict[torch.dtype, PrecisionSpec] = {
    torch.float64: FP64,
    torch.float32: FP32,
    torch.float16: FP16,
    torch.bfloat16: BF16,
}

_BY_NAME: Dict[str, PrecisionSpec] = {s.name: s for s in _SPECS.values()}


def spec_of(dtype_or_name) -> PrecisionSpec:
    """Look up the PrecisionSpec for a torch dtype or a name like 'fp16'."""
    if isinstance(dtype_or_name, PrecisionSpec):
        return dtype_or_name
    if isinstance(dtype_or_name, torch.dtype):
        try:
            return _SPECS[dtype_or_name]
        except KeyError:
            raise KeyError(f"unsupported CAMPS dtype: {dtype_or_name}")
    try:
        return _BY_NAME[str(dtype_or_name)]
    except KeyError:
        raise KeyError(f"unsupported CAMPS format name: {dtype_or_name!r}")


def convert(x: torch.Tensor, target) -> torch.Tensor:
    """Q_p(x): explicit rounding/conversion to the target format (paper §3)."""
    spec = spec_of(target)
    return x.to(spec.torch_dtype)


def conversion_report(x: torch.Tensor, target) -> Dict[str, float]:
    """Convert and audit: what did Q_q destroy? (paper §4.1 equilibration caveat)

    Reports overflow (finite -> inf), total underflow (nonzero -> zero), and the
    fraction of entries landing in the subnormal range of the target format.
    Used both for right-hand-side conversion diagnostics and to detect a child
    matrix A_q that was structurally damaged by conversion — which rhs scaling
    cannot repair (paper §4.1).
    """
    spec = spec_of(target)
    y = x.to(spec.torch_dtype)
    x_finite = torch.isfinite(x)
    x_nonzero = x != 0
    n = x.numel()
    overflow = int((x_finite & ~torch.isfinite(y)).sum())
    underflow_to_zero = int((x_nonzero & (y == 0)).sum())
    y_abs = y.abs().to(torch.float32)
    subnormal = int(((y_abs > 0) & (y_abs < spec.min_normal)).sum())
    return {
        "target": spec.name,
        "n": n,
        "overflow_count": overflow,
        "underflow_to_zero_count": underflow_to_zero,
        "underflow_fraction": underflow_to_zero / max(n, 1),
        "subnormal_count": subnormal,
        "all_zero": bool((y == 0).all()),
        "any_nonfinite": bool((~torch.isfinite(y)).any()),
    }


def colnorms(x: torch.Tensor, accumulate: Optional[torch.dtype] = None) -> torch.Tensor:
    """Per-column 2-norms with explicit accumulation promotion (choice I-12).

    Accepts (n,) or (n, k); returns a 0-d or (k,) tensor in the accumulation
    dtype of x's format. The promotion (half -> fp32) is deliberate and recorded
    in diagnostics.promotion_report(); norms of half tensors are NOT computed in
    half arithmetic.
    """
    spec = spec_of(x.dtype)
    acc = accumulate or spec.accumulation_dtype
    xa = x.to(acc)
    return torch.linalg.vector_norm(xa, dim=0) if xa.ndim == 2 else torch.linalg.vector_norm(xa)
