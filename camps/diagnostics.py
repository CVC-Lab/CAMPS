"""Backend precision diagnostics (spec §13).

Storage precision is not arithmetic precision: PyTorch and GPU backends promote
many operations internally. A claimed BF16/FP16 child solve must identify which
operations execute in the stated format and which accumulate in FP32. This
module (a) pins down configurable backend behavior, (b) documents known
promotion behavior per operation class, and (c) *probes* matmul accumulation at
runtime instead of trusting documentation.
"""
from __future__ import annotations

from typing import Dict

import torch


def configure_backend() -> Dict[str, bool]:
    """Pin backend switches so 'fp32' means IEEE fp32 multiplies.

    Disables TF32 for matmul and cuDNN; returns the applied settings. Call once
    before experiments; benchmarks record the returned dict.
    """
    applied = {}
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        applied["cuda.matmul.allow_tf32"] = False
        applied["cudnn.allow_tf32"] = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        applied["cuda.matmul.allow_fp16_reduced_precision_reduction"] = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        applied["cuda.matmul.allow_bf16_reduced_precision_reduction"] = False
    try:
        # float32 matmul precision: 'highest' = IEEE fp32
        torch.set_float32_matmul_precision("highest")
        applied["float32_matmul_precision"] = "highest"
    except Exception:
        pass
    return applied


def probe_matmul_accumulation(dtype: torch.dtype, device) -> Dict:
    """Empirically test whether matmul over `dtype` accumulates wider.

    Construct dot([1, eps, ..., eps], ones): with eps below the format's unit
    roundoff relative to 1, sequential native-format accumulation yields ~1.0,
    while FP32 (or wider) accumulation yields 1 + (n-1)*eps distinguishably.
    """
    from .precision import spec_of

    spec = spec_of(dtype)
    n = 2048
    eps = spec.unit_roundoff / 2.0
    a = torch.full((1, n), eps, dtype=dtype, device=device)
    a[0, 0] = 1.0
    b = torch.ones(n, 1, dtype=dtype, device=device)
    got = float((a @ b).to(torch.float64))
    expected_wide = 1.0 + (n - 1) * eps
    wide = abs(got - expected_wide) < abs(got - 1.0)
    return {
        "dtype": spec.name,
        "device": str(device),
        "result": got,
        "naive_sequential_estimate": 1.0,
        "wide_accumulation_estimate": expected_wide,
        # CAVEAT (spec §13): a 'True' here rules out naive sequential
        # storage-dtype accumulation but cannot by itself distinguish a wider
        # accumulator from a blocked/pairwise reduction in the storage dtype.
        # Width claims must combine this probe with the pinned backend flags
        # (configure_backend disables reduced-precision reductions, so cuBLAS
        # half matmuls accumulate in FP32 per PyTorch's documented contract).
        "inconsistent_with_naive_sequential_storage_accumulation": bool(wide),
    }


def promotion_report(device) -> Dict:
    """Which operations execute in the stated format vs. accumulate wider.

    Combines this implementation's explicit choices (spec I-12) with runtime
    probes. Cite this report before claiming 'the solve ran in FP16'
    (spec §13; results.md requirement).
    """
    device = torch.device(device)
    report: Dict = {
        "explicit_promotions_in_camps": {
            "norms": "half tensors are cast to FP32 before torch.linalg.vector_norm "
                     "(precision.colnorms, choice I-12)",
            "krylov_scalars": "Hessenberg/Givens/dot-product scalars kept in the "
                              "format's accumulation dtype (choice I-12)",
            "elementwise_axpy": "executed in storage dtype (no explicit promotion)",
            "scale_multiply": "power-of-two scaling exact in storage dtype (paper §4.1)",
        },
        "matmul_probes": {},
    }
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        try:
            report["matmul_probes"][str(dt)] = probe_matmul_accumulation(dt, device)
        except Exception as e:  # e.g. CPU half matmul unsupported in some builds
            report["matmul_probes"][str(dt)] = {"error": repr(e)}
    return report
