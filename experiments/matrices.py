"""Matrix-class generators for the CAMPS sweep.

Four classes, chosen so that each stresses a different aspect of the framework
(paper Section 5):

- dense well-conditioned      controlled kappa via SVD construction
- dense ill-conditioned       same construction, kappa sweep to 1e12
- sparse SPD                  3D Poisson seven-point stencil (paper §5.5)
- sparse nonsymmetric         3D convection-diffusion, wind sweep (paper §5.6)

All generators are deterministic given a seed. Right-hand sides come from a
manufactured solution x* ~ N(0, I)/sqrt(n), b = A x* formed in FP64, so the
forward error ||x - x*|| is measurable for every method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class Problem:
    """One linear system in FP64 plus metadata; per-precision copies on demand."""

    name: str
    kind: str                       # 'dense' | 'sparse'
    n: int
    A64: object                     # torch dense (cuda) or scipy CSR (host master)
    b64: torch.Tensor
    x_star: torch.Tensor
    device: torch.device
    meta: Dict = field(default_factory=dict)
    _cache: Dict = field(default_factory=dict)

    def dense(self, dtype: torch.dtype) -> torch.Tensor:
        if dtype == torch.float64:
            return self.A64  # master copy, never duplicated
        key = ("dense", dtype)
        if key not in self._cache:
            self._cache[key] = self.A64.to(dtype)
        return self._cache[key]

    def csr_parts(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(crow, col, vals, diag) torch tensors on device in dtype."""
        key = ("csr", dtype)
        if key not in self._cache:
            A = self.A64  # scipy csr fp64 master
            if ("crow") not in self._cache:
                self._cache["crow"] = torch.tensor(A.indptr, dtype=torch.int64, device=self.device)
                self._cache["col"] = torch.tensor(A.indices, dtype=torch.int64, device=self.device)
                self._cache["diag64_np"] = A.diagonal()
            vals = torch.tensor(A.data, dtype=torch.float64, device=self.device).to(dtype)
            diag = torch.tensor(self._cache["diag64_np"], dtype=torch.float64, device=self.device).to(dtype)
            self._cache[key] = (self._cache["crow"], self._cache["col"], vals, diag)
        return self._cache[key]


def _manufactured_rhs(matvec64, n: int, device, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x_star = torch.randn(n, generator=g, dtype=torch.float64) / np.sqrt(n)
    x_star = x_star.to(device)
    b = matvec64(x_star)
    return b, x_star


def dense_conditioned(n: int, kappa: float, device, seed: int = 0,
                      symmetric: bool = False) -> Problem:
    """A = U diag(sigma) V^T with log-spaced singular values in [1/kappa, 1].

    U, V orthogonal from QR of Gaussian matrices (V = U for the symmetric
    positive definite variant). cond_2(A) = kappa exactly by construction.
    Entries are O(1/sqrt(n)) — comfortably inside FP16 normal range, so the
    class is castable along the whole format ladder without equilibration.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    G1 = torch.randn(n, n, generator=gen, dtype=torch.float64)
    U, _ = torch.linalg.qr(G1.to(device))
    if symmetric:
        V = U
    else:
        G2 = torch.randn(n, n, generator=gen, dtype=torch.float64)
        V, _ = torch.linalg.qr(G2.to(device))
    sigma = torch.logspace(0, -np.log10(kappa), n, dtype=torch.float64, device=device)
    A = (U * sigma.unsqueeze(0)) @ V.T
    b, x_star = _manufactured_rhs(lambda x: A @ x, n, device, seed + 1)
    tag = "spd" if symmetric else "nonsym"
    return Problem(
        name=f"dense-{tag}-n{n}-k{kappa:.0e}", kind="dense", n=n,
        A64=A, b64=b, x_star=x_star, device=torch.device(device),
        meta={"kappa": kappa, "symmetric": symmetric,
              "sigma_max": 1.0, "sigma_min": 1.0 / kappa},
    )


def _poisson3d_csr(g: int) -> sp.csr_matrix:
    """Seven-point stencil, diagonal 6, neighbors -1 (paper §5.5)."""
    I = sp.eye(g, format="csr")
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(g, g), format="csr")
    A = (sp.kron(sp.kron(T, I), I) + sp.kron(sp.kron(I, T), I)
         + sp.kron(sp.kron(I, I), T))
    return A.tocsr()


def _convdiff3d_csr(g: int, w: float) -> sp.csr_matrix:
    """Nonsymmetric seven-point stencil with wind w in the third coordinate
    (paper §5.6): backward z-neighbor -1-w, forward and transverse -1,
    diagonal 6+w."""
    n = g ** 3
    diags = {
        0: np.full(n, 6.0 + w),
    }
    # index i = x*g*g + y*g + z (lexicographic; z fastest)
    idx = np.arange(n)
    z = idx % g
    y = (idx // g) % g
    x = idx // (g * g)
    rows, cols, vals = [idx], [idx], [diags[0]]

    def add(mask, offset, coeff):
        r = idx[mask]
        rows.append(r)
        cols.append(r + offset)
        vals.append(np.full(r.size, coeff))

    add(z > 0, -1, -1.0 - w)        # backward third-coordinate neighbor
    add(z < g - 1, +1, -1.0)        # forward
    add(y > 0, -g, -1.0)            # transverse
    add(y < g - 1, +g, -1.0)
    add(x > 0, -g * g, -1.0)
    add(x < g - 1, +g * g, -1.0)
    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    return A.tocsr()


def sparse_problem(kind: str, g: int, device, w: float = 20.0, seed: int = 0) -> Problem:
    if kind == "poisson":
        A = _poisson3d_csr(g)
        name = f"poisson3d-g{g}"
        meta = {"grid": g, "stencil": "7pt Laplacian", "spd": True}
    elif kind == "convdiff":
        A = _convdiff3d_csr(g, w)
        name = f"convdiff3d-g{g}-w{w:g}"
        meta = {"grid": g, "wind": w, "spd": False}
    else:
        raise ValueError(kind)
    n = A.shape[0]
    crow = torch.tensor(A.indptr, dtype=torch.int64, device=device)
    col = torch.tensor(A.indices, dtype=torch.int64, device=device)
    vals = torch.tensor(A.data, dtype=torch.float64, device=device)
    csr64 = torch.sparse_csr_tensor(crow, col, vals, size=(n, n))
    b, x_star = _manufactured_rhs(lambda x: csr64 @ x, n, device, seed + 1)
    prob = Problem(
        name=name, kind="sparse", n=n, A64=A, b64=b, x_star=x_star,
        device=torch.device(device),
        meta={**meta, "nnz": int(A.nnz)},
    )
    # pre-seed fp64 CSR cache with the tensors we already built
    prob._cache["crow"] = crow
    prob._cache["col"] = col
    prob._cache["diag64_np"] = A.diagonal()
    prob._cache[("csr", torch.float64)] = (
        crow, col, vals,
        torch.tensor(A.diagonal(), dtype=torch.float64, device=device),
    )
    return prob
