# CAMPS: Composable Adapters for Mixed-Precision Solvers

CAMPS is a framework for building mixed-precision linear solvers out of
nested parts. Its central piece is the **adapter** $\mathrm{IR}_{p\leftarrow q}[S_q]$,
which sits at the boundary between a parent precision $p$ and a child
precision $q$. The adapter computes residuals and updates in precision $p$,
sends a scaled residual to a child solver $S_q$ that works in precision $q$,
and checks every correction with an acceptance test before it keeps it. A
rejected correction leaves the iterate unchanged, and the adapter can stop or
switch to a fallback child. An adapter has the same interface as any other
solver node, so adapters can be placed at any compatible set of edges in a
solver tree and nested to any depth.

This repository has the reference implementation in Python and PyTorch, the
code for every experiment in the paper, the archived results behind every
number in it, and a verification suite that reruns the key checks on any
CUDA GPU.

> Kevin Li and Chandrajit Bajaj, *CAMPS: Composable Adapters for
> Mixed-Precision Solvers*, preprint, 2026. The University of Texas at Austin.

## Solver trees

A tree is written from the root down. For example,
`FGMRES64[IR64←32[LU32]]` is FGMRES in FP64 whose preconditioner is an
adapter from FP64 to FP32, and the adapter's child is an FP32 LU solve. The
paper names five instances.

| Instance | Tree | Used for |
|---|---|---|
| CAMPS-Direct | `IR64←32[S32]` | dense systems with an LU32 child, SPD systems with a CG32 child |
| CAMPS-Krylov | `FGMRES64[IR64←32[S32]]` | the same children, with FGMRES at the root |
| CAMPS-Cascade | `FGMRES64[IR64←32[FGMRES32[IR32←16[S16]]]]` | adapters at both precision boundaries, with an FP16 leaf |
| CAMPS-Apex | `IR64←32[FGMRES32^m[cast32←16[FGMRES16^m[Richardson16]]]]` | F3R's inner levels with an adapter in place of its outer FGMRES64 |
| CAMPS-Ladder | `FGMRES64[IR64←32[LU32 \| LU64 fallback] \| LU64]` | dense systems of unknown conditioning |

In the code, `←` is written `<-`, and these strings are the `method` field
of every record in `results/`.

## Quick start

Install a CUDA build of PyTorch and the pinned packages:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Solve a dense system with CAMPS-Direct, from the repository root:

```python
import torch
from camps import AdapterConfig, CampsAdapter, DenseLUSolver, DenseOperator, configure_backend

configure_backend()                      # FP32 means IEEE FP32 (TF32 off)
n = 2048
A = torch.randn(n, n, dtype=torch.float64, device="cuda") / n**0.5
A += 4 * torch.eye(n, dtype=torch.float64, device="cuda")
b = torch.randn(n, dtype=torch.float64, device="cuda")

parent = DenseOperator(A)                # residuals and updates in FP64
child = DenseLUSolver(DenseOperator(A.float()))   # FP32 LU factorization
solver = CampsAdapter(parent, child, AdapterConfig(J=30, tau=1e-10))

result = solver.solve(b)                 # IR64<-32[LU32]
x = result.correction
print(result.status, result.accepted)    # Status.SUCCESS [True, True]
```

`result.residual_history` holds the FP64 residual after each accepted
correction, and `result.child_results` holds what the child returned each
time.

## Reproducing the paper

```bash
python -m experiments.run_experiments --exp all          # full sweep
python -m experiments.run_experiments --exp all --pilot  # reduced sizes, a few minutes
python -m experiments.verify_recursion                   # recursive-bound check, about 1 min
python -m experiments.verify_all                         # verification report, about 20 min
python -m experiments.analyze                            # tables and figures into docs/
```

Results are written to `results/` by default. Set `CAMPS_RESULTS_DIR` to
write somewhere else and keep the archived files untouched.
`REPRODUCIBILITY.md` lists, for every result file, the command that
regenerates it, its seeds, the measurement protocol, and the recorded
environment.

| Experiment | Problem | Sweep | Paper |
|---|---|---|---|
| `dense_wc` | dense, $\kappa=10^{2}$ | $n=2048, 4096, 8192$ | §5.3 |
| `multirhs` | dense, $n=4096$ | $k=1, 4, 8, 16$ right-hand sides | §5.3 |
| `dense_ic` | dense, $n=2048$ | $\kappa=10^{4}$ to $10^{12}$, with fallback, ladder, and damping | §5.4 |
| `sparse_spd` | 3D Poisson | $32^3$, $48^3$, $64^3$ grids | §5.5 |
| `sparse_nsy` | 3D convection–diffusion | $w=2, 20, 100$ on $32^3$, $w=20$ on $48^3$ | §5.6 |
| `placement` | dense, $n=2048$ | one approximate-inverse child at different edges, $\kappa=10^{2}, 10^{4}, 10^{6}$ | §5.7 |
| `f3r` | Poisson and convection–diffusion | $32^3$, $48^3$, each tree tuned on $32^3$ | §5.8 |
| `verify_recursion` | dense, $n=512$ | 57 configurations of an FP64←FP32←FP16 adapter chain | §5.9 |
| `energy_check` | Poisson | residual and energy contracts on SPD problems | App. A.10 |

The dense matrices are $A=U\Sigma V^{\mathsf T}$ with random orthogonal $U$
and $V$ and singular values spaced evenly on a log scale, so the condition
number is set by construction. Every method stops at the same relative
residual, $\|b-Ax\|_2/\|b\|_2 \le 10^{-10}$, recomputed in FP64.

## Hardware

Every timing in the paper was measured on an NVIDIA GeForce RTX 5070 Ti
Laptop GPU (CUDA 12.8, torch 2.11.0+cu128). FP64 runs at 1/64 of the FP32
rate on this GPU, which favors lower-precision factorizations. On a GPU with
full-rate FP64, the dense speedups would be smaller. Iteration counts,
acceptance decisions, and residuals do not depend on timing and are
reproduced bit for bit on the same GPU.

We also ran the verification suite on an NVIDIA A100 node of Lonestar6 at
the Texas Advanced Computing Center, and it passed every check. The report
and the A100 results are in `results_ls6_a100/`, and `experiments/tacc/`
has the setup script and Slurm jobs to run it on a TACC system.

## Repository layout

- `camps/` is the core package.
  - `adapter.py` implements the adapter (Algorithm 1 of the paper): power-of-two
    scaling, the acceptance test, rollback, fallback children, stagnation
    detection, optional damping, and an energy test for SPD parents.
  - `contracts.py` has the residual contracts, and `scaling.py` the
    power-of-two scaling.
  - `gmres.py`, `fgmres.py`, `cg.py`, and `base_solvers.py` are the solver
    nodes (GMRES, FGMRES, CG, LU, Cholesky, Jacobi, approximate inverse).
  - `operators.py` and `precision.py` handle operators and explicit casts. A
    node that receives a vector in the wrong format raises an error instead
    of casting it.
  - `diagnostics.py` turns off TF32 and reduced-precision reductions.
- `experiments/` has the experiments.
  - `matrices.py` builds the test matrices, `nodes.py` the sparse operators
    and counting wrappers, and `instances.py` every CAMPS tree and baseline.
  - `harness.py` is the measurement protocol: warmups, medians over repeated
    runs, setup timed separately, the FP64 residual check, and memory
    accounting.
  - `run_experiments.py` runs the sweeps, and `verify_recursion.py`,
    `energy_check.py`, `regress_fallback.py`, `rerun_noboost.py`, and
    `side_checks.py` run the remaining checks in the paper.
  - `verify_all.py` runs the verification suite and writes
    `verification_report.md`.
  - `analyze.py` builds tables and figures, and `make_manifest.py` rebuilds
    `REPRODUCIBILITY.md`.
  - `tacc/` has the scripts for TACC systems.
- `results/` has the archived JSON results, each with its full environment
  record.
- `results_ls6_a100/` has the A100 verification run.
- `docs/` has tables and figures generated from the results.

## Citation

```bibtex
@misc{li2026camps,
  author = {Kevin Li and Chandrajit Bajaj},
  title  = {{CAMPS}: Composable Adapters for Mixed-Precision Solvers},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/CVC-Lab/CAMPS}
}
```

## License

MIT. See `LICENSE`.
