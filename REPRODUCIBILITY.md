# Reproducibility manifest

Every number in the paper comes from a JSON file in `results/`. This file lists, for each of them, the command that regenerates it, the seeds it used, the measurement protocol, the environment recorded at run time, and where it is used in the paper. Timings will differ on other hardware; iteration counts, acceptance decisions, and residuals are deterministic for the recorded seeds and were reproduced bit for bit on reruns (see `experiments/regress_fallback.py` and the comparison in the verify_recursion log).

Install with `pip install -r requirements.txt` (CUDA build of torch required). `python -m experiments.run_experiments --exp all` regenerates the seven sweep files; the remaining files have their own commands below. `python -m experiments.analyze` rebuilds `docs/tables/` and `docs/figures/`.

## `dense_wc.json`

- **Regenerate:** `python -m experiments.run_experiments --exp dense_wc`
- **Seeds:** dense_conditioned(n, kappa=1e2, seed=100+n)
- **Used in:** Section 5.3, dense well-conditioned table
- **Records:** 16 (3 problems x methods)
- **Protocol:** warmup=3, timed reps=15, setup reps=3, tau=1e-10
- **Problems:** `dense-nonsym-n2048-k1e+02`, `dense-nonsym-n4096-k1e+02`, `dense-nonsym-n8192-k1e+02`
- **Methods:** `FGMRES64(LU64)`, `FGMRES64[IR64<-32[FGMRES32[IR32<-16[INV16]]]]`, `FGMRES64[IR64<-32[FGMRES32[IR32<-b16[INVb16]]]]`, `FGMRES64[IR64<-32[LU32]]`, `IR64<-32[LU32] (plain IR)`, `LU64 (direct)`
- **Environment:** NVIDIA GeForce RTX 5070 Ti Laptop GPU; driver 610.47, 12227 MiB; CUDA 12.8; torch 2.11.0+cu128; Python 3.11.9; Windows-10-10.0.26200-SP0; TF32 matmul False, fp16 reduced-precision reductions False, float32 matmul precision 'highest'

## `dense_ic.json`

- **Regenerate:** `python -m experiments.run_experiments --exp dense_ic`
- **Seeds:** dense_conditioned(2048, kappa, seed=200+log10(kappa))
- **Used in:** Section 5.4, ill-conditioned sweep table (rows not involving an LU64 child; LU64 rows at kappa <= 1e8)
- **Records:** 60 (6 problems x methods)
- **Protocol:** warmup=3, timed reps=15, setup reps=3, tau=1e-10
- **Problems:** `dense-nonsym-n2048-k1e+04`, `dense-nonsym-n2048-k1e+06`, `dense-nonsym-n2048-k1e+07`, `dense-nonsym-n2048-k1e+08`, `dense-nonsym-n2048-k1e+10`, `dense-nonsym-n2048-k1e+12`
- **Methods:** `FGMRES64(LU64)`, `FGMRES64[IR64<-32[LU32]]`, `FGMRES64[IR64<-32[LU32]] (loose)`, `FGMRES64[IR64<-32[LU32]] damped`, `FGMRES64[IR64<-32[LU32|LU64 fb]|dir(LU64)] (ladder)`, `FGMRES64[IR64<-32[LU32|LU64 fb]|dir(LU64)] (ladder) damped`, `IR64<-32[LU32 | LU64 fallback]`, `IR64<-32[LU32] (plain IR)`, `IR64<-32[LU32] (plain IR) damped`, `LU64 (direct)`
- **Environment:** NVIDIA GeForce RTX 5070 Ti Laptop GPU; driver 610.47, 12227 MiB; CUDA 12.8; torch 2.11.0+cu128; Python 3.11.9; Windows-10-10.0.26200-SP0; TF32 matmul False, fp16 reduced-precision reductions False, float32 matmul precision 'highest'

## `dense_ic_noboost.json`

- **Regenerate:** `python -m experiments.rerun_noboost`
- **Seeds:** same problems and seeds as dense_ic.json (kappa = 1e10, 1e12)
- **Used in:** Section 5.4, LU64-dependent rows at kappa = 1e10 and 1e12, thresholded damped fallback row
- **Records:** 12 (2 problems x methods)
- **Protocol:** warmup=3, timed reps=15, setup reps=3, tau=1e-10
- **Problems:** `dense-nonsym-n2048-k1e+10`, `dense-nonsym-n2048-k1e+12`
- **Methods:** `FGMRES64(LU64)`, `FGMRES64[IR64<-32[LU32|LU64 fb]|dir(LU64)] (ladder)`, `FGMRES64[IR64<-32[LU32|LU64 fb]|dir(LU64)] (ladder) damped`, `IR64<-32[LU32 | LU64 fallback]`, `IR64<-32[LU32 | LU64 fallback] damped rho=0.9`, `LU64 (direct)`
- **Note:** LU64 nodes with pivot_tol=0 (no pivot boosting); kappa 1e10 and 1e12
- **Environment:** NVIDIA GeForce RTX 5070 Ti Laptop GPU; driver 610.47, 12227 MiB; CUDA 12.8; torch 2.11.0+cu128; Python 3.11.9; Windows-10-10.0.26200-SP0; TF32 matmul False, fp16 reduced-precision reductions False, float32 matmul precision 'highest'

## `dense_ic_side_checks.json`

- **Regenerate:** `python -m experiments.side_checks`
- **Seeds:** same problems and seeds as dense_ic.json
- **Used in:** Section 5.4, thresholded damped fallback at kappa <= 1e8 (the file also records pivot-boost counts, 278 at kappa = 1e12, which the paper does not report)
- **Contents:** `1e+04`, `1e+06`, `1e+07`, `1e+08`, `1e+10`, `1e+12`
- **Environment:** NVIDIA GeForce RTX 5070 Ti Laptop GPU; driver None; CUDA None; torch 2.11.0+cu128; Python None; None; TF32 matmul False, fp16 reduced-precision reductions False, float32 matmul precision 'highest'

## `sparse_spd.json`

- **Regenerate:** `python -m experiments.run_experiments --exp sparse_spd`
- **Seeds:** sparse_problem('poisson', g, seed=300+g)
- **Used in:** Section 5.5, Poisson table
- **Records:** 18 (3 problems x methods)
- **Protocol:** warmup=3, timed reps=10, setup reps=3, tau=1e-10
- **Problems:** `poisson3d-g32`, `poisson3d-g48`, `poisson3d-g64`
- **Methods:** `CG64`, `CG64(Jacobi64)`, `FGMRES64(Jacobi64)`, `FGMRES64[IR64<-32[CG32(Jacobi32)]]`, `FGMRES64[IR64<-32[FGMRES32[IR32<-16[FGMRES16(Jacobi16)]]]]`, `IR64<-32[CG32(Jacobi32)] (plain IR)`
- **Environment:** NVIDIA GeForce RTX 5070 Ti Laptop GPU; driver 610.47, 12227 MiB; CUDA 12.8; torch 2.11.0+cu128; Python 3.11.9; Windows-10-10.0.26200-SP0; TF32 matmul False, fp16 reduced-precision reductions False, float32 matmul precision 'highest'

## `spd_energy_check.json`

- **Regenerate:** `python -m experiments.energy_check`
- **Seeds:** sparse_problem('poisson', g, seed=300+g); counterexample is exact
- **Used in:** Section 4.3 counterexample; Appendix A.10 energy-contract margins and cost
- **Contents:** `g32_residual`, `g32_energy`, `g48_residual`, `g48_energy`, `counterexample`, `energy_margins`
- **Environment:** NVIDIA GeForce RTX 5070 Ti Laptop GPU; driver None; CUDA None; torch 2.11.0+cu128; Python None; None; TF32 matmul False, fp16 reduced-precision reductions False, float32 matmul precision 'highest'

## `sparse_nsy.json`

- **Regenerate:** `python -m experiments.run_experiments --exp sparse_nsy`
- **Seeds:** sparse_problem('convdiff', g, w, seed=400+g)
- **Used in:** Section 5.6, convection-diffusion table and wind sweep
- **Records:** 16 (4 problems x methods)
- **Protocol:** warmup=3, timed reps=10, setup reps=3, tau=1e-10
- **Problems:** `convdiff3d-g32-w100`, `convdiff3d-g32-w2`, `convdiff3d-g32-w20`, `convdiff3d-g48-w20`
- **Methods:** `FGMRES64(Jacobi64)`, `FGMRES64[IR64<-32[FGMRES32(Jacobi32)]]`, `FGMRES64[IR64<-32[FGMRES32[IR32<-16[FGMRES16(Jacobi16)]]]]`, `GMRES64(Jacobi64)`
- **Environment:** NVIDIA GeForce RTX 5070 Ti Laptop GPU; driver 610.47, 12227 MiB; CUDA 12.8; torch 2.11.0+cu128; Python 3.11.9; Windows-10-10.0.26200-SP0; TF32 matmul False, fp16 reduced-precision reductions False, float32 matmul precision 'highest'

## `placement.json`

- **Regenerate:** `python -m experiments.run_experiments --exp placement`
- **Seeds:** dense_conditioned(2048, kappa, seed=500)
- **Used in:** Section 5.7, edge-placement table
- **Records:** 18 (3 problems x methods)
- **Protocol:** warmup=3, timed reps=15, setup reps=3, tau=1e-10
- **Problems:** `dense-nonsym-n2048-k1e+02`, `dense-nonsym-n2048-k1e+04`, `dense-nonsym-n2048-k1e+06`
- **Methods:** `FGMRES64[IR64<-16[INV16]]`, `FGMRES64[IR64<-32[FGMRES32[IR32<-16[INV16]]]]`, `FGMRES64[IR64<-32[FGMRES32[IR32<-b16[INVb16]]]]`, `FGMRES64[IR64<-32[INV32]]`, `FGMRES64[IR64<-32[IR32<-16[INV16]]]`, `FGMRES64[IR64<-b16[INVb16]]`
- **Environment:** NVIDIA GeForce RTX 5070 Ti Laptop GPU; driver 610.47, 12227 MiB; CUDA 12.8; torch 2.11.0+cu128; Python 3.11.9; Windows-10-10.0.26200-SP0; TF32 matmul False, fp16 reduced-precision reductions False, float32 matmul precision 'highest'

## `f3r.json`

- **Regenerate:** `python -m experiments.run_experiments --exp f3r`
- **Seeds:** sparse_problem(kind, g, w, seed=600+g); tuning grids F3R_GRID, HYBRID_GRID, HIER_GRID, IRTOP_GRID in run_experiments.py; selected configurations stored under 'tuned_configs'
- **Used in:** Section 5.8, F3R comparison table
- **Records:** 22 (4 problems x methods)
- **Protocol:** warmup=3, timed reps=10, setup reps=3, tau=1e-10
- **Problems:** `convdiff3d-g32-w20`, `convdiff3d-g48-w20`, `poisson3d-g32`, `poisson3d-g48`
- **Methods:** `CG64`, `F3R: FGMRES64[FGMRES32^10[FGMRES16^5[Rich16x2]]]`, `F3R: FGMRES64[FGMRES32^10[FGMRES16^5[Rich16x4]]]`, `FGMRES64(Jacobi64)`, `FGMRES64[IR64<-32[FGMRES32[IR32<-16[FGMRES16(Jacobi16)]]]]`, `IR64<-32[FGMRES32^10[cast32<-16[FGMRES16^5[Rich16x2]]]]`, `hybrid: FGMRES64[IR64<-32[FGMRES32[cast32<-16[FGMRES16^5]]]]`
- **Selected configurations (fair tuning, fastest converged):** `{"poisson": {"f3r": {"m2": 10, "m3": 5, "sweeps": 2}, "hybrid": {"m3": 5, "mid_maxiter": 10}, "hier": {"J_outer": 1, "J_inner": 1, "mid_tol": 0.001, "mid_restart": 10, "mid_maxiter": 10, "leaf_tol": 0.1, "leaf_restart": 5, "leaf_maxiter": 5}, "irtop": {"m2": 10, "m3": 5, "sweeps": 2}}, "convdiff": {"f3r": {"m2": 10, "m3": 5, "sweeps": 4}, "hybrid": {"m3": 5, "mid_maxiter": 15}, "hier": {}, "irtop": {"m2": 10, "m3": 5, "sweeps": 2}}}`
- **Environment:** NVIDIA GeForce RTX 5070 Ti Laptop GPU; driver 610.47, 12227 MiB; CUDA 12.8; torch 2.11.0+cu128; Python 3.11.9; Windows-10-10.0.26200-SP0; TF32 matmul False, fp16 reduced-precision reductions False, float32 matmul precision 'highest'

## `multirhs.json`

- **Regenerate:** `python -m experiments.run_experiments --exp multirhs`
- **Seeds:** dense_conditioned(4096, kappa=1e2, seed=700); right-hand-side block from torch.Generator seed 701
- **Used in:** Section 5.3, multi-RHS amortization table
- **Records:** 20 (1 problems x methods)
- **Protocol:** warmup=2, timed reps=5, setup reps=3, tau=1e-10
- **Problems:** `dense-nonsym-n4096-k1e+02`
- **Methods:** `FGMRES64(LU64)`, `FGMRES64[IR64<-32[FGMRES32[IR32<-16[INV16]]]]`, `FGMRES64[IR64<-32[LU32]]`, `IR64<-32[LU32] (plain IR)`, `LU64 (direct)`
- **Environment:** NVIDIA GeForce RTX 5070 Ti Laptop GPU; driver 610.47, 12227 MiB; CUDA 12.8; torch 2.11.0+cu128; Python 3.11.9; Windows-10-10.0.26200-SP0; TF32 matmul False, fp16 reduced-precision reductions False, float32 matmul precision 'highest'

## `verify_recursion.json`

- **Regenerate:** `python -m experiments.verify_recursion`
- **Seeds:** dense_conditioned(512, kappa, seed=300+10*log10(kappa)+{0,1})
- **Used in:** Section 5.9, recursive-bound verification table
- **Contents:** `cases`
- **Environment:** NVIDIA GeForce RTX 5070 Ti Laptop GPU; driver None; CUDA None; torch 2.11.0+cu128; Python None; None; TF32 matmul False, fp16 reduced-precision reductions False, float32 matmul precision 'highest'

## Tuning grids (f3r.json)

From `experiments/run_experiments.py`: `F3R_GRID` = m2 in {10,20,30} x m3 in {5,10} x sweeps in {2,4}; `HYBRID_GRID` = {(m3=5, mid_maxiter=15), (5,10), (8,10)}; `IRTOP_GRID` = m2 in {10,15,20} x m3 in {5,8}, sweeps=2; `HIER_GRID` = four configurations listed in the script. Each candidate is run once (1 warmup, 2 timed) and the fastest converged configuration is kept, identically for every method.

## Fixed quantities

- Stopping rule everywhere: FP64-recomputed relative residual <= 1e-10, verified after the solve by an independent FP64 residual.
- Backend flags (`camps/diagnostics.py`): TF32 disabled for matmul and cuDNN, reduced-precision fp16/bf16 reductions disabled, float32 matmul precision 'highest', so FP32 means IEEE binary32 arithmetic.
- Dense LU nodes: partial pivoting, no pivot modification (`lu_pivot_tol=0`); the zero-pivot boost (1e-8 threshold, 1e-6 replacement, relative to the largest pivot) is retained only for ILU(0).
- Adapter defaults: power-of-two right-hand-side scaling, residual contract with slack of a few parent ulps, stagnation window 3 for plain-IR instances, attempt budgets as given per instance in `experiments/instances.py`.
