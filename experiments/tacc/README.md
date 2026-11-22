# Verifying CAMPS on TACC

Everything in the paper was measured on one laptop GPU. This directory lets a
second person reproduce it on a TACC system with one job and compare the
outcome against the archived results automatically.

## 0. Where this runs

The TACC portal (portal.tacc.utexas.edu) only manages accounts and
allocations; nothing is submitted there. The work happens over SSH on a
system's login nodes. You need a TACC project with a GPU allocation. The
scripts default to Lonestar6 (A100, x86_64), where we ran them, and the queue
line in each script lists the queue names for Vista and Stampede3.

```bash
ssh <username>@ls6.tacc.utexas.edu      # password, then the TACC Token code (MFA must be paired in the portal first)
```

## 1. One-time setup (login node, about 10 minutes)

```bash
cdw                                     # = cd $WORK
git clone https://github.com/CVC-Lab/CAMPS.git
cd CAMPS
bash experiments/tacc/setup_env.sh      # torch 2.11 (cu128) in $WORK/camps-venv, or a Miniforge env if python3 < 3.10
```

## 2. Verification job (about 20 to 40 minutes on one GPU)

`verify.slurm` uses the one-GPU queue `gpu-a100-small`. Pass your allocation
with `-A`, or edit the `#SBATCH -A` line in the script. Submit, watch, and
read:

```bash
sbatch -A <your-allocation> experiments/tacc/verify.slurm
squeue -u $USER                          # PD = waiting in queue, R = running
cat camps-verify.<jobid>.out             # progress; the job id is printed by sbatch
```

The job writes to `$WORK/CAMPS/verification_<node>/` and ends with
`verification_report.md`. The report records the host, GPU, driver, and
software versions, then one PASS/FAIL line per check:

| check | what must hold on any GPU |
|---|---|
| recursive bounds | no measured contraction factor exceeds its rigorous bound; root backward error is O(u64); exit statuses match the archive (Theorem 4, Section 5.9) |
| energy contract | the rounded-displacement counterexample is rejected; both SPD contracts converge in the same number of corrections with every correction accepted (Section 4.3, Appendix A.10) |
| fallback/ladder rows | statuses, attempt counts and accepted/rejected splits match the archive at kappa = 1e8, 1e10, 1e12 (Section 5.4) |
| pilot sweep | all seven experiments run at reduced size without exceptions |

Timings are reported next to the archived ones but never asserted: they are
hardware-dependent, and a datacenter GPU with full-rate FP64 (A100, H100,
GH200) will show much smaller dense speedups than the consumer GPU in the
paper, whose FP64 rate is 1/64 of its FP32 rate. Residuals and contraction
factors may differ in their trailing digits between GPUs; the checks require
agreement within a factor of ten.

To let another member of the project read the run directly on the machine:

```bash
chmod -R g+rX $WORK/CAMPS                # project members share a Unix group
```

Otherwise copy the report back and commit it, e.g. into `results_ls6/`.

## 3. Full sweep (several hours; optional)

```bash
sbatch -A <your-allocation> experiments/tacc/full_sweep.slurm
```

Regenerates every results file at full size into `$WORK/CAMPS/results_<node>/`
and adds a convergence-agreement table to the report: every method that
converged in the archive must converge with the same status. This is also the
"second architecture" data set the reviews asked for.

## 4. Reading the code against the paper

| paper | code |
|---|---|
| Algorithm 1 (adapter: scaling, cast, acceptance, rollback, fallback, damping, energy test) | `camps/adapter.py`, `CampsAdapter.solve` and `energy_certificate` |
| residual contract with slack (Section 4.3) | `camps/contracts.py` |
| power-of-two scaling and conversion reports | `camps/scaling.py`, `camps/precision.py` |
| FGMRES with stored directions and FP64 restart residuals (Section 4.6) | `camps/fgmres.py` |
| Algorithms 2 to 6 (named instances) and every baseline | `experiments/instances.py` |
| matrix classes with exactly controlled conditioning; 3D stencils | `experiments/matrices.py` |
| protocol: warmups, medians, FP64 recomputed residuals, backward errors, memory, per-precision matvec counts | `experiments/harness.py` |
| Section 5.9 verification of Theorem 4 | `experiments/verify_recursion.py` |
| Section 4.3 counterexample and Appendix A.10 energy margins | `experiments/energy_check.py` |
| Section 5.4 no-boost reruns and pivot-boost counts | `experiments/side_checks.py`, `experiments/rerun_noboost.py` |
| `REPRODUCIBILITY.md` (seeds, protocol, environment per result file) | `experiments/make_manifest.py` |
