# CAMPS verification report

- host: v330-014.ls6.tacc.utexas.edu  
- platform: Linux-4.18.0-553.150.1.el8_10.x86_64-x86_64-with-glibc2.28  
- python: 3.11.17  
- torch: 2.11.0+cu128  
- GPU: NVIDIA A100-PCIE-40GB  
- driver / memory: 570.195.03, 40960 MiB  
- output directory: `$WORK/CAMPS/verification_v330-014`  
- archive compared against: `$WORK/CAMPS/results`  
- date: 2026-10-16 10:47:09

## Summary

| check | result | wall time |
|---|---|---|
| recursive bounds | PASS | 0.9 min |
| certified energy contract | PASS | 0.3 min |
| fallback/ladder rows | PASS | 0.2 min |
| pilot sweep | PASS | 2.3 min |

**Overall: PASS**

### Recursive bounds (Theorem 4, Section 5.8)

| kappa | leaf damp | adversarial | J1 | eta_top new | eta_top archive | beta_root | bounds respected | status | row |
|---|---|---|---|---|---|---|---|---|---|
| 1e+01 | 1.0 | False | 1 | 5.34e-04 | 5.34e-04 | 0.72 u64 | yes | rejected | ok |
| 1e+01 | 1.0 | False | 2 | 3.16e-07 | 3.22e-07 | 0.62 u64 | yes | rejected | ok |
| 1e+01 | 1.0 | False | 3 | 1.11e-07 | 1.31e-07 | 0.67 u64 | yes | rejected | ok |
| 1e+01 | 1.0 | False | 4 | 1.09e-07 | 1.23e-07 | 0.70 u64 | yes | rejected | ok |
| 1e+01 | 1.0 | False | 6 | 1.12e-07 | 1.27e-07 | 0.68 u64 | yes | rejected | ok |
| 1e+01 | 1.0 | False | 8 | 1.12e-07 | 1.27e-07 | 0.68 u64 | yes | rejected | ok |
| 1e+01 | 0.5 | False | 1 | 5.00e-01 | 5.00e-01 | 0.57 u64 | yes | rejected | ok |
| 1e+01 | 0.5 | False | 2 | 2.50e-01 | 2.50e-01 | 0.59 u64 | yes | rejected | ok |
| 1e+01 | 0.5 | False | 4 | 6.25e-02 | 6.25e-02 | 0.66 u64 | yes | rejected | ok |
| 1e+01 | 0.5 | False | 8 | 3.91e-03 | 3.91e-03 | 0.68 u64 | yes | rejected | ok |
| 1e+01 | 0.5 | False | 16 | 1.53e-05 | 1.53e-05 | 0.71 u64 | yes | rejected | ok |
| 1e+01 | 0.5 | False | 24 | 9.47e-08 | 1.10e-07 | 0.69 u64 | yes | rejected | ok |
| 1e+01 | 0.5 | False | 32 | 9.47e-08 | 8.86e-08 | 0.70 u64 | yes | rejected | ok |
| 1e+01 | 1.0 | True | 1 | 1.03e-03 | 1.03e-03 | 0.83 u64 | yes | rejected | ok |
| 1e+01 | 1.0 | True | 2 | 7.67e-07 | 7.46e-07 | 0.81 u64 | yes | rejected | ok |
| 1e+01 | 1.0 | True | 3 | 4.44e-07 | 4.38e-07 | 0.79 u64 | yes | rejected | ok |
| 1e+01 | 1.0 | True | 4 | 4.03e-07 | 4.47e-07 | 0.78 u64 | yes | rejected | ok |
| 1e+01 | 1.0 | True | 6 | 4.03e-07 | 4.47e-07 | 0.81 u64 | yes | rejected | ok |
| 1e+01 | 1.0 | True | 8 | 4.03e-07 | 4.47e-07 | 0.81 u64 | yes | rejected | ok |
| 1e+02 | 1.0 | False | 1 | 2.26e-03 | 2.26e-03 | 0.58 u64 | yes | rejected | ok |
| 1e+02 | 1.0 | False | 2 | 5.06e-06 | 5.11e-06 | 0.56 u64 | yes | rejected | ok |
| 1e+02 | 1.0 | False | 3 | 1.11e-07 | 1.32e-07 | 0.55 u64 | yes | rejected | ok |
| 1e+02 | 1.0 | False | 4 | 1.11e-07 | 1.32e-07 | 0.58 u64 | yes | rejected | ok |
| 1e+02 | 1.0 | False | 6 | 1.11e-07 | 1.32e-07 | 0.58 u64 | yes | rejected | ok |
| 1e+02 | 1.0 | False | 8 | 1.11e-07 | 1.32e-07 | 0.58 u64 | yes | rejected | ok |
| 1e+02 | 0.5 | False | 1 | 5.00e-01 | 5.00e-01 | 0.45 u64 | yes | rejected | ok |
| 1e+02 | 0.5 | False | 2 | 2.50e-01 | 2.50e-01 | 0.51 u64 | yes | rejected | ok |
| 1e+02 | 0.5 | False | 4 | 6.25e-02 | 6.25e-02 | 0.54 u64 | yes | rejected | ok |
| 1e+02 | 0.5 | False | 8 | 3.90e-03 | 3.90e-03 | 0.54 u64 | yes | rejected | ok |
| 1e+02 | 0.5 | False | 16 | 1.52e-05 | 1.52e-05 | 0.54 u64 | yes | rejected | ok |
| 1e+02 | 0.5 | False | 24 | 9.75e-08 | 1.00e-07 | 0.57 u64 | yes | rejected | ok |
| 1e+02 | 0.5 | False | 32 | 7.59e-08 | 1.00e-07 | 0.55 u64 | yes | rejected | ok |
| 1e+02 | 1.0 | True | 1 | 7.35e-03 | 7.34e-03 | 0.66 u64 | yes | rejected | ok |
| 1e+02 | 1.0 | True | 2 | 1.68e-05 | 1.69e-05 | 0.63 u64 | yes | rejected | ok |
| 1e+02 | 1.0 | True | 3 | 3.23e-06 | 3.30e-06 | 0.65 u64 | yes | rejected | ok |
| 1e+02 | 1.0 | True | 4 | 2.98e-06 | 3.28e-06 | 0.67 u64 | yes | rejected | ok |
| 1e+02 | 1.0 | True | 6 | 2.98e-06 | 3.28e-06 | 0.67 u64 | yes | rejected | ok |
| 1e+02 | 1.0 | True | 8 | 2.98e-06 | 3.28e-06 | 0.67 u64 | yes | rejected | ok |
| 1e+03 | 1.0 | False | 1 | 1.50e-02 | 1.50e-02 | 0.48 u64 | yes | rejected | ok |
| 1e+03 | 1.0 | False | 2 | 2.16e-04 | 2.16e-04 | 0.48 u64 | yes | rejected | ok |
| 1e+03 | 1.0 | False | 3 | 3.20e-06 | 3.20e-06 | 0.50 u64 | yes | rejected | ok |
| 1e+03 | 1.0 | False | 4 | 1.31e-07 | 1.38e-07 | 0.49 u64 | yes | rejected | ok |
| 1e+03 | 1.0 | False | 6 | 1.12e-07 | 1.24e-07 | 0.50 u64 | yes | rejected | ok |
| 1e+03 | 1.0 | False | 8 | 1.12e-07 | 1.24e-07 | 0.50 u64 | yes | rejected | ok |
| 1e+03 | 0.5 | False | 1 | 5.00e-01 | 5.00e-01 | 0.39 u64 | yes | rejected | ok |
| 1e+03 | 0.5 | False | 2 | 2.50e-01 | 2.50e-01 | 0.42 u64 | yes | rejected | ok |
| 1e+03 | 0.5 | False | 4 | 6.25e-02 | 6.25e-02 | 0.47 u64 | yes | rejected | ok |
| 1e+03 | 0.5 | False | 8 | 3.91e-03 | 3.91e-03 | 0.47 u64 | yes | rejected | ok |
| 1e+03 | 0.5 | False | 16 | 1.55e-05 | 1.55e-05 | 0.48 u64 | yes | rejected | ok |
| 1e+03 | 0.5 | False | 24 | 9.35e-08 | 9.32e-08 | 0.48 u64 | yes | rejected | ok |
| 1e+03 | 0.5 | False | 32 | 6.81e-08 | 9.32e-08 | 0.48 u64 | yes | rejected | ok |
| 1e+03 | 1.0 | True | 1 | 5.26e-02 | 5.26e-02 | 0.60 u64 | yes | rejected | ok |
| 1e+03 | 1.0 | True | 2 | 8.13e-04 | 8.10e-04 | 0.55 u64 | yes | rejected | ok |
| 1e+03 | 1.0 | True | 3 | 2.95e-05 | 2.85e-05 | 0.60 u64 | yes | rejected | ok |
| 1e+03 | 1.0 | True | 4 | 2.55e-05 | 2.64e-05 | 0.62 u64 | yes | rejected | ok |
| 1e+03 | 1.0 | True | 6 | 2.55e-05 | 2.64e-05 | 0.60 u64 | yes | rejected | ok |
| 1e+03 | 1.0 | True | 8 | 2.55e-05 | 2.64e-05 | 0.60 u64 | yes | rejected | ok |

### Certified energy contract (Section 4.3, Section 5.4)

- counterexample A=1, y=2^53, d=1/4: proposed-step test accepts = True, certified test accepts = False (expected True / False)
| grid | contract | converged | corrections | accepted/rejected | rel. residual new | archive | solve ms new | archive |
|---|---|---|---|---|---|---|---|---|
| 32^3 | residual | True | 3 | 3/0 | 8.98e-13 | 8.96e-13 | 108.4 | 118.4 |
| 32^3 | energy | True | 3 | 3/0 | 8.98e-13 | 8.96e-13 | 110.8 | 150.7 |
| 48^3 | residual | True | 3 | 3/0 | 8.79e-13 | 8.79e-13 | 166.5 | 201.9 |
| 48^3 | energy | True | 3 | 3/0 | 8.79e-13 | 8.79e-13 | 169.0 | 225.7 |
- largest bound/|dphi| over all certified corrections: 1.92e-11 (must be < 1e-3)

### Fallback, damped fallback, and ladder rows (Section 5.3)

| row | new: status, attempts, acc/rej, residual | archive | match |
|---|---|---|---|
| k=1e+08 IR64<-32[LU32 | LU64 fallback] | success, 3, 2/1, 3.639882e-13 | success, 3, 2/1, 2.278594e-12 | ok |
| k=1e+08 FGMRES64[IR64<-32[LU32|LU64 fb]|dir(LU64)] (ladder) | success, 2, 2/2, 2.201827e-13 | success, 2, 2/2, 4.816875e-13 | ok |
| k=1e+10 IR64<-32[LU32 | LU64 fallback] | success, 3, 2/1, 6.521562e-13 | success, 3, 2/1, 4.691310e-13 | ok |
| k=1e+10 IR64<-32[LU32 | LU64 fallback] damped rho=0.9 | success, 3, 2/1, 6.720975e-13 | success, 3, 2/1, 3.902344e-13 | ok |
| k=1e+10 FGMRES64[IR64<-32[LU32|LU64 fb]|dir(LU64)] (ladder) | success, 2, 2/2, 3.059636e-13 | success, 2, 2/2, 3.704654e-13 | ok |
| k=1e+12 IR64<-32[LU32 | LU64 fallback] | success, 3, 2/1, 3.978626e-13 | success, 3, 2/1, 1.872327e-12 | ok |
| k=1e+12 IR64<-32[LU32 | LU64 fallback] damped rho=0.9 | success, 3, 2/1, 3.456223e-13 | success, 3, 2/1, 1.969472e-12 | ok |
| k=1e+12 FGMRES64[IR64<-32[LU32|LU64 fb]|dir(LU64)] (ladder) | success, 2, 2/2, 5.162398e-13 | success, 2, 2/2, 1.097814e-12 | ok |

### Pilot sweep (all seven experiments, reduced sizes)

- converged runs: 52; non-converged: 5; exceptions: 0 (any exception fails the check)

