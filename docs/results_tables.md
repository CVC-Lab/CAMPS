# CAMPS matrix-class sweep - results

## Environment

```
{
 "gpu": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
 "torch": "2.11.0+cu128",
 "cuda": "12.8",
 "platform": "Windows-10-10.0.26200-SP0",
 "python": "3.11.9",
 "backend_flags": {
  "cuda.matmul.allow_tf32": false,
  "cudnn.allow_tf32": false,
  "cuda.matmul.allow_fp16_reduced_precision_reduction": false,
  "cuda.matmul.allow_bf16_reduced_precision_reduction": false,
  "float32_matmul_precision": "highest"
 },
 "driver": "610.47, 12227 MiB"
}
```

## dense_wc

**dense-nonsym-n2048-k1e+02**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| LU64 (direct) | 24.5 [24.3, 24.6] | 0.8 [0.8, 0.9] | 25.4 | 1 | converged | 3.96e-14 | 2.63e-13 | 64.0 / 0.1 | fp64:1 |
| FGMRES64(LU64) | 22.1 [22.1, 22.2] | 2.8 [2.7, 2.8] | 24.9 | 1 | converged | 4.21e-14 | 2.67e-13 | 64.6 / 0.8 | fp64:5 |
| IR64<-32[LU32] (plain IR) | 5.1 [5.1, 5.8] | 8.2 [8.0, 8.7] | 13.3 | 3 | converged | 5.85e-14 | 2.32e-13 | 64.0 / 0.2 | fp32:3 fp64:5 |
| FGMRES64[IR64<-32[LU32]] | 5.8 [5.7, 5.9] | 14.9 [14.0, 15.3] | 20.7 | 2 | converged | 3.46e-16 | 3.66e-15 | 64.6 / 0.8 | fp32:4 fp64:13 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-16[INV16]]]] | 69.6 [69.6, 70.4] | 20.6 [19.8, 26.2] | 90.2 | 1 | converged | 2.77e-11 | 4.43e-11 | 64.8 / 1.0 | fp16:4 fp32:16 fp64:8 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-b16[INVb16]]]] | 69.7 [69.5, 69.8] | 41.0 [38.6, 43.3] | 110.7 | 1 | converged | 1.99e-13 | 2.19e-12 | 64.8 / 1.1 | bf16:8 fp32:26 fp64:8 |


**dense-nonsym-n4096-k1e+02**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| LU64 (direct) | 121.8 [121.7, 121.9] | 2.0 [1.9, 2.1] | 123.9 | 1 | converged | 1.04e-13 | 5.96e-13 | 256.0 / 0.2 | fp64:1 |
| FGMRES64(LU64) | 121.8 [121.6, 121.9] | 4.3 [4.2, 4.6] | 126.0 | 1 | converged | 1.09e-13 | 5.92e-13 | 257.3 / 1.6 | fp64:5 |
| IR64<-32[LU32] (plain IR) | 15.3 [15.2, 15.6] | 11.4 [11.0, 11.7] | 26.7 | 3 | converged | 8.58e-13 | 2.67e-12 | 256.0 / 0.3 | fp32:3 fp64:5 |
| FGMRES64[IR64<-32[LU32]] | 14.5 [14.4, 14.9] | 18.4 [17.2, 19.3] | 33.0 | 2 | converged | 3.68e-16 | 3.78e-15 | 257.3 / 1.7 | fp32:4 fp64:13 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-16[INV16]]]] | 458.1 [458.1, 458.9] | 23.5 [22.6, 23.9] | 481.5 | 1 | converged | 2.65e-11 | 4.39e-11 | 257.6 / 2.0 | fp16:4 fp32:16 fp64:8 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-b16[INVb16]]]] | 458.1 [458.0, 458.1] | 39.5 [36.4, 49.5] | 497.6 | 1 | converged | 9.20e-13 | 1.02e-11 | 257.6 / 2.1 | bf16:8 fp32:26 fp64:8 |


**dense-nonsym-n8192-k1e+02**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| LU64 (direct) | 864.2 [863.8, 864.4] | 4.1 [4.0, 4.3] | 868.3 | 1 | converged | 2.81e-13 | 1.63e-12 | 1024.0 / 0.3 | fp64:1 |
| FGMRES64(LU64) | 867.8 [867.6, 868.3] | 11.5 [11.3, 11.6] | 879.2 | 1 | converged | 2.84e-13 | 1.64e-12 | 1026.6 / 3.1 | fp64:5 |
| IR64<-32[LU32] (plain IR) | 69.6 [68.5, 70.6] | 21.1 [20.7, 21.3] | 90.7 | 3 | converged | 1.23e-11 | 3.55e-11 | 1024.0 / 0.6 | fp32:3 fp64:5 |
| FGMRES64[IR64<-32[LU32]] | 65.9 [65.5, 66.6] | 28.7 [28.3, 29.2] | 94.6 | 1 | converged | 1.19e-11 | 3.52e-11 | 1026.6 / 3.3 | fp32:3 fp64:9 |


## dense_ic

**Robustness across condition numbers (n=2048 dense)**

| Method | kappa=1e+04 | kappa=1e+06 | kappa=1e+07 | kappa=1e+08 | kappa=1e+10 | kappa=1e+12 |
|---|---|---|---|---|---|---|
| LU64 (direct) | ok (1 it, fwd 6.19e-12) | ok (1 it, fwd 3.63e-10) | ok (1 it, fwd 1.99e-9) | ok (1 it, fwd 1.74e-8) | ok (1 it, fwd 1.18e-6) | DNC (success) |
| FGMRES64(LU64) | ok (1 it, fwd 6.57e-12) | ok (1 it, fwd 3.13e-10) | ok (1 it, fwd 2.05e-9) | ok (1 it, fwd 1.83e-8) | ok (1 it, fwd 9.89e-7) | ok (111 it, fwd 8.09e-1) |
| IR64<-32[LU32] (plain IR) | ok (3 it, fwd 2.41e-9) | ok (5 it, fwd 4.55e-7) | ok (13 it, fwd 6.19e-6) | DNC (rejected) | DNC (rejected) | DNC (rejected) |
| FGMRES64[IR64<-32[LU32]] | ok (1 it, fwd 2.68e-9) | ok (2 it, fwd 2.10e-8) | ok (5 it, fwd 3.51e-7) | DNC (max_steps) | DNC (max_steps) | DNC (max_steps) |
| FGMRES64[IR64<-32[LU32]] (loose) | ok (1 it, fwd 2.68e-9) | ok (2 it, fwd 2.10e-8) | ok (5 it, fwd 3.51e-7) | DNC (max_steps) | DNC (max_steps) | DNC (max_steps) |
| IR64<-32[LU32 | LU64 fallback] | ok (3 it, fwd 2.41e-9) | ok (5 it, fwd 4.55e-7) | ok (13 it, fwd 6.19e-6) | ok (3 it, fwd 1.43e-6) | ok (3 it, fwd 2.42e-5) | DNC (stagnated) |
| FGMRES64[IR64<-32[LU32|LU64 fb]|dir(LU64)] (ladder) | ok (2 it, fwd 2.25e-12) | ok (3 it, fwd 2.17e-8) | ok (7 it, fwd 9.39e-7) | ok (2 it, fwd 2.79e-7) | ok (2 it, fwd 2.06e-5) | ok (330 it, fwd 4.28e-1) |


**dense-nonsym-n2048-k1e+04**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| LU64 (direct) | 24.4 [24.3, 24.4] | 0.9 [0.9, 0.9] | 25.3 | 1 | converged | 2.28e-14 | 6.19e-12 | 64.0 / 0.1 | fp64:1 |
| FGMRES64(LU64) | 24.3 [22.7, 24.3] | 2.4 [2.3, 2.4] | 26.6 | 1 | converged | 2.31e-14 | 6.57e-12 | 65.0 / 1.1 | fp64:5 |
| IR64<-32[LU32] (plain IR) | 5.1 [5.1, 5.1] | 6.3 [6.2, 6.5] | 11.4 | 3 | converged | 1.56e-11 | 2.41e-9 | 64.0 / 0.2 | fp32:3 fp64:5 |
| FGMRES64[IR64<-32[LU32]] | 4.9 [4.9, 4.9] | 7.7 [7.7, 7.9] | 12.6 | 1 | converged | 1.78e-11 | 2.68e-9 | 65.0 / 1.1 | fp32:3 fp64:9 |
| FGMRES64[IR64<-32[LU32]] (loose) | 4.8 [4.8, 4.8] | 7.9 [7.7, 8.0] | 12.7 | 1 | converged | 1.78e-11 | 2.68e-9 | 65.0 / 1.1 | fp32:3 fp64:9 |
| IR64<-32[LU32 | LU64 fallback] | 26.8 [26.6, 27.5] | 6.0 [5.9, 6.1] | 32.8 | 3 | converged | 1.56e-11 | 2.41e-9 | 96.0 / 0.2 | fp32:3 fp64:5 |
| FGMRES64[IR64<-32[LU32|LU64 fb]|dir(LU64)] (ladder) | 26.7 [26.6, 26.7] | 10.9 [10.8, 11.0] | 37.6 | 2 | converged | 1.66e-14 | 2.25e-12 | 97.0 / 1.1 | fp32:4 fp64:13 |


**dense-nonsym-n2048-k1e+06**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| LU64 (direct) | 21.8 [21.8, 22.0] | 0.8 [0.8, 0.8] | 22.6 | 1 | converged | 1.69e-14 | 3.63e-10 | 64.0 / 0.1 | fp64:1 |
| FGMRES64(LU64) | 21.9 [21.8, 21.9] | 2.3 [2.3, 2.3] | 24.2 | 1 | converged | 1.68e-14 | 3.13e-10 | 65.0 / 1.1 | fp64:5 |
| IR64<-32[LU32] (plain IR) | 4.9 [4.9, 5.0] | 10.1 [10.0, 10.5] | 15.0 | 5 | converged | 4.57e-11 | 4.55e-7 | 64.0 / 0.2 | fp32:5 fp64:7 |
| FGMRES64[IR64<-32[LU32]] | 4.9 [4.8, 4.9] | 15.5 [15.1, 15.8] | 20.4 | 2 | converged | 2.22e-12 | 2.10e-8 | 65.0 / 1.1 | fp32:6 fp64:15 |
| FGMRES64[IR64<-32[LU32]] (loose) | 4.8 [4.8, 4.8] | 14.8 [14.5, 15.0] | 19.6 | 2 | converged | 2.22e-12 | 2.10e-8 | 65.0 / 1.1 | fp32:6 fp64:15 |
| IR64<-32[LU32 | LU64 fallback] | 26.9 [26.7, 27.8] | 10.2 [10.0, 10.3] | 37.1 | 5 | converged | 4.57e-11 | 4.55e-7 | 96.0 / 0.2 | fp32:5 fp64:7 |
| FGMRES64[IR64<-32[LU32|LU64 fb]|dir(LU64)] (ladder) | 26.8 [26.7, 27.0] | 16.5 [16.2, 17.1] | 43.2 | 3 | converged | 2.27e-12 | 2.17e-8 | 97.0 / 1.2 | fp32:6 fp64:18 |


**dense-nonsym-n2048-k1e+07**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| LU64 (direct) | 21.8 [21.8, 21.8] | 0.8 [0.8, 0.8] | 22.7 | 1 | converged | 1.44e-14 | 1.99e-9 | 64.0 / 0.1 | fp64:1 |
| FGMRES64(LU64) | 21.9 [21.8, 22.1] | 2.2 [2.2, 2.3] | 24.1 | 1 | converged | 1.42e-14 | 2.05e-9 | 65.0 / 1.1 | fp64:5 |
| IR64<-32[LU32] (plain IR) | 4.9 [4.9, 5.0] | 25.6 [25.3, 25.8] | 30.5 | 13 | converged | 6.52e-11 | 6.19e-6 | 64.0 / 0.3 | fp32:13 fp64:15 |
| FGMRES64[IR64<-32[LU32]] | 4.9 [4.9, 4.9] | 38.3 [38.0, 38.6] | 43.1 | 5 | converged | 5.68e-12 | 3.51e-7 | 65.0 / 1.2 | fp32:15 fp64:33 |
| FGMRES64[IR64<-32[LU32]] (loose) | 5.2 [5.0, 5.4] | 40.4 [38.2, 42.5] | 45.6 | 5 | converged | 5.68e-12 | 3.51e-7 | 65.0 / 1.2 | fp32:15 fp64:33 |
| IR64<-32[LU32 | LU64 fallback] | 27.2 [27.2, 28.2] | 25.6 [25.5, 25.9] | 52.8 | 13 | converged | 6.52e-11 | 6.19e-6 | 96.0 / 0.3 | fp32:13 fp64:15 |
| FGMRES64[IR64<-32[LU32|LU64 fb]|dir(LU64)] (ladder) | 26.9 [26.8, 26.9] | 39.1 [38.6, 40.5] | 66.0 | 7 | converged | 1.24e-11 | 9.39e-7 | 97.0 / 1.2 | fp32:14 fp64:38 |


**dense-nonsym-n2048-k1e+08**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| LU64 (direct) | 21.9 [21.9, 22.1] | 0.8 [0.8, 0.8] | 22.7 | 1 | converged | 1.44e-14 | 1.74e-8 | 64.0 / 0.1 | fp64:1 |
| FGMRES64(LU64) | 21.9 [21.8, 21.9] | 2.3 [2.2, 2.3] | 24.1 | 1 | converged | 1.31e-14 | 1.83e-8 | 65.0 / 1.1 | fp64:5 |
| IR64<-32[LU32] (plain IR) | 5.0 [4.9, 5.0] | 4.3 [4.1, 4.3] | 9.2 | 2 | DNC (rejected) | 1.56e-3 | 3.69e+2 | 64.0 / 0.2 | fp32:2 fp64:4 |
| FGMRES64[IR64<-32[LU32]] | 4.9 [4.9, 5.0] | 2520.7 [2450.8, 3278.6] | 2525.7 | 400 | DNC (max_steps) | 1.12e-4 | 4.91e+1 | 65.0 / 0.9 | fp32:497 fp64:1713 |
| FGMRES64[IR64<-32[LU32]] (loose) | 5.0 [5.0, 5.9] | 3947.5 [3605.4, 4011.5] | 3952.5 | 400 | DNC (max_steps) | 1.51e-2 | 2.01e+3 | 65.0 / 0.8 | fp32:801 fp64:2017 |
| IR64<-32[LU32 | LU64 fallback] | 27.1 [27.0, 29.1] | 6.3 [6.2, 6.5] | 33.4 | 3 | converged | 2.28e-12 | 1.43e-6 | 96.0 / 0.2 | fp32:2 fp64:6 |
| FGMRES64[IR64<-32[LU32|LU64 fb]|dir(LU64)] (ladder) | 27.0 [27.0, 27.1] | 12.3 [12.1, 14.8] | 39.3 | 2 | converged | 4.82e-13 | 2.79e-7 | 97.0 / 1.2 | fp32:3 fp64:14 |


**dense-nonsym-n2048-k1e+10**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| LU64 (direct) | 22.1 [22.1, 22.4] | 0.9 [0.8, 0.9] | 22.9 | 1 | converged | 8.65e-15 | 1.18e-6 | 64.0 / 0.1 | fp64:1 |
| FGMRES64(LU64) | 22.0 [21.9, 22.1] | 2.6 [2.6, 2.7] | 24.6 | 1 | converged | 9.13e-15 | 9.89e-7 | 65.0 / 1.1 | fp64:5 |
| IR64<-32[LU32] (plain IR) | 5.2 [5.1, 5.2] | 4.9 [4.7, 5.1] | 10.1 | 2 | DNC (rejected) | 2.91e-4 | 8.52e+1 | 64.0 / 0.2 | fp32:2 fp64:4 |
| FGMRES64[IR64<-32[LU32]] | 5.0 [4.9, 5.0] | 3227.7 [3182.7, 3325.6] | 3232.7 | 400 | DNC (max_steps) | 6.09e-5 | 7.42e+1 | 65.0 / 0.9 | fp32:516 fp64:1732 |
| FGMRES64[IR64<-32[LU32]] (loose) | 5.0 [5.0, 5.0] | 3895.1 [3858.3, 4091.5] | 3900.0 | 400 | DNC (max_steps) | 5.21e-3 | 1.41e+3 | 65.0 / 0.8 | fp32:804 fp64:2020 |
| IR64<-32[LU32 | LU64 fallback] | 27.1 [27.1, 28.8] | 6.8 [6.3, 8.4] | 33.9 | 3 | converged | 4.69e-13 | 2.42e-5 | 96.0 / 0.2 | fp32:2 fp64:6 |
| FGMRES64[IR64<-32[LU32|LU64 fb]|dir(LU64)] (ladder) | 27.1 [26.9, 27.3] | 12.0 [11.6, 12.3] | 39.0 | 2 | converged | 3.70e-13 | 2.06e-5 | 97.0 / 1.2 | fp32:3 fp64:14 |


**dense-nonsym-n2048-k1e+12**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| LU64 (direct) | 22.7 [22.6, 22.8] | 0.9 [0.8, 0.9] | 23.6 | 1 | DNC (success) | 7.76e-9 | 8.27e+0 | 64.0 / 0.1 | fp64:1 |
| FGMRES64(LU64) | 22.9 [22.9, 23.1] | 559.6 [535.4, 591.5] | 582.4 | 111 | converged | 9.97e-11 | 8.09e-1 | 65.0 / 1.5 | fp64:228 |
| IR64<-32[LU32] (plain IR) | 4.9 [4.9, 5.0] | 5.2 [4.9, 5.7] | 10.2 | 2 | DNC (rejected) | 9.84e-4 | 4.74e+2 | 64.0 / 0.2 | fp32:2 fp64:4 |
| FGMRES64[IR64<-32[LU32]] | 5.3 [5.3, 5.4] | 2456.3 [2440.5, 2497.8] | 2461.6 | 400 | DNC (max_steps) | 1.94e-4 | 3.78e+2 | 65.0 / 0.9 | fp32:508 fp64:1724 |
| FGMRES64[IR64<-32[LU32]] (loose) | 4.9 [4.9, 5.0] | 3071.2 [3058.4, 3102.2] | 3076.1 | 400 | DNC (max_steps) | 1.16e-2 | 2.60e+3 | 65.0 / 0.8 | fp32:802 fp64:2018 |
| IR64<-32[LU32 | LU64 fallback] | 27.2 [27.1, 28.2] | 41.6 [41.4, 41.8] | 68.7 | 32 | DNC (stagnated) | 9.13e-7 | 1.51e+3 | 96.0 / 0.7 | fp32:2 fp64:64 |
| FGMRES64[IR64<-32[LU32|LU64 fb]|dir(LU64)] (ladder) | 27.6 [27.4, 27.7] | 1422.5 [1413.0, 1426.6] | 1450.1 | 330 | converged | 8.89e-11 | 4.28e-1 | 97.0 / 1.5 | fp32:9 fp64:705 |


## sparse_spd

**poisson3d-g32**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| CG64 | 0.0 [0.0, 0.0] | 53.4 [52.7, 54.1] | 53.4 | 130 | converged | 9.65e-11 | 1.34e-9 | 2.7 / 1.8 | fp64:131 |
| CG64(Jacobi64) | 0.1 [0.1, 0.2] | 77.2 [76.7, 78.4] | 77.3 | 130 | converged | 9.65e-11 | 1.34e-9 | 2.7 / 2.3 | fp64:261 |
| FGMRES64(Jacobi64) | 0.1 [0.1, 0.2] | 719.0 [718.2, 720.5] | 719.1 | 201 | converged | 9.64e-11 | 1.82e-8 | 17.0 / 24.5 | fp64:411 |
| IR64<-32[CG32(Jacobi32)] (plain IR) | 0.4 [0.4, 0.5] | 107.7 [107.5, 107.9] | 108.1 | 3 | converged | 8.96e-13 | 4.02e-11 | 2.6 / 2.5 | fp32:347 fp64:5 |
| FGMRES64[IR64<-32[CG32(Jacobi32)]] | 0.2 [0.2, 0.2] | 154.6 [152.8, 156.1] | 154.7 | 2 | converged | 3.08e-16 | 1.54e-15 | 17.8 / 18.4 | fp32:494 fp64:13 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-16[FGMRES16(Jacobi16)]]]] | 0.5 [0.5, 0.6] | 921.3 [914.6, 930.5] | 921.9 | 2 | converged | 2.07e-16 | 4.14e-16 | 24.7 / 26.3 | fp16:900 fp32:67 fp64:13 |


**poisson3d-g48**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| CG64 | 0.0 [0.0, 0.0] | 73.6 [72.9, 74.2] | 73.6 | 180 | converged | 9.42e-11 | 1.94e-9 | 9.2 / 5.9 | fp64:181 |
| CG64(Jacobi64) | 0.1 [0.1, 0.2] | 107.8 [107.6, 108.1] | 107.9 | 180 | converged | 9.42e-11 | 1.94e-9 | 9.2 / 7.6 | fp64:361 |
| FGMRES64(Jacobi64) | 0.1 [0.1, 0.1] | 1028.2 [1024.0, 1036.4] | 1028.3 | 292 | converged | 9.84e-11 | 4.34e-8 | 57.3 / 82.5 | fp64:596 |
| IR64<-32[CG32(Jacobi32)] (plain IR) | 0.9 [0.8, 0.9] | 169.0 [168.2, 170.5] | 169.9 | 3 | converged | 8.79e-13 | 2.78e-11 | 8.7 / 8.4 | fp32:529 fp64:5 |
| FGMRES64[IR64<-32[CG32(Jacobi32)]] | 0.2 [0.2, 0.2] | 245.3 [244.1, 246.5] | 245.5 | 2 | converged | 2.11e-16 | 3.43e-15 | 60.2 / 62.7 | fp32:758 fp64:13 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-16[FGMRES16(Jacobi16)]]]] | 0.9 [0.9, 1.0] | 1353.4 [1342.0, 1357.0] | 1354.3 | 2 | converged | 2.37e-16 | 6.15e-16 | 83.3 / 90.9 | fp16:1250 fp32:87 fp64:13 |


**poisson3d-g64**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| CG64 | 0.0 [0.0, 0.0] | 97.5 [96.7, 101.0] | 97.5 | 236 | converged | 9.49e-11 | 4.26e-9 | 21.8 / 14.7 | fp64:237 |
| CG64(Jacobi64) | 0.1 [0.1, 0.2] | 154.2 [153.8, 155.3] | 154.4 | 236 | converged | 9.49e-11 | 4.26e-9 | 21.8 / 18.7 | fp64:473 |
| FGMRES64(Jacobi64) | 0.1 [0.1, 0.2] | 1439.7 [1433.9, 1448.7] | 1439.8 | 383 | converged | 9.93e-11 | 8.10e-8 | 135.8 / 194.7 | fp64:781 |
| IR64<-32[CG32(Jacobi32)] (plain IR) | 1.6 [1.6, 2.4] | 217.9 [217.2, 218.9] | 219.5 | 3 | converged | 8.91e-13 | 5.04e-11 | 20.7 / 20.0 | fp32:693 fp64:5 |
| FGMRES64[IR64<-32[CG32(Jacobi32)]] | 0.2 [0.2, 0.2] | 319.0 [314.0, 321.7] | 319.2 | 2 | converged | 5.85e-16 | 2.60e-15 | 142.7 / 147.0 | fp32:1012 fp64:13 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-16[FGMRES16(Jacobi16)]]]] | 1.6 [1.6, 1.8] | 1584.4 [1577.0, 1602.1] | 1586.0 | 2 | converged | 2.42e-16 | 8.77e-16 | 197.7 / 210.5 | fp16:1518 fp32:102 fp64:13 |


## sparse_nsy

**convdiff3d-g32-w2**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| GMRES64(Jacobi64) | 0.2 [0.2, 0.2] | 530.1 [521.8, 548.1] | 530.3 | 155 | converged | 9.81e-11 | 3.62e-9 | 9.5 / 16.5 | fp64:324 |
| FGMRES64(Jacobi64) | 0.2 [0.1, 0.2] | 555.0 [552.1, 558.5] | 555.2 | 155 | converged | 9.81e-11 | 3.62e-9 | 17.0 / 24.5 | fp64:318 |
| FGMRES64[IR64<-32[FGMRES32(Jacobi32)]] | 0.5 [0.4, 0.5] | 805.0 [802.6, 811.5] | 805.4 | 2 | converged | 2.42e-16 | 4.66e-16 | 25.4 / 30.4 | fp32:475 fp64:13 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-16[FGMRES16(Jacobi16)]]]] | 0.5 [0.5, 0.6] | 357.9 [354.5, 362.9] | 358.4 | 1 | converged | 1.35e-12 | 1.61e-11 | 24.7 / 25.9 | fp16:356 fp32:31 fp64:8 |


**convdiff3d-g32-w20**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| GMRES64(Jacobi64) | 0.1 [0.1, 0.2] | 419.3 [416.2, 427.8] | 419.4 | 122 | converged | 9.55e-11 | 5.15e-10 | 9.5 / 16.5 | fp64:256 |
| FGMRES64(Jacobi64) | 0.1 [0.1, 0.2] | 440.6 [438.6, 444.3] | 440.7 | 122 | converged | 9.55e-11 | 5.15e-10 | 17.0 / 24.5 | fp64:251 |
| FGMRES64[IR64<-32[FGMRES32(Jacobi32)]] | 0.5 [0.5, 0.6] | 578.0 [569.9, 582.1] | 578.5 | 2 | converged | 3.21e-16 | 4.73e-16 | 25.4 / 29.5 | fp32:342 fp64:13 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-16[FGMRES16(Jacobi16)]]]] | 0.5 [0.5, 0.5] | 264.3 [263.5, 266.2] | 264.8 | 1 | converged | 4.99e-11 | 1.79e-10 | 24.7 / 26.6 | fp16:262 fp32:26 fp64:8 |


**convdiff3d-g32-w100**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| GMRES64(Jacobi64) | 0.1 [0.1, 0.1] | 625.1 [619.1, 627.9] | 625.3 | 179 | converged | 8.44e-11 | 6.16e-10 | 9.5 / 16.5 | fp64:372 |
| FGMRES64(Jacobi64) | 0.1 [0.1, 0.2] | 648.0 [646.9, 651.0] | 648.1 | 179 | converged | 8.44e-11 | 6.16e-10 | 17.0 / 24.5 | fp64:366 |
| FGMRES64[IR64<-32[FGMRES32(Jacobi32)]] | 0.5 [0.4, 0.5] | 678.4 [674.1, 698.5] | 678.9 | 2 | converged | 1.61e-16 | 5.34e-16 | 25.4 / 29.8 | fp32:390 fp64:13 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-16[FGMRES16(Jacobi16)]]]] | 0.5 [0.5, 0.6] | 544.9 [529.1, 693.6] | 545.5 | 2 | converged | 2.54e-16 | 5.73e-16 | 24.7 / 26.1 | fp16:534 fp32:52 fp64:13 |


**convdiff3d-g48-w20**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| GMRES64(Jacobi64) | 0.2 [0.2, 0.4] | 757.0 [749.0, 782.6] | 757.2 | 175 | converged | 9.23e-11 | 1.01e-9 | 32.0 / 55.7 | fp64:364 |
| FGMRES64(Jacobi64) | 0.2 [0.2, 0.3] | 792.3 [782.4, 904.5] | 792.5 | 175 | converged | 9.23e-11 | 1.01e-9 | 57.3 / 82.5 | fp64:358 |
| FGMRES64[IR64<-32[FGMRES32(Jacobi32)]] | 1.1 [1.1, 1.2] | 1224.9 [1156.5, 1390.4] | 1225.9 | 2 | converged | 2.91e-16 | 4.70e-16 | 85.9 / 100.8 | fp32:538 fp64:13 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-16[FGMRES16(Jacobi16)]]]] | 1.1 [1.0, 1.1] | 535.9 [517.9, 595.1] | 537.0 | 1 | converged | 3.00e-11 | 1.59e-10 | 83.3 / 90.1 | fp16:403 fp32:31 fp64:8 |


## placement

**dense-nonsym-n2048-k1e+02**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| FGMRES64[IR64<-32[INV32]] | 69.9 [69.8, 70.0] | 6.7 [6.1, 8.6] | 76.6 | 1 | converged | 9.64e-12 | 9.93e-12 | 64.6 / 0.8 | fp32:2 fp64:8 |
| FGMRES64[IR64<-16[INV16]] | 70.0 [70.0, 70.1] | 28.3 [27.3, 29.2] | 98.4 | 2 | converged | 5.30e-16 | 3.55e-15 | 48.6 / 0.8 | fp16:6 fp64:15 |
| FGMRES64[IR64<-b16[INVb16]] | 70.2 [70.1, 70.3] | 17.3 [16.2, 18.2] | 87.4 | 2 | converged | 3.93e-11 | 5.59e-11 | 48.6 / 0.8 | bf16:6 fp64:15 |
| FGMRES64[IR64<-32[IR32<-16[INV16]]] | 70.2 [69.9, 70.2] | 26.8 [24.7, 28.5] | 97.0 | 1 | converged | 2.59e-11 | 3.83e-11 | 64.6 / 0.8 | fp16:4 fp32:8 fp64:8 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-16[INV16]]]] | 70.3 [70.1, 70.6] | 28.8 [18.8, 30.6] | 99.1 | 1 | converged | 2.60e-11 | 4.06e-11 | 64.8 / 1.0 | fp16:4 fp32:16 fp64:8 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-b16[INVb16]]]] | 70.1 [70.0, 70.3] | 29.4 [29.2, 29.7] | 99.5 | 1 | converged | 1.90e-13 | 2.04e-12 | 64.8 / 1.1 | bf16:8 fp32:26 fp64:8 |


**dense-nonsym-n2048-k1e+04**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| FGMRES64[IR64<-32[INV32]] | 69.8 [69.7, 69.9] | 7.9 [7.7, 8.1] | 77.7 | 1 | converged | 4.01e-12 | 4.03e-12 | 64.6 / 0.8 | fp32:3 fp64:9 |
| FGMRES64[IR64<-16[INV16]] | 69.8 [69.8, 70.0] | 31.7 [31.2, 32.7] | 101.5 | 4 | converged | 3.09e-12 | 4.55e-12 | 48.6 / 0.8 | fp16:12 fp64:27 |
| FGMRES64[IR64<-b16[INVb16]] | 70.0 [69.8, 70.1] | 582.0 [571.0, 602.0] | 652.0 | 60 | converged | 8.09e-11 | 1.65e-10 | 48.6 / 0.8 | bf16:180 fp64:365 |
| FGMRES64[IR64<-32[IR32<-16[INV16]]] | 70.1 [69.9, 70.2] | 42.5 [41.6, 47.4] | 112.6 | 3 | converged | 2.97e-12 | 4.44e-12 | 64.6 / 0.8 | fp16:12 fp32:24 fp64:18 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-16[INV16]]]] | 71.1 [71.0, 71.2] | 41.7 [41.3, 43.9] | 112.9 | 1 | converged | 9.92e-12 | 4.92e-9 | 64.8 / 1.1 | fp16:12 fp32:36 fp64:8 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-b16[INVb16]]]] | 71.0 [70.9, 71.0] | 857.6 [840.8, 880.0] | 928.5 | 3 | converged | 5.93e-11 | 1.06e-10 | 64.8 / 1.1 | bf16:239 fp32:623 fp64:18 |


**dense-nonsym-n2048-k1e+06**

| Method | Setup ms (med [IQR]) | Solve ms (med [IQR]) | Total | Outer | Status | rel. resid | fwd err | Mem MiB (analytic/peak) | matvecs |
|---|---|---|---|---|---|---|---|---|---|
| FGMRES64[IR64<-32[INV32]] | 70.0 [69.9, 70.2] | 15.2 [15.0, 15.4] | 85.2 | 2 | converged | 1.15e-12 | 1.13e-11 | 64.6 / 0.8 | fp32:6 fp64:15 |
| FGMRES64[IR64<-16[INV16]] | 70.1 [69.9, 70.3] | 1697.9 [1658.9, 1735.1] | 1768.0 | 300 | DNC (max_steps) | 9.20e-1 | 1.09e+0 | 48.6 / 0.9 | fp16:300 fp64:1217 |
| FGMRES64[IR64<-b16[INVb16]] | 70.0 [70.0, 70.2] | 1806.1 [1713.2, 1886.2] | 1876.1 | 300 | DNC (max_steps) | 9.20e-1 | 1.09e+0 | 48.6 / 0.9 | bf16:300 fp64:1217 |
| FGMRES64[IR64<-32[IR32<-16[INV16]]] | 70.0 [70.0, 70.1] | 3698.3 [3505.7, 3880.6] | 3768.3 | 300 | DNC (max_steps) | 9.20e-1 | 1.09e+0 | 64.6 / 0.9 | fp16:600 fp32:1800 fp64:1517 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-16[INV16]]]] | 69.9 [69.9, 70.5] | 61964.0 [59415.0, 63880.6] | 62033.9 | 300 | DNC (max_steps) | 8.58e-1 | 1.14e+0 | 64.8 / 1.1 | fp16:12000 fp32:50400 fp64:1517 |
| FGMRES64[IR64<-32[FGMRES32[IR32<-b16[INVb16]]]] | 70.0 [69.7, 70.2] | 64135.9 [51096.6, 64918.2] | 64206.0 | 300 | DNC (max_steps) | 8.58e-1 | 1.14e+0 | 64.8 / 1.1 | bf16:12000 fp32:50400 fp64:1517 |

